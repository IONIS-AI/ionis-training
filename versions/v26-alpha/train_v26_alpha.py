#!/usr/bin/env python3
"""
train_v26_alpha.py — IONIS V26-alpha Training

V26 hypothesis: The shared output head forces the optimizer to compromise
between bands with opposite physics (10m needs darkness penalty, 160m needs
darkness bonus). Replace single base_head with 9 band-specific output heads.

Key change from V22-gamma:
    - base_head (256→128→1) → band_heads (9 × 256→128→1)
    - band_idx feature at index 17 for routing (NOT fed to trunk)
    - Forward pass routes through band-specific head via gather()
    - Parameters: ~205K → ~468K

Everything else IDENTICAL to V22-gamma:
    - Trunk: 15 → 512 → 256
    - Sun sidecar: MonotonicMLP(1→8→1), Defibrillator, clamp [0.5, 2.0]
    - Storm sidecar: MonotonicMLP(1→8→1), Defibrillator, clamp [0.5, 2.0]
    - Gates, gate variance loss, HuberLoss(delta=1.0)
    - Data recipe: 20M WSPR + 13M DX (50x) + 5.7M Contest

Acid test: DN13→JN48, 10m, 02 UTC, Feb, SFI=150 must output ≤ 0.0σ
(V22-gamma outputs +3.7σ for this case)

Architecture constraints (V16 Physics Laws):
    1. Gates from trunk output (256-dim), not raw input
    2. HuberLoss(delta=1.0) — robust to synthetic anchors
    3. Gate variance loss — forces context sensitivity
    4. Defibrillator init — weights uniform(0.8-1.2), fc2.bias=-10
    5. Weight clamp [0.5, 2.0] after EVERY step
"""

import gc
import json
import logging
import math
import os
import socket
import sys
import time
import uuid

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from safetensors.torch import save_file as save_safetensors

import clickhouse_connect

# Add parent for common imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VERSIONS_DIR = os.path.dirname(SCRIPT_DIR)
COMMON_DIR = os.path.join(VERSIONS_DIR, "common")
sys.path.insert(0, VERSIONS_DIR)
sys.path.insert(0, COMMON_DIR)

from common.train_common import (
    MonotonicMLP,
    SignatureDataset,
    engineer_features,
    init_defibrillator,
    clamp_sidecars,
    load_source_data,
    log_config,
    grid4_to_latlon_arrays,
    solar_elevation_vectorized,
)
from common.model import _gate

# ── Load Configuration ────────────────────────────────────────────────────────

CONFIG_FILE = os.path.join(SCRIPT_DIR, "config_v26_alpha.json")

with open(CONFIG_FILE) as f:
    CONFIG = json.load(f)

RESULTS_DIR = os.path.join(VERSIONS_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_FILE = os.path.join(RESULTS_DIR, f"v26_alpha_{time.strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ],
)
log = logging.getLogger(f"ionis-{CONFIG['version']}")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ── IonisGateV26 Architecture ─────────────────────────────────────────────────

class IonisGateV26(nn.Module):
    """
    IONIS V26 — Band-specific output heads for physics-constrained SNR prediction.

    Key change from IonisGate (V22-gamma):
        - Single base_head replaced with 9 band_heads (one per HF band)
        - Each head: 256→128→1 (same topology as V22-gamma base_head)
        - band_idx feature routes to the correct head via gather()

    The hypothesis: The shared output head forced the optimizer to compromise
    between bands with opposite physics. Now each band head can learn its own
    darkness response without affecting other bands.

    Args:
        dnn_dim: Number of geography/time features (default 15, indices 0-14)
        sidecar_hidden: Hidden units in MonotonicMLP (default 8)
        sfi_idx: Index of SFI feature in input (default 15)
        kp_penalty_idx: Index of Kp penalty feature in input (default 16)
        band_idx: Index of band routing feature in input (default 17)
        num_bands: Number of band-specific heads (default 9)
        gate_init_bias: Initial bias for scaler heads (default -ln(2))
    """

    def __init__(self, dnn_dim=15, sidecar_hidden=8, sfi_idx=15, kp_penalty_idx=16,
                 band_idx=17, num_bands=9, gate_init_bias=None):
        super().__init__()

        if gate_init_bias is None:
            gate_init_bias = -math.log(2.0)

        self.dnn_dim = dnn_dim
        self.sfi_idx = sfi_idx
        self.kp_penalty_idx = kp_penalty_idx
        self.band_idx = band_idx
        self.num_bands = num_bands

        # Trunk: IDENTICAL to V22-gamma (15→512→256)
        self.trunk = nn.Sequential(
            nn.Linear(dnn_dim, 512), nn.Mish(),
            nn.Linear(512, 256), nn.Mish(),
        )

        # V26 CHANGE: 9 band-specific output heads replace single base_head
        # Each head: 256→128→1 (same topology as V22-gamma base_head)
        self.band_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128), nn.Mish(),
                nn.Linear(128, 1),
            )
            for _ in range(num_bands)
        ])

        # Gate heads: IDENTICAL to V22-gamma
        self.sun_scaler_head = nn.Sequential(
            nn.Linear(256, 64), nn.Mish(),
            nn.Linear(64, 1),
        )
        self.storm_scaler_head = nn.Sequential(
            nn.Linear(256, 64), nn.Mish(),
            nn.Linear(64, 1),
        )

        # Physics sidecars: IDENTICAL to V22-gamma
        self.sun_sidecar = MonotonicMLP(hidden_dim=sidecar_hidden)
        self.storm_sidecar = MonotonicMLP(hidden_dim=sidecar_hidden)

        self._init_scaler_heads(gate_init_bias)

    def _init_scaler_heads(self, gate_init_bias):
        """Initialize scaler head biases for balanced gates."""
        for head in [self.sun_scaler_head, self.storm_scaler_head]:
            final_layer = head[-1]
            nn.init.zeros_(final_layer.weight)
            nn.init.constant_(final_layer.bias, gate_init_bias)

    def forward(self, x):
        x_deep = x[:, :self.dnn_dim]
        x_sfi = x[:, self.sfi_idx:self.sfi_idx + 1]
        x_kp = x[:, self.kp_penalty_idx:self.kp_penalty_idx + 1]
        x_band = x[:, self.band_idx].long()

        trunk_out = self.trunk(x_deep)

        # V26: Route through band-specific head
        # Compute all heads, then gather the correct one per sample
        all_heads = torch.cat([h(trunk_out) for h in self.band_heads], dim=1)  # [batch, 9]
        base_snr = all_heads.gather(1, x_band.unsqueeze(1))  # [batch, 1]

        # Gates and sidecars: IDENTICAL to V22-gamma
        sun_logit = self.sun_scaler_head(trunk_out)
        storm_logit = self.storm_scaler_head(trunk_out)
        sun_gate = _gate(sun_logit)
        storm_gate = _gate(storm_logit)

        return base_snr + sun_gate * self.sun_sidecar(x_sfi) + \
               storm_gate * self.storm_sidecar(x_kp)

    def forward_with_gates(self, x):
        """Forward pass returning gate values for variance loss."""
        x_deep = x[:, :self.dnn_dim]
        x_sfi = x[:, self.sfi_idx:self.sfi_idx + 1]
        x_kp = x[:, self.kp_penalty_idx:self.kp_penalty_idx + 1]
        x_band = x[:, self.band_idx].long()

        trunk_out = self.trunk(x_deep)

        all_heads = torch.cat([h(trunk_out) for h in self.band_heads], dim=1)
        base_snr = all_heads.gather(1, x_band.unsqueeze(1))

        sun_logit = self.sun_scaler_head(trunk_out)
        storm_logit = self.storm_scaler_head(trunk_out)
        sun_gate = _gate(sun_logit)
        storm_gate = _gate(storm_logit)

        sun_boost = self.sun_sidecar(x_sfi)
        storm_boost = self.storm_sidecar(x_kp)

        return base_snr + sun_gate * sun_boost + storm_gate * storm_boost, \
               sun_gate, storm_gate

    def get_sun_effect(self, sfi_normalized, device):
        """Get raw sun sidecar output for a given SFI value."""
        with torch.no_grad():
            x = torch.tensor([[sfi_normalized]], dtype=torch.float32, device=device)
            return self.sun_sidecar(x).item()

    def get_storm_effect(self, kp_penalty, device):
        """Get raw storm sidecar output for a given Kp penalty value."""
        with torch.no_grad():
            x = torch.tensor([[kp_penalty]], dtype=torch.float32, device=device)
            return self.storm_sidecar(x).item()

    def get_gates(self, x):
        """Get gate values without gradient tracking."""
        x_deep = x[:, :self.dnn_dim]
        with torch.no_grad():
            trunk_out = self.trunk(x_deep)
            sun_logit = self.sun_scaler_head(trunk_out)
            storm_logit = self.storm_scaler_head(trunk_out)
        return _gate(sun_logit), _gate(storm_logit)


# ── V26 Feature Engineering ───────────────────────────────────────────────────

def engineer_features_v26(df, config):
    """
    Compute features for V26 (V22-gamma features + band_idx).

    band_idx is the routing signal — NOT fed to trunk, only used in forward pass.
    dnn_dim remains 15 (features 0-14).

    Returns:
        np.ndarray of shape (n_rows, 18)
        - Features 0-14: V22-gamma DNN features
        - Feature 15: sfi
        - Feature 16: kp_penalty
        - Feature 17: band_idx (0-8)
    """
    # Get base V22-gamma features (17D: 15 DNN + SFI + Kp)
    X_base = engineer_features(df, config)

    # Add band_idx column (feature 17)
    band = df['band'].values.astype(np.int32)
    band_to_idx = config["band_to_idx"]
    if isinstance(list(band_to_idx.keys())[0], str):
        band_to_idx = {int(k): v for k, v in band_to_idx.items()}

    band_idx = np.array([band_to_idx.get(b, 0) for b in band], dtype=np.float32)

    # Append band_idx as column 17
    X = np.column_stack([X_base, band_idx])

    return X


# ── Training Audit Trail ─────────────────────────────────────────────────────

AUDIT_HOST = "http://10.60.1.1:8123"

def audit_start(run_id: str, config: dict):
    """Record training run start in ClickHouse."""
    try:
        config_json = json.dumps(config).replace('"', '\\"')
        features_array = "['" + "','".join(config["features"]) + "']"
        query = f"""
            INSERT INTO training.runs (
                run_id, version, variant, status, hostname,
                started_at, config_json,
                dnn_dim, hidden_dim, input_dim, training_epochs, batch_size,
                trunk_lr, sidecar_lr,
                wspr_sample, rbn_full_sample, rbn_dx_upsample, contest_upsample,
                features
            ) VALUES (
                '{run_id}', '{config["version"]}', '{config.get("variant", "alpha")}', 'running', '{socket.gethostname()}',
                now(),
                '{config_json}',
                {config["model"]["dnn_dim"]}, {config["model"]["hidden_dim"]},
                {config["model"]["input_dim"]}, {config["training"]["epochs"]},
                {config["training"]["batch_size"]},
                {config["training"]["trunk_lr"]}, {config["training"]["sidecar_lr"]},
                {config["data"]["wspr_sample"]}, {config["data"].get("rbn_full_sample", 0)},
                {config["data"]["rbn_dx_upsample"]}, {config["data"]["contest_upsample"]},
                {features_array}
            )
        """
        requests.post(AUDIT_HOST, params={"query": query}, timeout=5)
    except Exception as e:
        log.warning(f"Audit start failed: {e}")


def audit_epoch(run_id: str, epoch: int, train_loss: float, val_loss: float,
                val_rmse: float, val_pearson: float, sfi_benefit: float,
                storm_cost: float, epoch_seconds: float, is_best: bool):
    """Record epoch metrics in ClickHouse."""
    try:
        query = f"""
            INSERT INTO training.epochs (
                run_id, epoch, train_loss, val_loss, val_rmse, val_pearson,
                sfi_benefit, storm_cost, epoch_seconds, is_best
            ) VALUES (
                '{run_id}', {epoch}, {train_loss}, {val_loss}, {val_rmse},
                {val_pearson}, {sfi_benefit}, {storm_cost}, {epoch_seconds}, {1 if is_best else 0}
            )
        """
        requests.post(AUDIT_HOST, params={"query": query}, timeout=5)
    except Exception:
        pass


def audit_complete(run_id: str, config: dict, train_rows: int, val_rows: int,
                   date_range: str, val_rmse: float, val_pearson: float,
                   sfi_benefit: float, storm_cost: float, best_epoch: int,
                   wall_seconds: float):
    """Record training completion in ClickHouse."""
    try:
        config_json = json.dumps(config).replace('"', '\\"')
        features_array = "['" + "','".join(config["features"]) + "']"
        date_min, date_max = date_range.split(" to ")
        query = f"""
            INSERT INTO training.runs (
                run_id, version, variant, status, hostname,
                started_at, completed_at, wall_seconds,
                config_json,
                dnn_dim, hidden_dim, input_dim, training_epochs, batch_size,
                trunk_lr, sidecar_lr,
                wspr_sample, rbn_full_sample, rbn_dx_upsample, contest_upsample,
                total_train_rows, total_val_rows, data_date_min, data_date_max,
                features,
                val_rmse, val_pearson, sfi_benefit, storm_cost, best_epoch
            ) VALUES (
                '{run_id}', '{config["version"]}', '{config.get("variant", "alpha")}', 'completed', '{socket.gethostname()}',
                now() - INTERVAL {int(wall_seconds)} SECOND, now(), {wall_seconds},
                '{config_json}',
                {config["model"]["dnn_dim"]}, {config["model"]["hidden_dim"]},
                {config["model"]["input_dim"]}, {config["training"]["epochs"]},
                {config["training"]["batch_size"]},
                {config["training"]["trunk_lr"]}, {config["training"]["sidecar_lr"]},
                {config["data"]["wspr_sample"]}, {config["data"].get("rbn_full_sample", 0)},
                {config["data"]["rbn_dx_upsample"]}, {config["data"]["contest_upsample"]},
                {train_rows}, {val_rows}, '{date_min}', '{date_max}',
                {features_array},
                {val_rmse}, {val_pearson}, {sfi_benefit}, {storm_cost}, {best_epoch}
            )
        """
        requests.post(AUDIT_HOST, params={"query": query}, timeout=5)
    except Exception as e:
        log.warning(f"Audit complete failed: {e}")


# ── Per-Band Normalization ────────────────────────────────────────────────────

def normalize_snr_per_band(df, source, norm_constants):
    """Apply per-source per-band Z-score normalization."""
    snr = df['median_snr'].values.astype(np.float32).copy()
    band = df['band'].values

    for band_str, sources in norm_constants.items():
        b = int(band_str)
        mask = band == b
        if mask.sum() > 0 and source in sources:
            mean = sources[source]['mean']
            std = sources[source]['std']
            snr[mask] = (snr[mask] - mean) / std

    return snr


def pearson_r(pred, target):
    """Pearson correlation coefficient."""
    p = pred.flatten()
    t = target.flatten()
    pm, tm = p.mean(), t.mean()
    num = ((p - pm) * (t - tm)).sum()
    den = torch.sqrt(((p - pm) ** 2).sum() * ((t - tm) ** 2).sum())
    return (num / den).item() if den > 0 else 0.0


# ── V26 Optimizer Groups ──────────────────────────────────────────────────────

def get_optimizer_groups_v26(model, trunk_lr=1e-5, scaler_lr=5e-5, sidecar_lr=1e-3):
    """
    Get optimizer configuration for V26.

    V26 change: base_head replaced with band_heads (9 separate heads).
    All band_heads learn at trunk_lr (same as V22-gamma's base_head).
    """
    return [
        {'params': model.trunk.parameters(), 'lr': trunk_lr},
        # V26: All band heads instead of single base_head
        {'params': [p for head in model.band_heads for p in head.parameters()], 'lr': trunk_lr},
        {'params': model.sun_scaler_head.parameters(), 'lr': scaler_lr},
        {'params': model.storm_scaler_head.parameters(), 'lr': scaler_lr},
        {'params': [p for p in model.sun_sidecar.parameters() if p.requires_grad],
         'lr': sidecar_lr},
        {'params': [p for p in model.storm_sidecar.parameters() if p.requires_grad],
         'lr': sidecar_lr},
    ]


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    """Main training function."""
    run_id = str(uuid.uuid4())
    training_start_time = time.perf_counter()

    log_config(CONFIG, CONFIG_FILE, DEVICE)
    log.info("")
    log.info(">>> V26 ARCHITECTURE: 9 BAND-SPECIFIC OUTPUT HEADS <<<")
    log.info(">>> Testing hypothesis: shared output head caused +3.7σ defect <<<")
    log.info("")

    audit_start(run_id, CONFIG)

    # ── Load Data ──
    ch = CONFIG["clickhouse"]
    client = clickhouse_connect.get_client(host=ch["host"], port=ch["port"])

    wspr_df = load_source_data(client, "wspr.signatures_v2_terrestrial",
                               CONFIG["data"]["wspr_sample"],
                               include_day_of_year=True)
    rbn_dx_df = load_source_data(client, "rbn.dxpedition_signatures", None,
                                  include_day_of_year=True)
    contest_df = load_source_data(client, "contest.signatures", None,
                                   include_day_of_year=True)

    date_result = client.query("""
        SELECT formatDateTime(min(timestamp), '%Y-%m-%d'),
               formatDateTime(max(timestamp), '%Y-%m-%d')
        FROM wspr.bronze
    """)
    date_range = f"{date_result.result_rows[0][0]} to {date_result.result_rows[0][1]}"
    client.close()

    # ── Per-Band Normalization ──
    norm_consts = CONFIG["norm_constants_per_band"]
    contest_src = CONFIG["data"]["contest_norm_source"]

    log.info("")
    log.info("=== PER-SOURCE PER-BAND NORMALIZATION ===")

    wspr_snr = normalize_snr_per_band(wspr_df, 'wspr', norm_consts)
    rbn_dx_snr = normalize_snr_per_band(rbn_dx_df, 'rbn', norm_consts)
    contest_snr = normalize_snr_per_band(contest_df, contest_src, norm_consts)

    log.info(f"  WSPR:    mean={wspr_snr.mean():.3f}, std={wspr_snr.std():.3f}")
    log.info(f"  RBN DX:  mean={rbn_dx_snr.mean():.3f}, std={rbn_dx_snr.std():.3f}")
    log.info(f"  Contest: mean={contest_snr.mean():.3f}, std={contest_snr.std():.3f}")

    # ── Features (V26: V22-gamma + band_idx) ──
    log.info("")
    log.info("Engineering features...")
    log.info(f"  dnn_dim={CONFIG['model']['dnn_dim']} (trunk sees features 0-14 only)")
    log.info(f"  band_idx at index 17 (routing signal, NOT fed to trunk)")

    wspr_X = engineer_features_v26(wspr_df, CONFIG)
    rbn_dx_X = engineer_features_v26(rbn_dx_df, CONFIG)
    contest_X = engineer_features_v26(contest_df, CONFIG)

    log.info(f"  Feature shape: {wspr_X.shape[1]} columns (15 DNN + SFI + Kp + band_idx)")

    # ── Weights ──
    wspr_w = wspr_df['spot_count'].values.astype(np.float32)
    wspr_w /= wspr_w.mean()
    rbn_dx_w = rbn_dx_df['spot_count'].values.astype(np.float32)
    rbn_dx_w /= rbn_dx_w.mean()
    contest_w = contest_df['spot_count'].values.astype(np.float32)
    contest_w /= contest_w.mean()

    # ── Upsample ──
    rbn_dx_up = CONFIG["data"]["rbn_dx_upsample"]
    contest_up = CONFIG["data"]["contest_upsample"]

    log.info(f"Upsampling RBN DXpedition {rbn_dx_up}x...")
    rbn_dx_X_up = np.tile(rbn_dx_X, (rbn_dx_up, 1))
    rbn_dx_snr_up = np.tile(rbn_dx_snr, rbn_dx_up)
    rbn_dx_w_up = np.tile(rbn_dx_w, rbn_dx_up)

    if contest_up > 1:
        log.info(f"Upsampling Contest {contest_up}x...")
        contest_X_up = np.tile(contest_X, (contest_up, 1))
        contest_snr_up = np.tile(contest_snr, contest_up)
        contest_w_up = np.tile(contest_w, contest_up)
    else:
        contest_X_up = contest_X
        contest_snr_up = contest_snr
        contest_w_up = contest_w

    # ── Combine ──
    X = np.vstack([wspr_X, rbn_dx_X_up, contest_X_up])
    y = np.concatenate([wspr_snr, rbn_dx_snr_up, contest_snr_up]).reshape(-1, 1)
    w = np.concatenate([wspr_w, rbn_dx_w_up, contest_w_up]).reshape(-1, 1)

    n = len(X)
    log.info("")
    log.info(f"Combined dataset: {n:,} rows")
    log.info(f"  WSPR:     {len(wspr_X):,} ({100*len(wspr_X)/n:.1f}%)")
    log.info(f"  RBN DX:   {len(rbn_dx_snr_up):,} ({100*len(rbn_dx_snr_up)/n:.1f}%)")
    log.info(f"  Contest:  {len(contest_snr_up):,} ({100*len(contest_snr_up)/n:.1f}%)")
    log.info(f"Normalized SNR: mean={y.mean():.3f}, std={y.std():.3f}")
    log.info(f"Source data range: {date_range}")

    # V26 verification: check row count matches V22-gamma recipe
    expected_rows = 38_760_089
    if n != expected_rows:
        log.warning(f"Row count {n:,} differs from V22-gamma ({expected_rows:,})")
    else:
        log.info(f"Row count verified: {n:,} (matches V22-gamma)")

    del wspr_df, rbn_dx_df, contest_df
    del wspr_X, rbn_dx_X, contest_X, rbn_dx_X_up, contest_X_up
    gc.collect()

    # ── Dataset + Split ──
    dataset = SignatureDataset(X, y, w)
    val_size = int(n * CONFIG["training"]["val_split"])
    train_size = n - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    batch_size = CONFIG["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    log.info(f"Split: {train_size:,} train / {val_size:,} val")

    # ── Model (V26: IonisGateV26 with band_heads) ──
    model = IonisGateV26(
        dnn_dim=CONFIG["model"]["dnn_dim"],
        sidecar_hidden=CONFIG["model"]["sidecar_hidden"],
        sfi_idx=CONFIG["model"]["sfi_idx"],
        kp_penalty_idx=CONFIG["model"]["kp_penalty_idx"],
        band_idx=CONFIG["model"]["band_idx"],
        num_bands=CONFIG["model"]["num_bands"],
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model: IonisGateV26 ({total_params:,} params)")
    log.info(f"  band_heads: {CONFIG['model']['num_bands']} heads × (256→128→1)")

    # ── Defibrillator ──
    init_defibrillator(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable: {trainable:,} / {total_params:,}")

    # ── Optimizer (V26: band_heads instead of base_head) ──
    param_groups = get_optimizer_groups_v26(
        model,
        trunk_lr=CONFIG["training"]["trunk_lr"],
        scaler_lr=CONFIG["training"]["scaler_lr"],
        sidecar_lr=CONFIG["training"]["sidecar_lr"],
    )
    optimizer = optim.AdamW(param_groups, weight_decay=CONFIG["training"]["weight_decay"])

    epochs = CONFIG["training"]["epochs"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ── HuberLoss ──
    criterion = nn.HuberLoss(reduction='none', delta=CONFIG["training"]["huber_delta"])
    lambda_var = CONFIG["training"]["lambda_var"]

    checkpoint_name = CONFIG["checkpoint"]
    model_path = os.path.join(SCRIPT_DIR, checkpoint_name)
    meta_path = model_path.replace(".safetensors", "_meta.json")

    best_val_loss = float('inf')
    best_pearson = -1.0
    best_kp = 0.0
    best_sfi = 0.0
    best_epoch = 0

    # ── Training Loop ──
    log.info("")
    log.info(f"Training started ({epochs} epochs)")
    hdr = (f"{'Ep':>3s}  {'Train':>8s}  {'Val':>8s}  "
           f"{'RMSE':>7s}  {'Pearson':>8s}  "
           f"{'SFI+':>5s}  {'Kp9-':>5s}  {'Time':>6s}")
    log.info(hdr)
    log.info("-" * len(hdr))

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        # Train
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for bx, by, bw in train_loader:
            bx, by, bw = bx.to(DEVICE), by.to(DEVICE), bw.to(DEVICE)
            optimizer.zero_grad()

            out, sun_gate, storm_gate = model.forward_with_gates(bx)
            primary_loss = (criterion(out, by) * bw).mean()
            var_loss = -lambda_var * (sun_gate.var() + storm_gate.var())
            loss = primary_loss + var_loss

            loss.backward()
            optimizer.step()

            # Weight clamp
            clamp_sidecars(model)

            train_loss_sum += primary_loss.item()
            train_batches += 1

        train_loss = train_loss_sum / train_batches

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for bx, by, bw in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out = model(bx)
                val_loss_sum += criterion(out, by).mean().item()
                val_batches += 1
                all_preds.append(out.cpu())
                all_targets.append(by.cpu())

        val_loss = val_loss_sum / val_batches
        val_rmse = np.sqrt(val_loss)
        val_pearson = pearson_r(torch.cat(all_preds), torch.cat(all_targets))

        sfi_benefit = (model.get_sun_effect(200.0 / 300.0, DEVICE) -
                       model.get_sun_effect(70.0 / 300.0, DEVICE))
        storm_cost = (model.get_storm_effect(1.0, DEVICE) -
                      model.get_storm_effect(0.0, DEVICE))

        scheduler.step()
        epoch_sec = time.perf_counter() - t0

        is_best = val_loss < best_val_loss
        marker = ""
        if is_best:
            best_val_loss = val_loss
            best_pearson = val_pearson
            best_kp = storm_cost
            best_sfi = sfi_benefit
            best_epoch = epoch

            save_safetensors(model.state_dict(), model_path)

            metadata = {
                'version': CONFIG['version'],
                'variant': CONFIG.get('variant', 'alpha'),
                'architecture': 'IonisGateV26',
                'date_range': date_range,
                'sample_size': n,
                'val_rmse': val_rmse,
                'val_pearson': val_pearson,
                'sfi_benefit': sfi_benefit,
                'storm_cost': storm_cost,
            }
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            marker = " *"

        log.info(
            f"{epoch:3d}  {train_loss:8.4f}  {val_loss:8.4f}  "
            f"{val_rmse:6.3f}σ  {val_pearson:+7.4f}  "
            f"{sfi_benefit:+4.2f}  {storm_cost:+4.2f}  "
            f"{epoch_sec:5.1f}s{marker}"
        )

        audit_epoch(run_id, epoch, train_loss, val_loss, val_rmse, val_pearson,
                    sfi_benefit, storm_cost, epoch_sec, is_best)

    # ── Final Report ──
    best_rmse = np.sqrt(best_val_loss)
    wall_seconds = time.perf_counter() - training_start_time

    audit_complete(run_id, CONFIG, train_size, val_size, date_range,
                   best_rmse, best_pearson, best_sfi, best_kp, best_epoch, wall_seconds)

    log.info("-" * len(hdr))
    log.info(f"Training complete. Best RMSE: {best_rmse:.4f}σ, Pearson: {best_pearson:+.4f}")
    log.info(f"Checkpoint: {model_path}")

    log.info("")
    log.info("=" * 70)
    log.info("V26-alpha RESULTS")
    log.info("=" * 70)

    p_min = CONFIG["validation"]["pearson_min"]
    kp_min = CONFIG["validation"]["kp_storm_min"]
    sfi_min = CONFIG["validation"]["sfi_benefit_min"]

    p_pass = best_pearson >= p_min
    kp_pass = best_kp >= kp_min
    sfi_pass = best_sfi >= sfi_min

    log.info("")
    log.info("SUCCESS CRITERIA:")
    log.info(f"  Pearson >= +{p_min}:    {best_pearson:+.4f}  {'PASS' if p_pass else 'FAIL'}")
    log.info(f"  Kp sidecar >= +{kp_min}σ: {best_kp:+.3f}σ  {'PASS' if kp_pass else 'FAIL'}")
    log.info(f"  SFI sidecar >= +{sfi_min}σ: {best_sfi:+.3f}σ  {'PASS' if sfi_pass else 'FAIL'}")

    log.info("")
    log.info(">>> Run TST-900 to verify band×time discrimination <<<")
    log.info(">>> Run ACID TEST: DN13→JN48, 10m, 02 UTC, Feb, SFI=150 must be ≤ 0.0σ <<<")
    log.info("")
    log.info("V22-gamma REFERENCE: Pearson +0.492, TST-900 9/11, +3.7σ on acid test")
    log.info("V26 TARGET: TST-900 >= 10/11, acid test ≤ 0.0σ")


if __name__ == '__main__':
    train()
