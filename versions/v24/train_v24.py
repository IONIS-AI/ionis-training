#!/usr/bin/env python3
"""
train_v24.py — IONIS V24 Training: The Subtraction

V24 removes the sun sidecar entirely per V24-ARCHITECTURE.md.

Key change:
    - Sun sidecar REMOVED — was contributing only a fixed +0.48σ (clamp floor artifact)
    - Storm sidecar unchanged — fc2 shows real variance (1.17-1.47), load-bearing
    - Trunk retains all V22-gamma cross-products for band×time physics
    - No IRI features (proven to create shortcuts in V23/AUDIT-03)

Hypothesis:
    V22-gamma achieved 9/11 TST-900 while the trunk fought the sidecar's +0.48σ
    forced error on every prediction. Removing the sidecar frees trunk capacity.

Expected outcome:
    TST-900: 9/11 → 10/11 or 11/11
    Pearson: may decrease slightly (the +0.48σ boost helped high bands)

Architecture constraints (V16 Physics Laws, adapted):
    1. Architecture: IonisGateV24 (no sun sidecar)
    2. Loss: HuberLoss(delta=1.0) — robust to synthetic anchors
    3. Regularization: Gate variance loss on storm gate only
    4. Init: Defibrillator on storm sidecar only
    5. Constraint: Weight clamp [0.5, 2.0] on storm sidecar only

Success criteria: TST-900 >= 10/11, Pearson >= +0.46
Escape valve: If TST-900 < 9/11, reintroduce sun sidecar with lower clamp floor
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
    load_source_data,
    log_config,
)
from common.model import _gate, get_device

# ── Load Configuration ────────────────────────────────────────────────────────

CONFIG_FILE = os.path.join(SCRIPT_DIR, "config_v24.json")

with open(CONFIG_FILE) as f:
    CONFIG = json.load(f)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(f"ionis-{CONFIG['version']}")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ── IonisGateV24: No Sun Sidecar ──────────────────────────────────────────────

class IonisGateV24(nn.Module):
    """
    IONIS V24 Model — storm sidecar only, sun sidecar removed.

    The sun sidecar was contributing only a fixed +0.48σ (clamp floor artifact).
    Removing it frees trunk capacity for band×time physics.

    Architecture:
        - Trunk: geography/time features (15-dim) → 256-dim representation
        - Storm gate from trunk output (256-dim)
        - Gate range 0.5-2.0
        - Separate base_head (256→128→1) and storm_scaler_head (256→64→1)
        - Uses Mish activation

    Args:
        dnn_dim: Number of geography/time features (default 15)
        sidecar_hidden: Hidden units in MonotonicMLP (default 8)
        kp_penalty_idx: Index of Kp penalty feature in input (default 16)
        gate_init_bias: Initial bias for scaler head (default -ln(2))
    """

    def __init__(self, dnn_dim=15, sidecar_hidden=8, kp_penalty_idx=16,
                 gate_init_bias=None):
        super().__init__()

        if gate_init_bias is None:
            gate_init_bias = -math.log(2.0)

        self.dnn_dim = dnn_dim
        self.kp_penalty_idx = kp_penalty_idx

        # Trunk: geography/time features → 256-dim representation
        self.trunk = nn.Sequential(
            nn.Linear(dnn_dim, 512), nn.Mish(),
            nn.Linear(512, 256), nn.Mish(),
        )

        # Base head: trunk → SNR prediction
        self.base_head = nn.Sequential(
            nn.Linear(256, 128), nn.Mish(),
            nn.Linear(128, 1),
        )

        # Storm scaler head: trunk → gate logit (256-dim input)
        self.storm_scaler_head = nn.Sequential(
            nn.Linear(256, 64), nn.Mish(),
            nn.Linear(64, 1),
        )

        # Storm sidecar (monotonic) — the only sidecar
        self.storm_sidecar = MonotonicMLP(hidden_dim=sidecar_hidden)

        # Initialize scaler head
        self._init_scaler_head(gate_init_bias)

    def _init_scaler_head(self, gate_init_bias):
        """Initialize scaler head bias for balanced gate."""
        final_layer = self.storm_scaler_head[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.constant_(final_layer.bias, gate_init_bias)

    def forward(self, x):
        x_deep = x[:, :self.dnn_dim]
        x_kp = x[:, self.kp_penalty_idx:self.kp_penalty_idx + 1]

        trunk_out = self.trunk(x_deep)
        base_snr = self.base_head(trunk_out)

        storm_logit = self.storm_scaler_head(trunk_out)
        storm_gate = _gate(storm_logit)

        return base_snr + storm_gate * self.storm_sidecar(x_kp)

    def forward_with_gates(self, x):
        """Forward pass returning gate value for variance loss."""
        x_deep = x[:, :self.dnn_dim]
        x_kp = x[:, self.kp_penalty_idx:self.kp_penalty_idx + 1]

        trunk_out = self.trunk(x_deep)
        base_snr = self.base_head(trunk_out)

        storm_logit = self.storm_scaler_head(trunk_out)
        storm_gate = _gate(storm_logit)
        storm_boost = self.storm_sidecar(x_kp)

        return base_snr + storm_gate * storm_boost, storm_gate

    def get_storm_effect(self, kp_penalty, device):
        """Get raw storm sidecar output for a given Kp penalty value."""
        with torch.no_grad():
            x = torch.tensor([[kp_penalty]], dtype=torch.float32, device=device)
            return self.storm_sidecar(x).item()

    def get_gates(self, x):
        """Get gate value without gradient tracking."""
        x_deep = x[:, :self.dnn_dim]
        with torch.no_grad():
            trunk_out = self.trunk(x_deep)
            storm_logit = self.storm_scaler_head(trunk_out)
        return _gate(storm_logit)


# ── V24 Training Utilities ────────────────────────────────────────────────────

def init_defibrillator_v24(model):
    """
    Initialize storm sidecar with defibrillator pattern.

    Only storm sidecar — sun sidecar is removed in V24.

    Args:
        model: IonisGateV24 model
    """
    with torch.no_grad():
        for name, param in model.storm_sidecar.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, 0.8, 1.2)
            elif 'bias' in name and 'fc2' in name:
                nn.init.constant_(param, -10.0)
        # Freeze fc1.bias
        model.storm_sidecar.fc1.bias.requires_grad = False
    log.info("Defibrillator init: storm sidecar weights [0.8-1.2], fc2.bias=-10")


def clamp_sidecar_v24(model, min_val=0.5, max_val=2.0):
    """
    Clamp storm sidecar weights to [min_val, max_val].

    Only storm sidecar — sun sidecar is removed in V24.

    Args:
        model: IonisGateV24 model
        min_val: Minimum weight value (default 0.5)
        max_val: Maximum weight value (default 2.0)
    """
    with torch.no_grad():
        for param in model.storm_sidecar.parameters():
            if param.requires_grad:
                param.clamp_(min_val, max_val)


def get_optimizer_groups_v24(model, trunk_lr=1e-5, scaler_lr=5e-5, sidecar_lr=1e-3):
    """
    Get 4-group optimizer configuration for V24.

    Groups:
        1. Trunk (slowest LR)
        2. Base head (trunk LR)
        3. Storm scaler head (intermediate LR)
        4. Storm sidecar (fastest LR)

    Args:
        model: IonisGateV24 model
        trunk_lr: Learning rate for trunk and base_head
        scaler_lr: Learning rate for storm scaler head
        sidecar_lr: Learning rate for storm sidecar

    Returns:
        List of parameter groups for AdamW optimizer
    """
    return [
        {'params': model.trunk.parameters(), 'lr': trunk_lr},
        {'params': model.base_head.parameters(), 'lr': trunk_lr},
        {'params': model.storm_scaler_head.parameters(), 'lr': scaler_lr},
        {'params': [p for p in model.storm_sidecar.parameters() if p.requires_grad],
         'lr': sidecar_lr},
    ]


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
                val_rmse: float, val_pearson: float, storm_cost: float,
                epoch_seconds: float, is_best: bool):
    """Record epoch metrics in ClickHouse."""
    try:
        query = f"""
            INSERT INTO training.epochs (
                run_id, epoch, train_loss, val_loss, val_rmse, val_pearson,
                sfi_benefit, storm_cost, epoch_seconds, is_best
            ) VALUES (
                '{run_id}', {epoch}, {train_loss}, {val_loss}, {val_rmse},
                {val_pearson}, 0.0, {storm_cost}, {epoch_seconds}, {1 if is_best else 0}
            )
        """
        requests.post(AUDIT_HOST, params={"query": query}, timeout=5)
    except Exception:
        pass


def audit_complete(run_id: str, config: dict, train_rows: int, val_rows: int,
                   date_range: str, val_rmse: float, val_pearson: float,
                   storm_cost: float, best_epoch: int, wall_seconds: float):
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
                {val_rmse}, {val_pearson}, 0.0, {storm_cost}, {best_epoch}
            )
        """
        requests.post(AUDIT_HOST, params={"query": query}, timeout=5)
    except Exception as e:
        log.warning(f"Audit complete failed: {e}")


# ── Per-Band Normalization ───────────────────────────────────────────────────

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


# ── Main Training ────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("IONIS V24 TRAINING: THE SUBTRACTION")
    log.info("=" * 70)
    log.info("")
    log.info("Architecture: IonisGateV24 (sun sidecar REMOVED)")
    log.info("Hypothesis: Trunk fought +0.48σ forced error. Removing frees capacity.")
    log.info("Target: TST-900 >= 10/11")
    log.info("")

    run_id = str(uuid.uuid4())
    log.info(f"Run ID: {run_id}")
    log.info(f"Device: {DEVICE}")
    log_config(CONFIG, CONFIG_FILE, DEVICE)

    wall_start = time.time()

    # ── Connect to ClickHouse ─────────────────────────────────────────────────
    ch_host = CONFIG["clickhouse"]["host"]
    ch_port = CONFIG["clickhouse"]["port"]
    log.info(f"Connecting to ClickHouse at {ch_host}:{ch_port}...")
    client = clickhouse_connect.get_client(host=ch_host, port=ch_port)

    # ── Load Data ─────────────────────────────────────────────────────────────
    log.info("-" * 65)
    log.info("LOADING DATA")
    log.info("-" * 65)

    norm_constants = CONFIG["norm_constants_per_band"]

    # WSPR
    wspr_sample = CONFIG["data"]["wspr_sample"]
    log.info(f"Loading WSPR signatures (sample={wspr_sample:,})...")
    df_wspr = load_source_data(client, "wspr.signatures_v2_terrestrial",
                                sample_size=wspr_sample, include_day_of_year=True)
    log.info(f"  WSPR: {len(df_wspr):,} rows")

    # DXpedition
    dx_upsample = CONFIG["data"]["rbn_dx_upsample"]
    log.info(f"Loading DXpedition signatures (upsample={dx_upsample}x)...")
    df_dx = load_source_data(client, "rbn.dxpedition_signatures",
                              include_day_of_year=True)
    log.info(f"  DXpedition: {len(df_dx):,} rows (before upsample)")

    # Contest
    contest_upsample = CONFIG["data"]["contest_upsample"]
    log.info(f"Loading Contest signatures (upsample={contest_upsample}x)...")
    df_contest = load_source_data(client, "contest.signatures",
                                   include_day_of_year=True)
    log.info(f"  Contest: {len(df_contest):,} rows")

    # ── Feature Engineering ───────────────────────────────────────────────────
    log.info("-" * 65)
    log.info("FEATURE ENGINEERING")
    log.info("-" * 65)

    X_wspr = engineer_features(df_wspr, CONFIG)
    y_wspr = normalize_snr_per_band(df_wspr, "wspr", norm_constants)
    w_wspr = np.ones(len(y_wspr), dtype=np.float32)

    X_dx = engineer_features(df_dx, CONFIG)
    y_dx = normalize_snr_per_band(df_dx, "rbn", norm_constants)
    w_dx = np.ones(len(y_dx), dtype=np.float32)

    # Upsample DXpedition by tiling (not just weighting)
    log.info(f"Upsampling DXpedition {dx_upsample}x...")
    X_dx_up = np.tile(X_dx, (dx_upsample, 1))
    y_dx_up = np.tile(y_dx, dx_upsample)
    w_dx_up = np.tile(w_dx, dx_upsample)

    X_contest = engineer_features(df_contest, CONFIG)
    contest_norm_source = CONFIG["data"].get("contest_norm_source", "wspr")
    y_contest = normalize_snr_per_band(df_contest, contest_norm_source, norm_constants)
    w_contest = np.ones(len(y_contest), dtype=np.float32)

    # Upsample Contest if needed
    if contest_upsample > 1:
        log.info(f"Upsampling Contest {contest_upsample}x...")
        X_contest_up = np.tile(X_contest, (contest_upsample, 1))
        y_contest_up = np.tile(y_contest, contest_upsample)
        w_contest_up = np.tile(w_contest, contest_upsample)
    else:
        X_contest_up = X_contest
        y_contest_up = y_contest
        w_contest_up = w_contest

    # Combine
    X = np.vstack([X_wspr, X_dx_up, X_contest_up])
    y = np.concatenate([y_wspr, y_dx_up, y_contest_up])
    w = np.concatenate([w_wspr, w_dx_up, w_contest_up])

    log.info(f"Combined: {len(X):,} rows")
    log.info(f"Features: {X.shape[1]}")

    # Date range for audit (query from wspr.bronze - signature tables don't have timestamps)
    date_result = client.query("""
        SELECT formatDateTime(min(timestamp), '%Y-%m-%d'),
               formatDateTime(max(timestamp), '%Y-%m-%d')
        FROM wspr.bronze
    """)
    date_range = f"{date_result.result_rows[0][0]} to {date_result.result_rows[0][1]}"
    log.info(f"Date range: {date_range}")

    # Free memory
    del df_wspr, df_dx, df_contest
    del X_wspr, X_dx, X_dx_up, X_contest, X_contest_up
    del y_wspr, y_dx, y_dx_up, y_contest, y_contest_up
    del w_wspr, w_dx, w_dx_up, w_contest, w_contest_up
    gc.collect()

    # ── Train/Val Split ───────────────────────────────────────────────────────
    dataset = SignatureDataset(X, y, w)
    val_split = CONFIG["training"]["val_split"]
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    batch_size = CONFIG["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    log.info(f"Train: {train_size:,}, Val: {val_size:,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    log.info("-" * 65)
    log.info("MODEL INITIALIZATION")
    log.info("-" * 65)

    model = IonisGateV24(
        dnn_dim=CONFIG["model"]["dnn_dim"],
        sidecar_hidden=CONFIG["model"]["sidecar_hidden"],
        kp_penalty_idx=CONFIG["model"]["kp_penalty_idx"],
        gate_init_bias=CONFIG["model"].get("gate_init_bias"),
    ).to(DEVICE)

    # Defibrillator init on storm sidecar
    init_defibrillator_v24(model)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {n_params:,}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    opt_groups = get_optimizer_groups_v24(
        model,
        trunk_lr=CONFIG["training"]["trunk_lr"],
        scaler_lr=CONFIG["training"]["scaler_lr"],
        sidecar_lr=CONFIG["training"]["sidecar_lr"],
    )
    optimizer = optim.AdamW(opt_groups, weight_decay=CONFIG["training"]["weight_decay"])

    # Loss
    huber_delta = CONFIG["training"]["huber_delta"]
    criterion = nn.HuberLoss(delta=huber_delta)
    lambda_var = CONFIG["training"]["lambda_var"]

    # ── Training Loop ─────────────────────────────────────────────────────────
    log.info("-" * 65)
    log.info("TRAINING")
    log.info("-" * 65)

    epochs = CONFIG["training"]["epochs"]
    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    best_metrics = {}

    audit_start(run_id, CONFIG)

    log.info(f"{'Epoch':>5} {'Train':>10} {'Val':>10} {'RMSE':>7} {'Pearson':>8} "
             f"{'Kp9-':>7} {'Time':>7}")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for X_batch, y_batch, w_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE).unsqueeze(1)
            w_batch = w_batch.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()

            pred, storm_gate = model.forward_with_gates(X_batch)

            # Huber loss (weighted)
            loss = (criterion(pred, y_batch) * w_batch).mean()

            # Gate variance loss (storm only in V24)
            gate_var_loss = -storm_gate.var()
            loss = loss + lambda_var * gate_var_loss

            loss.backward()
            optimizer.step()

            # Clamp storm sidecar weights
            clamp_sidecar_v24(model)

            train_loss_sum += loss.item() * len(X_batch)
            train_n += len(X_batch)

        train_loss = train_loss_sum / train_n

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        all_pred = []
        all_target = []

        with torch.no_grad():
            for X_batch, y_batch, w_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE).unsqueeze(1)
                w_batch = w_batch.to(DEVICE).unsqueeze(1)

                pred = model(X_batch)
                loss = (criterion(pred, y_batch) * w_batch).mean()

                val_loss_sum += loss.item() * len(X_batch)
                val_n += len(X_batch)

                all_pred.append(pred.cpu())
                all_target.append(y_batch.cpu())

        val_loss = val_loss_sum / val_n
        val_rmse = np.sqrt(val_loss)

        preds = torch.cat(all_pred)
        targets = torch.cat(all_target)
        val_pearson = pearson_r(preds, targets)

        # Storm sidecar effect: Kp=9 (penalty=0) vs Kp=0 (penalty=1)
        storm_kp9 = model.get_storm_effect(0.0, DEVICE)  # kp_penalty=0 → Kp=9
        storm_kp0 = model.get_storm_effect(1.0, DEVICE)  # kp_penalty=1 → Kp=0
        storm_cost = storm_kp0 - storm_kp9  # Higher value = more storm cost

        epoch_time = time.time() - epoch_start

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                'val_rmse': val_rmse,
                'val_pearson': val_pearson,
                'storm_cost': storm_cost,
            }

        marker = "*" if is_best else ""
        log.info(f"{epoch:5d} {train_loss:10.4f} {val_loss:10.4f} {val_rmse:6.3f}σ "
                 f"{val_pearson:+7.4f} {storm_cost:+6.2f} {epoch_time:6.1f}s {marker}")

        # Log sidecar weights every 10 epochs
        if epoch % 10 == 0:
            with torch.no_grad():
                fc1_w = model.storm_sidecar.fc1.weight.abs()
                fc2_w = model.storm_sidecar.fc2.weight.abs()
                log.info(f"  STORM SIDECAR WEIGHTS (epoch {epoch}):")
                log.info(f"    fc1: min={fc1_w.min():.4f}, max={fc1_w.max():.4f}")
                log.info(f"    fc2: min={fc2_w.min():.4f}, max={fc2_w.max():.4f}")

        audit_epoch(run_id, epoch, train_loss, val_loss, val_rmse, val_pearson,
                    storm_cost, epoch_time, is_best)

    # ── Save Checkpoint ───────────────────────────────────────────────────────
    log.info("-" * 65)
    log.info("SAVING CHECKPOINT")
    log.info("-" * 65)

    checkpoint_path = os.path.join(SCRIPT_DIR, CONFIG["checkpoint"])
    save_safetensors(best_state, checkpoint_path)
    log.info(f"Saved: {checkpoint_path}")

    # Save metadata
    meta = {
        'run_id': run_id,
        'version': CONFIG['version'],
        'variant': CONFIG.get('variant', 'alpha'),
        'best_epoch': best_epoch,
        'val_rmse': best_metrics['val_rmse'],
        'val_pearson': best_metrics['val_pearson'],
        'storm_cost': best_metrics['storm_cost'],
        'date_range': date_range,
        'train_rows': train_size,
        'val_rows': val_size,
    }
    meta_path = checkpoint_path.replace('.safetensors', '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"Saved: {meta_path}")

    wall_seconds = time.time() - wall_start
    audit_complete(run_id, CONFIG, train_size, val_size, date_range,
                   best_metrics['val_rmse'], best_metrics['val_pearson'],
                   best_metrics['storm_cost'], best_epoch, wall_seconds)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("V24 TRAINING COMPLETE")
    log.info("=" * 70)
    log.info("")
    log.info(f"Best Epoch: {best_epoch}")
    log.info(f"Best RMSE: {best_metrics['val_rmse']:.4f}σ")
    log.info(f"Best Pearson: {best_metrics['val_pearson']:+.4f}")
    log.info(f"Storm Cost (Kp9-): {best_metrics['storm_cost']:+.3f}σ")
    log.info(f"Wall Time: {wall_seconds/60:.1f} minutes")
    log.info("")
    log.info("Next: Run TST-900 tests to verify band×time physics")
    log.info("")


if __name__ == "__main__":
    main()
