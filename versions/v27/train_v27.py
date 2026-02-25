#!/usr/bin/env python3
"""
train_v27.py — IONIS V27 Training: Physics-Informed Loss

V27 Strategy: Fix the loss, not the network.

V22-gamma achieves 16/17 on KI7MT operator tests. Its one failure (+0.540σ on
10m at 02 UTC) is 0.540σ away from perfect. Six architectural changes (V23-V26)
all made it worse.

V27 keeps V22-gamma architecture LOCKED and adds a physics-informed loss penalty:
- IF freq >= 21 MHz (15m, 12m, 10m)
- AND tx_solar_elevation < -6° (TX past civil twilight)
- AND rx_solar_elevation < -6° (RX past civil twilight)
- AND predicted_snr > -1.0σ (model predicts viable signal)
- THEN loss *= 10.0

This teaches the model that being wrong about a dead high band costs more than
being wrong about an open band.

Fine-tunes from V22-gamma checkpoint with reduced learning rates (10% of V22).
30 epochs (nudging, not retraining).
"""

import gc
import json
import logging
import os
import socket
import sys
import time
import uuid
from datetime import datetime

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from safetensors.torch import save_file as save_safetensors, load_file as load_safetensors

import clickhouse_connect

# Add parent for common imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VERSIONS_DIR = os.path.dirname(SCRIPT_DIR)
COMMON_DIR = os.path.join(VERSIONS_DIR, "common")
sys.path.insert(0, VERSIONS_DIR)
sys.path.insert(0, COMMON_DIR)

from common.train_common import (
    IonisGate,
    clamp_sidecars,
    get_optimizer_groups,
    load_source_data,
    grid4_to_latlon_arrays,
    solar_elevation_vectorized,
)

# ── Load Configuration ────────────────────────────────────────────────────────

CONFIG_FILE = os.path.join(SCRIPT_DIR, "config_v27.json")

with open(CONFIG_FILE) as f:
    CONFIG = json.load(f)

# Results logging
RESULTS_DIR = os.path.join(VERSIONS_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(RESULTS_DIR, f"v27_{timestamp}.log")

# Set up dual logging (console + file)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ]
)
log = logging.getLogger(f"ionis-{CONFIG['version']}")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ── Physics-Informed Loss ─────────────────────────────────────────────────────

class PhysicsInformedLoss(nn.Module):
    """
    HuberLoss + diurnal physics penalty.

    When the model predicts viable signal on a high-frequency band during darkness
    at both endpoints, the loss for those rows is multiplied by a penalty factor.
    """

    def __init__(self, delta=1.0, penalty_factor=10.0, closure_threshold=-1.0,
                 freq_threshold=21.0, solar_threshold=-6.0):
        super().__init__()
        self.base_loss = nn.HuberLoss(delta=delta, reduction='none')
        self.penalty_factor = penalty_factor
        self.closure_threshold = closure_threshold
        self.freq_threshold = freq_threshold
        self.solar_threshold = solar_threshold
        self._last_violation_count = 0
        self._last_batch_size = 0

    def forward(self, pred, target, freq_mhz, tx_solar, rx_solar, weights=None):
        """
        Args:
            pred: [batch, 1] predicted SNR (σ)
            target: [batch, 1] actual SNR (σ)
            freq_mhz: [batch] frequency in MHz (raw, not normalized)
            tx_solar: [batch] TX solar elevation in degrees (raw)
            rx_solar: [batch] RX solar elevation in degrees (raw)
            weights: [batch, 1] optional sample weights

        Returns:
            Scalar loss value
        """
        # Base Huber loss per sample
        base = self.base_loss(pred, target)

        # Physics violation mask (all conditions must be true)
        high_band = freq_mhz >= self.freq_threshold
        tx_dark = tx_solar < self.solar_threshold
        rx_dark = rx_solar < self.solar_threshold
        pred_positive = pred.squeeze() > self.closure_threshold

        # Violation: high band + both endpoints dark + model says "open"
        violation = high_band & tx_dark & rx_dark & pred_positive

        # Apply penalty multiplier to violating rows
        multiplier = torch.ones_like(base)
        multiplier[violation.unsqueeze(1)] = self.penalty_factor

        penalized = base * multiplier

        # Apply sample weights if provided
        if weights is not None:
            penalized = penalized * weights

        # Track violations for logging
        self._last_violation_count = violation.sum().item()
        self._last_batch_size = pred.shape[0]

        return penalized.mean()

    @property
    def violation_rate(self):
        """Fraction of last batch that were physics violations."""
        if self._last_batch_size > 0:
            return self._last_violation_count / self._last_batch_size
        return 0.0


# ── Training Audit Trail ──────────────────────────────────────────────────────

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


# ── Helper Functions ──────────────────────────────────────────────────────────

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


def engineer_features_with_aux(df, config):
    """
    Engineer features for V27.

    Returns:
        X: [N, 17] feature tensor (same as V22-gamma)
        aux: [N, 3] auxiliary columns [freq_mhz, tx_solar, rx_solar] in RAW form
    """
    n = len(df)
    band_to_hz = {int(k): v for k, v in config["band_to_hz"].items()}

    # Extract raw values from signature columns
    band = df['band'].values
    freq_hz = np.array([band_to_hz.get(b, 14097100) for b in band], dtype=np.float32)
    freq_mhz = freq_hz / 1_000_000.0

    # Convert grid4 to lat/lon
    tx_lats, tx_lons = grid4_to_latlon_arrays(df['tx_grid_4'].values)
    rx_lats, rx_lons = grid4_to_latlon_arrays(df['rx_grid_4'].values)

    hour_utc = df['hour'].values.astype(np.float32)
    month = df['month'].values.astype(np.float32)

    # Get day_of_year
    if 'day_of_year' in df.columns:
        day_of_year = df['day_of_year'].values.astype(np.float32)
    else:
        day_of_year = (month - 1) * 30.5 + 15

    # Compute solar elevations (raw, for loss function)
    tx_solar = solar_elevation_vectorized(tx_lats, tx_lons, hour_utc, day_of_year)
    rx_solar = solar_elevation_vectorized(rx_lats, rx_lons, hour_utc, day_of_year)

    # Features (same as V22-gamma)
    X = np.zeros((n, 17), dtype=np.float32)

    # Distance (normalized)
    X[:, 0] = df['avg_distance'].values / 20000.0

    # Frequency (log, normalized)
    freq_log = np.log10(freq_hz) / 8.0
    X[:, 1] = freq_log

    # Hour (cyclic)
    hour_rad = 2 * np.pi * hour_utc / 24.0
    X[:, 2] = np.sin(hour_rad)
    X[:, 3] = np.cos(hour_rad)

    # Azimuth (cyclic)
    az = df['avg_azimuth'].values.astype(np.float32)
    az_rad = np.radians(az)
    X[:, 4] = np.sin(az_rad)
    X[:, 5] = np.cos(az_rad)

    # Latitude features
    X[:, 6] = np.abs(tx_lats - rx_lats) / 180.0  # lat_diff
    midpoint_lat = (tx_lats + rx_lats) / 2.0
    X[:, 7] = midpoint_lat / 90.0

    # Season (cyclic)
    season_rad = 2 * np.pi * month / 12.0
    X[:, 8] = np.sin(season_rad)
    X[:, 9] = np.cos(season_rad)

    # Vertex latitude (great circle)
    azimuth_rad = np.radians(az)
    tx_lat_rad = np.radians(tx_lats)
    vertex_lat_rad = np.arccos(np.abs(np.sin(azimuth_rad) * np.cos(tx_lat_rad)))
    X[:, 10] = np.degrees(vertex_lat_rad) / 90.0

    # Solar depression (normalized to [-1, 1] range)
    X[:, 11] = tx_solar / 90.0
    X[:, 12] = rx_solar / 90.0

    # Cross-products: freq × darkness (high band + dark = big negative)
    # Centered around 10 MHz pivot — below helps, above hurts in darkness
    freq_centered = freq_log - 0.875
    tx_dark = np.clip(-tx_solar / 18.0, 0, 1)
    rx_dark = np.clip(-rx_solar / 18.0, 0, 1)
    X[:, 13] = freq_centered * tx_dark
    X[:, 14] = freq_centered * rx_dark

    # SFI (normalized)
    sfi = df['avg_sfi'].values.astype(np.float32)
    X[:, 15] = sfi / 300.0

    # Kp penalty (computed from raw Kp)
    kp = df['avg_kp'].values.astype(np.float32)
    kp_penalty = np.where(kp >= 5, (kp - 4) / 5.0, 0.0)
    X[:, 16] = kp_penalty

    # Auxiliary columns for loss function (RAW values, not normalized)
    aux = np.zeros((n, 3), dtype=np.float32)
    aux[:, 0] = freq_mhz
    aux[:, 1] = tx_solar
    aux[:, 2] = rx_solar

    return X, aux


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    """Main training function."""
    run_id = str(uuid.uuid4())
    training_start_time = time.perf_counter()

    # Log header
    log.info("=" * 70)
    log.info("IONIS v27 | Development")
    log.info("=" * 70)
    log.info(f"Config: {CONFIG_FILE}")
    log.info("")
    log.info(">>> V27 STRATEGY: Physics-Informed Loss <<<")
    log.info(">>> Fine-tuning V22-gamma with 10x penalty on high-band night violations <<<")
    log.info("")
    log.info("=== TRAINING ===")
    log.info(f"  Device: {DEVICE}")
    log.info(f"  Epochs: {CONFIG['training']['epochs']}")
    log.info(f"  Batch size: {CONFIG['training']['batch_size']:,}")
    log.info(f"  Validation split: {int(CONFIG['training']['val_split'] * 100)}%")
    log.info("")
    log.info("=== LEARNING RATES (10% of V22-gamma for fine-tuning) ===")
    log.info(f"  Trunk + Gates: {CONFIG['training']['trunk_lr']}")
    log.info(f"  Sidecars: {CONFIG['training']['sidecar_lr']}")
    log.info("")
    log.info("=== PHYSICS PENALTY ===")
    physics = CONFIG["physics_loss"]
    log.info(f"  Penalty factor: {physics['penalty_factor']}x")
    log.info(f"  Closure threshold: {physics['closure_threshold']}σ")
    log.info(f"  Frequency threshold: >= {physics['freq_mhz_threshold']} MHz (15m, 12m, 10m)")
    log.info(f"  Solar threshold: < {physics['solar_elev_threshold']}° (civil twilight)")
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

    log.info("=== PER-SOURCE PER-BAND NORMALIZATION ===")
    wspr_snr = normalize_snr_per_band(wspr_df, 'wspr', norm_consts)
    rbn_dx_snr = normalize_snr_per_band(rbn_dx_df, 'rbn', norm_consts)
    contest_snr = normalize_snr_per_band(contest_df, contest_src, norm_consts)

    log.info(f"  WSPR:    mean={wspr_snr.mean():.3f}, std={wspr_snr.std():.3f}")
    log.info(f"  RBN DX:  mean={rbn_dx_snr.mean():.3f}, std={rbn_dx_snr.std():.3f}")
    log.info(f"  Contest: mean={contest_snr.mean():.3f}, std={contest_snr.std():.3f}")

    # ── Features + Auxiliary ──
    log.info("")
    log.info("Engineering features with auxiliary columns...")
    wspr_X, wspr_aux = engineer_features_with_aux(wspr_df, CONFIG)
    rbn_dx_X, rbn_dx_aux = engineer_features_with_aux(rbn_dx_df, CONFIG)
    contest_X, contest_aux = engineer_features_with_aux(contest_df, CONFIG)
    log.info(f"  Feature shape: {wspr_X.shape[1]} columns")
    log.info(f"  Auxiliary shape: {wspr_aux.shape[1]} columns (freq_mhz, tx_solar, rx_solar)")

    # ── Weights ──
    wspr_w = wspr_df['spot_count'].values.astype(np.float32)
    wspr_w /= wspr_w.mean()
    rbn_dx_w = rbn_dx_df['spot_count'].values.astype(np.float32)
    rbn_dx_w /= rbn_dx_w.mean()
    contest_w = contest_df['spot_count'].values.astype(np.float32)
    contest_w /= contest_w.mean()

    # ── Upsample ──
    rbn_dx_up = CONFIG["data"]["rbn_dx_upsample"]
    log.info(f"Upsampling RBN DXpedition {rbn_dx_up}x...")
    rbn_dx_X_up = np.tile(rbn_dx_X, (rbn_dx_up, 1))
    rbn_dx_aux_up = np.tile(rbn_dx_aux, (rbn_dx_up, 1))
    rbn_dx_snr_up = np.tile(rbn_dx_snr, rbn_dx_up)
    rbn_dx_w_up = np.tile(rbn_dx_w, rbn_dx_up)

    # ── Combine ──
    X = np.vstack([wspr_X, rbn_dx_X_up, contest_X])
    aux = np.vstack([wspr_aux, rbn_dx_aux_up, contest_aux])
    y = np.concatenate([wspr_snr, rbn_dx_snr_up, contest_snr]).reshape(-1, 1)
    w = np.concatenate([wspr_w, rbn_dx_w_up, contest_w]).reshape(-1, 1)

    n = len(X)
    log.info("")
    log.info(f"Combined dataset: {n:,} rows")
    log.info(f"  WSPR:     {len(wspr_X):,} ({100*len(wspr_X)/n:.1f}%)")
    log.info(f"  RBN DX:   {len(rbn_dx_snr_up):,} ({100*len(rbn_dx_snr_up)/n:.1f}%)")
    log.info(f"  Contest:  {len(contest_snr):,} ({100*len(contest_snr)/n:.1f}%)")
    log.info(f"Normalized SNR: mean={y.mean():.3f}, std={y.std():.3f}")
    log.info(f"Source data range: {date_range}")

    # Cleanup
    del wspr_df, rbn_dx_df, contest_df
    del wspr_X, rbn_dx_X, contest_X, rbn_dx_X_up
    del wspr_aux, rbn_dx_aux, contest_aux, rbn_dx_aux_up
    gc.collect()

    # ── Dataset + Split ──
    # Custom dataset that includes auxiliary columns
    X_t = torch.tensor(X, dtype=torch.float32)
    aux_t = torch.tensor(aux, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    w_t = torch.tensor(w, dtype=torch.float32)

    dataset = TensorDataset(X_t, aux_t, y_t, w_t)

    val_size = int(n * CONFIG["training"]["val_split"])
    train_size = n - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    batch_size = CONFIG["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    log.info(f"Split: {train_size:,} train / {val_size:,} val")

    # ── Model (Load V22-gamma checkpoint) ──
    model = IonisGate(
        dnn_dim=CONFIG["model"]["dnn_dim"],
        sidecar_hidden=CONFIG["model"]["sidecar_hidden"],
        sfi_idx=CONFIG["model"]["sfi_idx"],
        kp_penalty_idx=CONFIG["model"]["kp_penalty_idx"],
    ).to(DEVICE)

    # Load V22-gamma weights
    base_ckpt = os.path.join(SCRIPT_DIR, CONFIG["base_checkpoint"])
    log.info(f"Loading base checkpoint: {base_ckpt}")
    state_dict = load_safetensors(base_ckpt)
    model.load_state_dict(state_dict)
    log.info("  V22-gamma weights loaded successfully")

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model: IonisGate ({total_params:,} params)")

    # ── Optimizer (reduced LR for fine-tuning) ──
    param_groups = get_optimizer_groups(
        model,
        trunk_lr=CONFIG["training"]["trunk_lr"],
        scaler_lr=CONFIG["training"]["scaler_lr"],
        sidecar_lr=CONFIG["training"]["sidecar_lr"],
    )
    optimizer = optim.AdamW(param_groups, weight_decay=CONFIG["training"]["weight_decay"])

    epochs = CONFIG["training"]["epochs"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    # ── Physics-Informed Loss ──
    physics = CONFIG["physics_loss"]
    criterion = PhysicsInformedLoss(
        delta=CONFIG["training"]["huber_delta"],
        penalty_factor=physics["penalty_factor"],
        closure_threshold=physics["closure_threshold"],
        freq_threshold=physics["freq_mhz_threshold"],
        solar_threshold=physics["solar_elev_threshold"],
    )
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
           f"{'SFI+':>5s}  {'Kp9-':>5s}  {'Viol%':>6s}  {'Time':>6s}")
    log.info(hdr)
    log.info("-" * len(hdr))

    total_violations = 0
    total_samples = 0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        # Train
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        epoch_violations = 0
        epoch_samples = 0

        for bx, baux, by, bw in train_loader:
            bx = bx.to(DEVICE)
            baux = baux.to(DEVICE)
            by = by.to(DEVICE)
            bw = bw.to(DEVICE)

            optimizer.zero_grad()

            # Forward with gates for variance loss
            out, sun_gate, storm_gate = model.forward_with_gates(bx)

            # Physics-informed loss (pass auxiliary columns)
            primary_loss = criterion(
                out, by,
                freq_mhz=baux[:, 0],
                tx_solar=baux[:, 1],
                rx_solar=baux[:, 2],
                weights=bw,
            )
            var_loss = -lambda_var * (sun_gate.var() + storm_gate.var())
            loss = primary_loss + var_loss

            loss.backward()
            optimizer.step()

            # Weight clamp
            clamp_sidecars(model)

            train_loss_sum += primary_loss.item()
            train_batches += 1
            epoch_violations += criterion._last_violation_count
            epoch_samples += criterion._last_batch_size

        train_loss = train_loss_sum / train_batches
        violation_rate = epoch_violations / epoch_samples if epoch_samples > 0 else 0.0
        total_violations += epoch_violations
        total_samples += epoch_samples

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for bx, baux, by, bw in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out = model(bx)
                # Use base Huber for validation (no penalty)
                val_loss_sum += nn.HuberLoss()(out, by).item()
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
                'base_checkpoint': CONFIG['base_checkpoint'],
                'date_range': date_range,
                'sample_size': n,
                'val_rmse': val_rmse,
                'val_pearson': val_pearson,
                'sfi_benefit': sfi_benefit,
                'storm_cost': storm_cost,
                'physics_loss': physics,
            }
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            marker = " *"

        log.info(
            f"{epoch:3d}  {train_loss:8.4f}  {val_loss:8.4f}  "
            f"{val_rmse:6.3f}σ  {val_pearson:+7.4f}  "
            f"{sfi_benefit:+4.2f}  {storm_cost:+4.2f}  "
            f"{violation_rate:5.2%}  {epoch_sec:5.1f}s{marker}"
        )

        audit_epoch(run_id, epoch, train_loss, val_loss, val_rmse, val_pearson,
                    sfi_benefit, storm_cost, epoch_sec, is_best)

    # ── Final Report ──
    best_rmse = np.sqrt(best_val_loss)
    wall_seconds = time.perf_counter() - training_start_time
    total_violation_rate = total_violations / total_samples if total_samples > 0 else 0.0

    log.info("-" * len(hdr))
    log.info(f"Training complete. Best RMSE: {best_rmse:.4f}σ, Pearson: {best_pearson:+.4f}")
    log.info(f"Total violation rate: {total_violation_rate:.2%}")
    log.info(f"Checkpoint: {model_path}")

    log.info("")
    log.info("=" * 70)
    log.info("V27 RESULTS")
    log.info("=" * 70)

    log.info("")
    log.info("SUCCESS CRITERIA:")
    log.info(f"  RMSE:    {best_rmse:.3f}σ")
    log.info(f"  Pearson: {best_pearson:+.4f}")
    log.info(f"  SFI:     {best_sfi:+.3f}σ")
    log.info(f"  Kp:      {best_kp:+.3f}σ")
    log.info("")
    log.info(">>> Run ACID TEST: DN13→JN48, 10m, 02 UTC, Feb, SFI=150 must be ≤ 0.0σ <<<")
    log.info(">>> Run KI7MT 17-path operator tests (target: 17/17) <<<")
    log.info(">>> Run TST-900 (target: >= 9/11) <<<")
    log.info("")
    log.info("V22-gamma REFERENCE: Pearson +0.492, KI7MT 16/17, +0.540σ on acid test")
    log.info("V27 TARGET: acid test ≤ 0.0σ AND no regression on KI7MT")


if __name__ == '__main__':
    train()
