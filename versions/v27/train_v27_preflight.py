#!/usr/bin/env python3
"""
train_v27_preflight.py — V27 Pre-Flight on 9975WX (RTX PRO 6000)

Validates the PhysicsInformedLoss on a 1M-row sample before M3 commits
to a full 30-epoch fine-tune from V22-gamma.

Pre-flight checklist:
  - [ ] Loss decreasing epoch-over-epoch
  - [ ] Violation percentage decreasing epoch-over-epoch
  - [ ] Acid test prediction trending negative
  - [ ] No NaN/Inf in gradients
  - [ ] No CUDA device errors
  - [ ] Row count correct

Usage:
    /mnt/ai-stack/ionis-ai/.cuda-venv/bin/python versions/v27/train_v27_preflight.py
"""

import gc
import json
import logging
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file as load_safetensors

# ── Path Setup ────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(SCRIPT_DIR)
COMMON_DIR = os.path.join(TRAINING_DIR, "common")
sys.path.insert(0, COMMON_DIR)

from model import (
    MonotonicMLP, IonisGate, get_device, build_features,
    grid4_to_latlon, BAND_FREQ_HZ,
)
from train_common import (
    engineer_features, solar_elevation_vectorized,
    grid4_to_latlon_arrays, load_source_data,
)

import clickhouse_connect

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

# V22-gamma config (locked architecture)
V22_CONFIG_PATH = os.path.join(TRAINING_DIR, "v22", "config_v22.json")
V22_CHECKPOINT_PATH = os.path.join(TRAINING_DIR, "v22", "ionis_v22_gamma.safetensors")

# Pre-flight parameters
PREFLIGHT_EPOCHS = 10  # More room for acid test to converge
PREFLIGHT_WSPR_SAMPLE = 1_000_000  # 1M rows (vs 20M production)
BATCH_SIZE = 65536
VAL_SPLIT = 0.2

# V27 physics-informed loss parameters
PENALTY_FACTOR = 10.0  # Baseline eval proved penalty isn't the issue (0.75% violation rate)
CLOSURE_THRESHOLD = -1.0  # σ units
SOLAR_ELEV_THRESHOLD = -6.0  # degrees (civil twilight)
FREQ_MHZ_THRESHOLD = 21.0  # 15m and above

# Fine-tune learning rates (10% of V22-gamma)
TRUNK_LR = 1e-6
SIDECAR_LR = 1e-4

# ClickHouse on localhost (9975WX)
CH_HOST = "192.168.1.90"
CH_PORT = 8123

# WSPR decode floor
WSPR_DECODE_FLOOR_DB = -28.0


# ── Physics-Informed Loss ─────────────────────────────────────────────────────

class PhysicsInformedLoss(nn.Module):
    """
    HuberLoss + diurnal physics penalty.

    When the model predicts viable signal on a high-frequency band
    during darkness at both endpoints, the loss for those rows is
    multiplied by a penalty factor.

    V27 Rule (one rule, one variable):
        IF freq_mhz >= 21.0 (15m, 12m, 10m)
        AND tx_solar_elevation < -6.0 (civil twilight or darker)
        AND rx_solar_elevation < -6.0 (civil twilight or darker)
        AND predicted_snr > closure_threshold
        THEN loss *= penalty_factor
    """

    def __init__(self, delta=1.0, penalty_factor=10.0, closure_threshold=-1.0):
        super().__init__()
        self.base_loss = nn.HuberLoss(delta=delta, reduction='none')
        self.penalty_factor = penalty_factor
        self.closure_threshold = closure_threshold
        self._last_violation_count = 0
        self._last_batch_size = 0
        self._last_violation_by_band = {}

    def forward(self, pred, target, freq_mhz, tx_solar, rx_solar, weights=None):
        """
        Args:
            pred: [batch, 1] predicted SNR (σ)
            target: [batch, 1] actual SNR (σ)
            freq_mhz: [batch] frequency in MHz (RAW, not normalized)
            tx_solar: [batch] TX solar elevation in degrees (RAW)
            rx_solar: [batch] RX solar elevation in degrees (RAW)
            weights: [batch, 1] optional sample weights

        Returns:
            Scalar loss value
        """
        base = self.base_loss(pred, target)

        # Physics violation mask
        high_band = freq_mhz >= FREQ_MHZ_THRESHOLD
        tx_dark = tx_solar < SOLAR_ELEV_THRESHOLD
        rx_dark = rx_solar < SOLAR_ELEV_THRESHOLD
        pred_positive = pred.squeeze() > self.closure_threshold

        # Violation: high band + both endpoints dark + model says "open"
        violation = high_band & tx_dark & rx_dark & pred_positive

        # Apply penalty multiplier
        multiplier = torch.ones_like(base)
        multiplier[violation.unsqueeze(1)] = self.penalty_factor

        penalized = base * multiplier

        if weights is not None:
            penalized = penalized * weights

        # Track violations for logging
        self._last_violation_count = violation.sum().item()
        self._last_batch_size = pred.shape[0]

        # Per-band violation tracking
        self._last_violation_by_band = {}
        for band_name, freq_lo, freq_hi in [
            ("15m", 21.0, 21.5), ("12m", 24.5, 25.5), ("10m", 28.0, 29.0)
        ]:
            band_mask = (freq_mhz >= freq_lo) & (freq_mhz < freq_hi)
            band_violations = (violation & band_mask).sum().item()
            band_total = band_mask.sum().item()
            self._last_violation_by_band[band_name] = (band_violations, band_total)

        return penalized.mean()

    @property
    def violation_rate(self):
        if self._last_batch_size > 0:
            return self._last_violation_count / self._last_batch_size
        return 0.0


# ── Gate function ─────────────────────────────────────────────────────────────

def _gate(x):
    return 0.5 + 1.5 * torch.sigmoid(x)


# ── Dataset with auxiliary columns ────────────────────────────────────────────

class SignatureDatasetWithAux(Dataset):
    """PyTorch Dataset that also carries auxiliary columns for the loss."""

    def __init__(self, X, y, w, aux):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.w = torch.tensor(w, dtype=torch.float32)
        self.aux = torch.tensor(aux, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.w[idx], self.aux[idx]


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_preflight_data(config):
    """Load 1M WSPR sample with auxiliary columns for physics loss.

    Returns:
        X, y, w, aux — where aux[:, 0] = freq_mhz, aux[:, 1] = tx_solar, aux[:, 2] = rx_solar
    """
    log.info(f"Connecting to ClickHouse at {CH_HOST}:{CH_PORT}...")
    client = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT)

    # Load 1M WSPR signatures with day_of_year
    wspr_df = load_source_data(
        client, "wspr.signatures_v2_terrestrial",
        sample_size=PREFLIGHT_WSPR_SAMPLE,
        include_day_of_year=True,
    )

    client.close()

    # ── Normalize SNR ──
    norms = config["norm_constants_per_band"]
    band = wspr_df['band'].values.astype(np.int32)
    snr_raw = wspr_df['median_snr'].values.astype(np.float32)

    # Per-band Z-score normalization using config constants
    y = np.zeros(len(wspr_df), dtype=np.float32)
    for band_id_str, band_norms in norms.items():
        band_id = int(band_id_str)
        mask = band == band_id
        if mask.sum() > 0:
            wspr_norms = band_norms.get("wspr", band_norms)
            mean = wspr_norms["mean"]
            std = wspr_norms["std"]
            y[mask] = (snr_raw[mask] - mean) / std

    # ── Engineer features ──
    X = engineer_features(wspr_df, config)

    # ── Weights ──
    w = wspr_df['spot_count'].values.astype(np.float32)
    w = w / w.mean()

    # ── Compute auxiliary columns (RAW values, NOT from normalized features) ──
    band_to_hz = config["band_to_hz"]
    if isinstance(list(band_to_hz.keys())[0], str):
        band_to_hz = {int(k): v for k, v in band_to_hz.items()}

    freq_hz = np.array([band_to_hz.get(b, 14_097_100) for b in band], dtype=np.float32)
    freq_mhz = freq_hz / 1e6

    tx_lats, tx_lons = grid4_to_latlon_arrays(wspr_df['tx_grid_4'].values)
    rx_lats, rx_lons = grid4_to_latlon_arrays(wspr_df['rx_grid_4'].values)
    hour = wspr_df['hour'].values.astype(np.float32)

    if 'day_of_year' in wspr_df.columns:
        day_of_year = wspr_df['day_of_year'].values.astype(np.float32)
    else:
        month = wspr_df['month'].values.astype(np.float32)
        day_of_year = (month - 1) * 30.5 + 15

    tx_solar = solar_elevation_vectorized(tx_lats, tx_lons, hour, day_of_year)
    rx_solar = solar_elevation_vectorized(rx_lats, rx_lons, hour, day_of_year)

    aux = np.zeros((len(wspr_df), 3), dtype=np.float32)
    aux[:, 0] = freq_mhz
    aux[:, 1] = tx_solar
    aux[:, 2] = rx_solar

    # Reshape
    y = y.reshape(-1, 1)
    w = w.reshape(-1, 1)

    # Free DataFrame
    del wspr_df
    gc.collect()

    log.info(f"Data loaded: {len(X):,} rows, {X.shape[1]} features")
    log.info(f"  freq_mhz range: {freq_mhz.min():.1f} - {freq_mhz.max():.1f}")
    log.info(f"  tx_solar range: {tx_solar.min():.1f}° to {tx_solar.max():.1f}°")
    log.info(f"  rx_solar range: {rx_solar.min():.1f}° to {rx_solar.max():.1f}°")

    # Count potential violations in the data
    high_band_mask = freq_mhz >= FREQ_MHZ_THRESHOLD
    both_dark = (tx_solar < SOLAR_ELEV_THRESHOLD) & (rx_solar < SOLAR_ELEV_THRESHOLD)
    potential_violations = (high_band_mask & both_dark).sum()
    log.info(f"  Potential violation rows (high band + both dark): {potential_violations:,} "
             f"({100*potential_violations/len(X):.2f}%)")

    return X, y, w, aux


# ── Acid Test ─────────────────────────────────────────────────────────────────

def run_acid_test(model, config, device):
    """DN13→JN48, 10m, 02 UTC, February, SFI=150 — must be ≤ 0σ."""
    tx_lat, tx_lon = grid4_to_latlon("DN13")
    rx_lat, rx_lon = grid4_to_latlon("JN48")
    freq_hz = BAND_FREQ_HZ["10m"]

    features = build_features(
        tx_lat, tx_lon, rx_lat, rx_lon, freq_hz,
        sfi=150, kp=2, hour_utc=2, month=2,
        day_of_year=46,
        include_solar_depression=True,
    )

    tensor = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        sigma = model(tensor).item()

    return sigma


# ── Main Pre-Flight ───────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("V27 PRE-FLIGHT — Physics-Informed Loss Validation")
    log.info("=" * 70)
    log.info(f"Host: 9975WX (RTX PRO 6000 Blackwell)")
    log.info(f"Epochs: {PREFLIGHT_EPOCHS}")
    log.info(f"WSPR sample: {PREFLIGHT_WSPR_SAMPLE:,}")
    log.info(f"Penalty factor: {PENALTY_FACTOR}")
    log.info(f"Closure threshold: {CLOSURE_THRESHOLD}σ")
    log.info(f"Solar elevation threshold: {SOLAR_ELEV_THRESHOLD}°")
    log.info(f"Frequency threshold: {FREQ_MHZ_THRESHOLD} MHz")
    log.info("")

    # ── Device ──
    device = get_device()
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"  Compute: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")
    log.info("")

    # ── Load V22-gamma config ──
    with open(V22_CONFIG_PATH) as f:
        config = json.load(f)

    # Override ClickHouse host for local execution
    config["clickhouse"]["host"] = CH_HOST
    config["clickhouse"]["port"] = CH_PORT

    log.info(f"Config: {V22_CONFIG_PATH}")
    log.info(f"Checkpoint: {V22_CHECKPOINT_PATH}")
    log.info(f"Architecture: {config['model']['architecture']}")
    log.info(f"dnn_dim: {config['model']['dnn_dim']}, input_dim: {config['model']['input_dim']}")
    log.info("")

    # ── CRITICAL: Verify closure_threshold dB mapping ──
    log.info("=" * 70)
    log.info("CLOSURE THRESHOLD VERIFICATION")
    log.info("=" * 70)

    norms = config["norm_constants_per_band"]
    for band_id, band_name, freq in [("109", "15m", 21.1), ("110", "12m", 24.9), ("111", "10m", 28.1)]:
        wspr_mean = norms[band_id]["wspr"]["mean"]
        wspr_std = norms[band_id]["wspr"]["std"]
        raw_db_at_threshold = wspr_mean + (CLOSURE_THRESHOLD * wspr_std)
        decode_floor_sigma = (WSPR_DECODE_FLOOR_DB - wspr_mean) / wspr_std
        log.info(f"  {band_name} ({band_id}): {CLOSURE_THRESHOLD}σ = {raw_db_at_threshold:.1f} dB | "
                 f"decode floor ({WSPR_DECODE_FLOOR_DB} dB) = {decode_floor_sigma:.2f}σ")

    log.info("")
    log.info("Plan's comment said -1.0σ = -0.95 dB — INCORRECT (used normalized target stats).")
    log.info("Actual: -1.0σ ≈ -24.5 dB (3.5 dB ABOVE decode floor).")
    log.info("This means penalty fires when model predicts a DETECTABLE signal on a dead band.")
    log.info("On a physically closed band, ANY detectable signal prediction is wrong.")
    log.info("Threshold is VALID. Decode floor would be ~-1.5σ if tighter threshold needed.")
    log.info("")

    # ── Load data ──
    X, y, w, aux = load_preflight_data(config)

    # ── Train/val split ──
    n = len(X)
    n_val = int(n * VAL_SPLIT)
    indices = np.random.RandomState(42).permutation(n)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_ds = SignatureDatasetWithAux(X[train_idx], y[train_idx], w[train_idx], aux[train_idx])
    val_ds = SignatureDatasetWithAux(X[val_idx], y[val_idx], w[val_idx], aux[val_idx])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    del X, y, w, aux
    gc.collect()

    log.info(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    log.info("")

    # ── Load V22-gamma model ──
    model_cfg = config["model"]
    model = IonisGate(
        dnn_dim=model_cfg["dnn_dim"],
        sidecar_hidden=model_cfg["sidecar_hidden"],
        sfi_idx=model_cfg["sfi_idx"],
        kp_penalty_idx=model_cfg["kp_penalty_idx"],
        gate_init_bias=model_cfg.get("gate_init_bias"),
    )

    state_dict = load_safetensors(V22_CHECKPOINT_PATH, device="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    log.info("V22-gamma checkpoint loaded.")

    # ── Acid test BEFORE fine-tuning ──
    acid_before = run_acid_test(model, config, device)
    log.info(f"Acid test (BEFORE): {acid_before:+.3f}σ {'PASS' if acid_before <= 0 else 'FAIL'}")
    log.info(f"  (V22-gamma reference: +0.540σ)")
    log.info("")

    # ── Baseline evaluation (V22-gamma on this 1M sample, BEFORE any training) ──
    log.info("=" * 70)
    log.info("BASELINE EVALUATION (V22-gamma on 1M WSPR, no fine-tuning)")
    log.info("=" * 70)
    model.eval()
    baseline_preds = []
    baseline_targets = []
    with torch.no_grad():
        for X_batch, y_batch, w_batch, aux_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(X_batch)
            baseline_preds.append(pred.cpu().numpy())
            baseline_targets.append(y_batch.cpu().numpy())
    baseline_preds = np.vstack(baseline_preds)
    baseline_targets = np.vstack(baseline_targets)
    baseline_rmse = np.sqrt(np.mean((baseline_preds - baseline_targets) ** 2))
    baseline_pearson = np.corrcoef(baseline_preds.ravel(), baseline_targets.ravel())[0, 1]
    log.info(f"  Baseline RMSE:    {baseline_rmse:.3f}σ")
    log.info(f"  Baseline Pearson: {baseline_pearson:+.4f}")
    log.info(f"  Pred range:       [{baseline_preds.min():.3f}, {baseline_preds.max():.3f}]")
    log.info(f"  Target range:     [{baseline_targets.min():.3f}, {baseline_targets.max():.3f}]")
    log.info(f"  Pred mean/std:    {baseline_preds.mean():.3f} / {baseline_preds.std():.3f}")
    log.info(f"  Target mean/std:  {baseline_targets.mean():.3f} / {baseline_targets.std():.3f}")
    if baseline_rmse > 1.0:
        log.warning("V22-gamma RMSE > 1.0σ on this sample — metrics may reflect data mix, not model quality")
    log.info("")

    # ── Setup optimizer (fine-tune rates: 10% of V22-gamma) ──
    trunk_params = list(model.trunk.parameters()) + list(model.base_head.parameters())
    gate_params = list(model.sun_scaler_head.parameters()) + list(model.storm_scaler_head.parameters())
    sun_params = list(model.sun_sidecar.parameters())
    storm_params = list(model.storm_sidecar.parameters())

    optimizer = torch.optim.AdamW([
        {"params": trunk_params, "lr": TRUNK_LR, "weight_decay": 1e-5},
        {"params": gate_params, "lr": TRUNK_LR, "weight_decay": 1e-5},
        {"params": sun_params, "lr": SIDECAR_LR, "weight_decay": 0},
        {"params": storm_params, "lr": SIDECAR_LR, "weight_decay": 0},
    ])

    # ── Loss function ──
    criterion = PhysicsInformedLoss(
        delta=1.0,
        penalty_factor=PENALTY_FACTOR,
        closure_threshold=CLOSURE_THRESHOLD,
    )

    # Gate variance loss (V16 Physics Law)
    lambda_var = 0.001

    # ── Training Loop ──
    log.info("=" * 70)
    log.info("  Ep  Train    Val     RMSE    Pearson  SFI+   Kp-    Viol%   Acid    Time")
    log.info("-" * 70)

    results = []

    for epoch in range(1, PREFLIGHT_EPOCHS + 1):
        t0 = time.perf_counter()
        model.train()

        train_losses = []
        epoch_violations = 0
        epoch_samples = 0

        for X_batch, y_batch, w_batch, aux_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            w_batch = w_batch.to(device)
            aux_batch = aux_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)

            # Physics-informed loss
            loss = criterion(
                pred, y_batch,
                freq_mhz=aux_batch[:, 0],
                tx_solar=aux_batch[:, 1],
                rx_solar=aux_batch[:, 2],
                weights=w_batch,
            )

            # Gate variance loss (V16 Physics Law)
            trunk_out = model.trunk(X_batch[:, :model_cfg["dnn_dim"]])
            sun_gate = _gate(model.sun_scaler_head(trunk_out))
            storm_gate = _gate(model.storm_scaler_head(trunk_out))
            gate_var_loss = -(sun_gate.var() + storm_gate.var())
            total_loss = loss + lambda_var * gate_var_loss

            total_loss.backward()

            # Check for NaN gradients
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    log.error(f"NaN gradient in {name} at epoch {epoch}!")
                    sys.exit(1)

            optimizer.step()

            # Weight clamp (V16 Physics Law)
            with torch.no_grad():
                for sidecar in [model.sun_sidecar, model.storm_sidecar]:
                    for layer in [sidecar.fc1, sidecar.fc2]:
                        layer.weight.data.clamp_(0.5, 2.0)

            train_losses.append(loss.item())
            epoch_violations += criterion._last_violation_count
            epoch_samples += criterion._last_batch_size

        # ── Validation ──
        model.eval()
        val_preds = []
        val_targets = []
        val_losses = []

        with torch.no_grad():
            for X_batch, y_batch, w_batch, aux_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                w_batch = w_batch.to(device)
                aux_batch = aux_batch.to(device)

                pred = model(X_batch)
                vloss = criterion(
                    pred, y_batch,
                    freq_mhz=aux_batch[:, 0],
                    tx_solar=aux_batch[:, 1],
                    rx_solar=aux_batch[:, 2],
                    weights=w_batch,
                )
                val_losses.append(vloss.item())
                val_preds.append(pred.cpu().numpy())
                val_targets.append(y_batch.cpu().numpy())

        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)

        # Metrics
        val_rmse = np.sqrt(np.mean((val_preds - val_targets) ** 2))
        pearson = np.corrcoef(val_preds.ravel(), val_targets.ravel())[0, 1]

        # Sidecar metrics
        with torch.no_grad():
            test_sfi = torch.linspace(0.01, 0.9, 50, device=device).unsqueeze(1)
            sfi_out = model.sun_sidecar(test_sfi)
            sfi_benefit = sfi_out.max().item() - sfi_out.min().item()

            test_kp = torch.linspace(0.01, 0.9, 50, device=device).unsqueeze(1)
            kp_out = model.storm_sidecar(test_kp)
            storm_cost = kp_out.max().item() - kp_out.min().item()

        # Acid test
        acid_sigma = run_acid_test(model, config, device)

        viol_pct = 100 * epoch_violations / epoch_samples if epoch_samples > 0 else 0

        elapsed = time.perf_counter() - t0

        # Mark best
        marker = " *" if epoch == 1 or val_rmse < min(r["rmse"] for r in results) else ""

        log.info(f"  {epoch:2d}   {np.mean(train_losses):.4f}  {np.mean(val_losses):.4f}  "
                 f"{val_rmse:.3f}σ  {pearson:+.4f}  {sfi_benefit:+.2f}  {storm_cost:+.2f}  "
                 f"{viol_pct:5.2f}%  {acid_sigma:+.3f}σ  {elapsed:.1f}s{marker}")

        results.append({
            "epoch": epoch,
            "train_loss": np.mean(train_losses),
            "val_loss": np.mean(val_losses),
            "rmse": val_rmse,
            "pearson": pearson,
            "sfi_benefit": sfi_benefit,
            "storm_cost": storm_cost,
            "viol_pct": viol_pct,
            "acid_sigma": acid_sigma,
            "elapsed": elapsed,
        })

    # ── Summary ──
    log.info("")
    log.info("=" * 70)
    log.info("V27 PRE-FLIGHT RESULTS")
    log.info("=" * 70)

    # Check criteria — RMSE compared against BASELINE, not V22-gamma's training-mix metrics
    loss_decreasing = results[-1]["train_loss"] < results[0]["train_loss"]
    viol_decreasing = results[-1]["viol_pct"] <= results[0]["viol_pct"]
    acid_trending = results[-1]["acid_sigma"] < results[0]["acid_sigma"]
    acid_passed = results[-1]["acid_sigma"] <= 0.0
    pearson_ok = results[-1]["pearson"] > 0.0
    rmse_improving = results[-1]["rmse"] < baseline_rmse  # Compare vs loaded checkpoint, not 0.821σ

    log.info(f"  Loss decreasing:      {'PASS' if loss_decreasing else 'FAIL'} "
             f"({results[0]['train_loss']:.4f} → {results[-1]['train_loss']:.4f})")
    log.info(f"  Violations stable/dec: {'PASS' if viol_decreasing else 'FAIL'} "
             f"({results[0]['viol_pct']:.2f}% → {results[-1]['viol_pct']:.2f}%)")
    log.info(f"  Acid trending neg:    {'PASS' if acid_trending else 'FAIL'} "
             f"({results[0]['acid_sigma']:+.3f}σ → {results[-1]['acid_sigma']:+.3f}σ)")
    log.info(f"  Acid test passed:     {'PASS' if acid_passed else 'not yet (OK for preflight)'} "
             f"({results[-1]['acid_sigma']:+.3f}σ)")
    log.info(f"  Pearson positive:     {'PASS' if pearson_ok else 'FAIL'} "
             f"({results[-1]['pearson']:+.4f})")
    log.info(f"  RMSE vs baseline:     {'PASS' if rmse_improving else 'FAIL'} "
             f"(baseline {baseline_rmse:.3f}σ → {results[-1]['rmse']:.3f}σ)")
    log.info("")

    # Per-band violation detail from last epoch
    log.info("  Per-band violations (last epoch training):")
    for band_name, (violations, total) in criterion._last_violation_by_band.items():
        pct = 100 * violations / total if total > 0 else 0
        log.info(f"    {band_name}: {violations}/{total} ({pct:.1f}%)")
    log.info("")

    # Overall verdict
    all_pass = loss_decreasing and viol_decreasing and acid_trending and pearson_ok and rmse_improving
    if all_pass:
        log.info("  >>> PRE-FLIGHT: PASS — Ready for M3 production run <<<")
        log.info("  Send config to M3 via messages.inbox")
    else:
        log.info("  >>> PRE-FLIGHT: NEEDS REVIEW <<<")
        if not acid_trending:
            log.info("  Acid test not trending — consider bumping trunk LR to 5e-6")
        if not viol_decreasing:
            log.info("  Violations not decreasing — penalty may be too weak or threshold too tight")

    log.info("")
    log.info("V22-gamma reference: Acid +0.540σ, RMSE 0.821σ, Pearson +0.492")
    log.info("=" * 70)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
