#!/usr/bin/env python3
"""
validate_v18.py — V18 Global Normalization Fix vs VOACAP comparison on Step I paths

Queries 1M contest paths from validation.step_i_paths, runs V18 inference,
denormalizes to dB using global_mean/global_std, applies mode thresholds.

V18 fix: Global raw-dB normalization (not per-source Z-score)
- Denormalization: snr_dB = prediction_σ × global_std + global_mean
- Predictions should be in real dB range (-30 to +30)
- Recall should be realistic (85-95%, not 100%)

Baselines:
- VOACAP: 75.82%
- IONIS V15: 86.89%
- IONIS V16: 96.38%
- IONIS V17: 100% (broken — normalization issue)

Target: 85-95% recall with realistic dB predictions
"""

import math
import os
import sys

import clickhouse_connect
import numpy as np
import torch
import torch.nn as nn

# ── Config ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "ionis_v18.pth")

CH_HOST = "10.60.1.1"
CH_PORT = 8123

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Band ID to frequency mapping (for freq_log calculation)
FREQ_MHZ = {
    102: 1.8, 103: 3.5, 104: 5.3, 105: 7.0, 106: 10.1,
    107: 14.0, 108: 18.1, 109: 21.0, 110: 24.9, 111: 28.0,
}

# MHz to band ID
MHZ_TO_BAND = {
    1.8: 102, 3.5: 103, 5.3: 104, 7.0: 105, 10.1: 106,
    14.0: 107, 18.1: 108, 21.0: 109, 24.9: 110, 28.0: 111,
}

# Mode thresholds in dB (from Step I)
THRESHOLDS = {
    'CW': -22.0,
    'DG': -22.0,  # Digital
    'PH': -21.0,  # Phone/SSB
    'RY': -21.0,  # RTTY
}

DNN_DIM = 11
SFI_IDX = 11
KP_PENALTY_IDX = 12
INPUT_DIM = 13
GATE_INIT_BIAS = -math.log(2.0)


# ── Model Architecture (must match training) ─────────────────────────────────

class MonotonicMLP(nn.Module):
    def __init__(self, hidden_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=True)
        self.activation = nn.Softplus()

    def forward(self, x):
        w1 = torch.abs(self.fc1.weight)
        w2 = torch.abs(self.fc2.weight)
        h = self.activation(nn.functional.linear(x, w1, self.fc1.bias))
        return nn.functional.linear(h, w2, self.fc2.bias)


def _gate(x):
    return 0.5 + 1.5 * torch.sigmoid(x)


class IonisV12Gate(nn.Module):
    def __init__(self, dnn_dim=11, sidecar_hidden=8):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dnn_dim, 512), nn.Mish(),
            nn.Linear(512, 256), nn.Mish(),
        )
        self.base_head = nn.Sequential(
            nn.Linear(256, 128), nn.Mish(),
            nn.Linear(128, 1),
        )
        self.sun_scaler_head = nn.Sequential(
            nn.Linear(256, 64), nn.Mish(),
            nn.Linear(64, 1),
        )
        self.storm_scaler_head = nn.Sequential(
            nn.Linear(256, 64), nn.Mish(),
            nn.Linear(64, 1),
        )
        self.sun_sidecar = MonotonicMLP(hidden_dim=sidecar_hidden)
        self.storm_sidecar = MonotonicMLP(hidden_dim=sidecar_hidden)
        self._init_scaler_heads()

    def _init_scaler_heads(self):
        for head in [self.sun_scaler_head, self.storm_scaler_head]:
            final_layer = head[-1]
            nn.init.zeros_(final_layer.weight)
            nn.init.constant_(final_layer.bias, GATE_INIT_BIAS)

    def forward(self, x):
        x_deep = x[:, :DNN_DIM]
        x_sfi = x[:, SFI_IDX:SFI_IDX + 1]
        x_kp = x[:, KP_PENALTY_IDX:KP_PENALTY_IDX + 1]
        trunk_out = self.trunk(x_deep)
        base_snr = self.base_head(trunk_out)
        sun_logit = self.sun_scaler_head(trunk_out)
        storm_logit = self.storm_scaler_head(trunk_out)
        sun_gate = _gate(sun_logit)
        storm_gate = _gate(storm_logit)
        return base_snr + sun_gate * self.sun_sidecar(x_sfi) + \
               storm_gate * self.storm_sidecar(x_kp)


# ── Feature Engineering ──────────────────────────────────────────────────────

def engineer_features(tx_grid, rx_grid, band_id, hour, month, sfi, kp, distance, azimuth):
    """Build feature vector for a single path."""
    # Grid to lat/lon
    def grid4_to_latlon(g):
        s = str(g).strip().upper()[:4]
        if len(s) < 4:
            s = 'JJ00'
        lon = (ord(s[0]) - ord('A')) * 20.0 - 180.0 + int(s[2]) * 2.0 + 1.0
        lat = (ord(s[1]) - ord('A')) * 10.0 - 90.0 + int(s[3]) * 1.0 + 0.5
        return lat, lon

    tx_lat, tx_lon = grid4_to_latlon(tx_grid)
    rx_lat, rx_lon = grid4_to_latlon(rx_grid)

    midpoint_lat = (tx_lat + rx_lat) / 2.0
    midpoint_lon = (tx_lon + rx_lon) / 2.0

    freq_mhz = FREQ_MHZ.get(band_id, 14.0)
    freq_hz = freq_mhz * 1e6
    kp_penalty = 1.0 - kp / 9.0

    return np.array([
        distance / 20000.0,
        np.log10(freq_hz) / 8.0,
        np.sin(2.0 * np.pi * hour / 24.0),
        np.cos(2.0 * np.pi * hour / 24.0),
        np.sin(2.0 * np.pi * azimuth / 360.0),
        np.cos(2.0 * np.pi * azimuth / 360.0),
        abs(tx_lat - rx_lat) / 180.0,
        midpoint_lat / 90.0,
        np.sin(2.0 * np.pi * month / 12.0),
        np.cos(2.0 * np.pi * month / 12.0),
        np.cos(2.0 * np.pi * (hour + midpoint_lon / 15.0) / 24.0),
        sfi / 300.0,
        kp_penalty,
    ], dtype=np.float32)


def engineer_features_batch(df):
    """Build feature matrix for a batch of paths."""
    n = len(df)
    features = np.zeros((n, INPUT_DIM), dtype=np.float32)

    for i in range(n):
        row = df.iloc[i]
        features[i] = engineer_features(
            row['tx_grid'], row['rx_grid'],
            row['band_id'], row['hour'], row['month'],
            row['sfi'], row['kp'],
            row['distance'], row['azimuth']
        )

    return features


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  IONIS V18 Global Normalization Fix — Step I Validation")
    print("=" * 70)
    print()

    # Load checkpoint
    print(f"Loading checkpoint: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Checkpoint not found at {MODEL_PATH}")
        return 1

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    # V18: Use global_mean and global_std (raw dB)
    global_mean = checkpoint.get('global_mean')
    global_std = checkpoint.get('global_std')

    if global_mean is None or global_std is None:
        print("ERROR: Checkpoint missing global_mean or global_std")
        print("  This checkpoint may not be a V18 model.")
        return 1

    print(f"  global_mean = {global_mean:.2f} dB")
    print(f"  global_std  = {global_std:.2f} dB")
    print(f"  Denormalization: snr_dB = pred_σ × {global_std:.2f} + {global_mean:.2f}")
    print()

    # Verify expected ranges (from 9975WX spec)
    if not (-5 <= global_mean <= 5):
        print(f"  WARNING: global_mean ({global_mean:.2f}) outside expected range [-5, +5]")
    if not (15 <= global_std <= 25):
        print(f"  WARNING: global_std ({global_std:.2f}) outside expected range [15, 25]")

    # Load model
    model = IonisV12Gate(
        dnn_dim=checkpoint.get('dnn_dim', DNN_DIM),
        sidecar_hidden=checkpoint.get('sidecar_hidden', 8)
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: IonisV12Gate ({total_params:,} params)")
    print()

    # Load validation paths
    print(f"Connecting to ClickHouse at {CH_HOST}:{CH_PORT}...")
    client = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT)

    query = """
    SELECT
        tx_grid, rx_grid, band_mhz, mode, hour, month,
        sfi, kp, distance, azimuth
    FROM validation.step_i_paths
    """

    print("Loading Step I paths...")
    df = client.query_df(query)
    client.close()

    n_samples = len(df)
    print(f"  Loaded {n_samples:,} paths")
    print()

    # Map band_mhz to band_id
    df['band_id'] = df['band_mhz'].map(MHZ_TO_BAND)

    # Map mode to threshold
    threshold = df['mode'].map(THRESHOLDS).values.astype(np.float32)
    mode = df['mode'].values
    band_ids = df['band_id'].values

    # Engineer features
    print("Engineering features...")
    features = engineer_features_batch(df)
    print(f"  Features shape: {features.shape}")
    print()

    # Run inference
    print("Running inference...")
    batch_size = 65536
    predictions_sigma = np.zeros(n_samples, dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            batch = torch.tensor(features[i:end], dtype=torch.float32, device=DEVICE)
            pred = model(batch).cpu().numpy().flatten()
            predictions_sigma[i:end] = pred
            if (i // batch_size) % 5 == 0:
                print(f"  Processed {end:,} / {n_samples:,} ({100*end/n_samples:.1f}%)")

    print(f"  Inference complete")
    print(f"  Predictions (σ): min={predictions_sigma.min():.3f}, max={predictions_sigma.max():.3f}, mean={predictions_sigma.mean():.3f}")
    print()

    # V18: Denormalize using global_mean and global_std (raw dB)
    print("Denormalizing predictions to dB (V18 global normalization)...")
    predictions_db = predictions_sigma * global_std + global_mean

    print(f"  Predictions (dB): min={predictions_db.min():.1f}, max={predictions_db.max():.1f}, mean={predictions_db.mean():.1f}")
    print(f"  Thresholds: min={threshold.min():.1f}, max={threshold.max():.1f}")
    print()

    # Verify predictions are in expected dB range
    if predictions_db.min() > -10 or predictions_db.max() < 0:
        print("  WARNING: Predictions may not be in expected dB range (-30 to +30)")

    # Apply thresholds
    print("Applying mode thresholds...")
    band_open = predictions_db >= threshold

    # Calculate recall
    total = len(band_open)
    open_count = band_open.sum()
    recall = 100.0 * open_count / total

    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print()
    print(f"  Total paths:     {total:,}")
    print(f"  Band open:       {open_count:,}")
    print(f"  V18 Recall:      {recall:.2f}%")
    print()
    print("  Baselines:")
    print(f"    VOACAP:        75.82%")
    print(f"    IONIS V15:     86.89%")
    print(f"    IONIS V16:     96.38%")
    print(f"    IONIS V17:     100.00% (broken)")
    print()

    delta_voacap = recall - 75.82
    delta_v15 = recall - 86.89
    delta_v16 = recall - 96.38
    print(f"  V18 vs VOACAP:   {delta_voacap:+.2f} pp")
    print(f"  V18 vs V15:      {delta_v15:+.2f} pp")
    print(f"  V18 vs V16:      {delta_v16:+.2f} pp")
    print()

    # Breakdown by mode
    print("  Recall by Mode:")
    for m in ['CW', 'PH', 'RY', 'DG']:
        mask = mode == m
        if mask.sum() > 0:
            mode_recall = 100.0 * band_open[mask].sum() / mask.sum()
            print(f"    {m}:  {mode_recall:.2f}% ({mask.sum():,} paths)")
    print()

    # Breakdown by band
    print("  Recall by Band:")
    band_names = {102: '160m', 103: '80m', 105: '40m', 107: '20m', 109: '15m', 111: '10m'}
    for band_id, band_name in sorted(band_names.items()):
        mask = band_ids == band_id
        if mask.sum() > 0:
            band_recall = 100.0 * band_open[mask].sum() / mask.sum()
            print(f"    {band_name}:  {band_recall:.2f}% ({mask.sum():,} paths)")
    print()

    # Prediction distribution check
    print("  Prediction Distribution (dB):")
    percentiles = [5, 25, 50, 75, 95]
    pcts = np.percentile(predictions_db, percentiles)
    for p, v in zip(percentiles, pcts):
        print(f"    p{p:02d}: {v:+.1f} dB")
    print()

    print("=" * 70)

    # V18 specific checks
    if recall > 99.0:
        print("  STATUS: WARNING — Recall > 99% suggests normalization may still be broken")
        return 1
    elif recall >= 80.0:
        print("  STATUS: PASS (recall >= 80% and < 99%)")
        return 0
    else:
        print("  STATUS: FAIL (recall < 80%)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
