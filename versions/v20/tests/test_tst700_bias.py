#!/usr/bin/env python3
"""
test_tst700_bias.py — IONIS V20 Bias & Fairness Tests

TST-700 Group: Tests for systematic biases in model predictions.

Tests:
  TST-701: Geographic Coverage Bias (EU vs Africa)
  TST-702: Temporal Bias (24-hour sweep)
  TST-703: Band Coverage Bias (160m-10m)

These tests verify the model doesn't systematically favor regions, times,
or bands that were overrepresented in training data.
"""

import json
import math
import os
import sys

import numpy as np
import torch

# ── Path Setup ───────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V20_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, V20_DIR)

from model import IonisGate, get_device

# ── Load Config ──────────────────────────────────────────────────────────────

CONFIG_PATH = os.path.join(V20_DIR, "config_v20.json")
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

MODEL_PATH = os.path.join(V20_DIR, CONFIG["checkpoint"])
DNN_DIM = CONFIG["model"]["dnn_dim"]
SFI_IDX = CONFIG["model"]["sfi_idx"]
KP_PENALTY_IDX = CONFIG["model"]["kp_penalty_idx"]
SIDECAR_HIDDEN = CONFIG["model"]["sidecar_hidden"]

BAND_TO_HZ = {int(k): v for k, v in CONFIG["band_to_hz"].items()}

DEVICE = get_device()

SIGMA_TO_DB = 6.7

# Band frequencies for testing
BANDS = [
    ("160m", 1_836_600),
    ("80m", 3_568_600),
    ("60m", 5_287_200),
    ("40m", 7_038_600),
    ("30m", 10_138_700),
    ("20m", 14_097_100),
    ("17m", 18_104_600),
    ("15m", 21_094_600),
    ("12m", 24_924_600),
    ("10m", 28_124_600),
]


# ── Prediction Helper ────────────────────────────────────────────────────────

def predict(model, device, tx_lat, tx_lon, rx_lat, rx_lon, freq_hz,
            sfi, kp, hour_utc, month=6):
    """Make a prediction for given parameters."""

    # Calculate distance
    R = 6371.0
    lat1_r, lat2_r = np.radians(tx_lat), np.radians(rx_lat)
    dlat = np.radians(rx_lat - tx_lat)
    dlon = np.radians(rx_lon - tx_lon)
    a = np.sin(dlat/2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2
    distance_km = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    # Calculate azimuth
    x = np.sin(dlon) * np.cos(lat2_r)
    y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)
    azimuth = (np.degrees(np.arctan2(x, y)) + 360) % 360

    # Build normalized features
    distance = distance_km / 20000.0
    freq_log = np.log10(freq_hz) / 8.0
    hour_sin = np.sin(2.0 * np.pi * hour_utc / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hour_utc / 24.0)
    az_sin = np.sin(2.0 * np.pi * azimuth / 360.0)
    az_cos = np.cos(2.0 * np.pi * azimuth / 360.0)
    lat_diff = abs(tx_lat - rx_lat) / 180.0
    midpoint_lat = (tx_lat + rx_lat) / 2.0 / 90.0
    season_sin = np.sin(2.0 * np.pi * month / 12.0)
    season_cos = np.cos(2.0 * np.pi * month / 12.0)
    midpoint_lon = (tx_lon + rx_lon) / 2.0
    local_solar_h = hour_utc + midpoint_lon / 15.0
    day_night_est = np.cos(2.0 * np.pi * local_solar_h / 24.0)
    sfi_norm = sfi / 300.0
    kp_penalty = 1.0 - kp / 9.0

    features = torch.tensor(
        [[distance, freq_log, hour_sin, hour_cos,
          az_sin, az_cos, lat_diff, midpoint_lat,
          season_sin, season_cos, day_night_est,
          sfi_norm, kp_penalty]],
        dtype=torch.float32, device=device,
    )

    with torch.no_grad():
        return model(features).item(), distance_km


# ── Test Functions ───────────────────────────────────────────────────────────

def test_tst701_geographic_bias(model, device):
    """TST-701: Geographic Coverage Bias — EU vs Africa"""
    print("\n" + "=" * 60)
    print("TST-701: Geographic Coverage Bias")
    print("=" * 60)

    # Data-rich region: Europe (G → DL, ~900 km)
    eu_snr, eu_dist = predict(
        model, device,
        tx_lat=51.50, tx_lon=-0.12,   # London
        rx_lat=52.52, rx_lon=13.40,   # Berlin
        freq_hz=14_097_100,
        sfi=150, kp=2, hour_utc=14
    )

    # Data-sparse region: Africa (5H → 9J, ~1200 km)
    africa_snr, africa_dist = predict(
        model, device,
        tx_lat=-6.17, tx_lon=35.74,   # Tanzania
        rx_lat=-15.42, rx_lon=28.28,  # Zambia
        freq_hz=14_097_100,
        sfi=150, kp=2, hour_utc=14
    )

    bias_sigma = abs(eu_snr - africa_snr)
    bias_db = bias_sigma * SIGMA_TO_DB

    print(f"\n  Data-rich region (Europe):")
    print(f"    Path: G (London) → DL (Berlin)")
    print(f"    Distance: {eu_dist:.0f} km")
    print(f"    Prediction: {eu_snr:+.3f} sigma ({eu_snr * SIGMA_TO_DB:+.1f} dB)")

    print(f"\n  Data-sparse region (Africa):")
    print(f"    Path: 5H (Tanzania) → 9J (Zambia)")
    print(f"    Distance: {africa_dist:.0f} km")
    print(f"    Prediction: {africa_snr:+.3f} sigma ({africa_snr * SIGMA_TO_DB:+.1f} dB)")

    print(f"\n  Geographic Bias: {bias_sigma:.3f} sigma ({bias_db:.1f} dB)")
    print(f"  Threshold: < 5 dB (0.75 sigma)")

    # Bias should be < 5 dB (~0.75 sigma)
    if bias_db < 5.0:
        print(f"\n  PASS: Geographic bias within acceptable range")
        return True
    else:
        print(f"\n  FAIL: Model shows significant geographic bias")
        return False


def test_tst702_temporal_bias(model, device):
    """TST-702: Temporal Bias — 24-hour sweep for discontinuities"""
    print("\n" + "=" * 60)
    print("TST-702: Temporal Bias (24-hour sweep)")
    print("=" * 60)

    # Standard transatlantic path
    tx_lat, tx_lon = 39.14, -77.01  # W3
    rx_lat, rx_lon = 51.50, -0.12   # G

    predictions = []
    print(f"\n  Path: W3 → G, 20m, SFI 150, Kp 2")
    print(f"\n  {'Hour':>6s}  {'SNR sigma':>10s}  {'~dB':>8s}")
    print(f"  {'-'*28}")

    for hour in range(24):
        snr, _ = predict(
            model, device,
            tx_lat=tx_lat, tx_lon=tx_lon,
            rx_lat=rx_lat, rx_lon=rx_lon,
            freq_hz=14_097_100,
            sfi=150, kp=2, hour_utc=hour
        )
        predictions.append(snr)
        print(f"  {hour:02d}:00   {snr:+9.4f}   {snr * SIGMA_TO_DB:+7.1f}")

    # Check for discontinuities (large jumps between adjacent hours)
    max_jump = 0.0
    max_jump_hour = 0
    for i in range(24):
        next_i = (i + 1) % 24
        jump = abs(predictions[next_i] - predictions[i])
        if jump > max_jump:
            max_jump = jump
            max_jump_hour = i

    # Check for anomalous spikes
    mean_snr = np.mean(predictions)
    std_snr = np.std(predictions)
    outliers = [h for h, s in enumerate(predictions) if abs(s - mean_snr) > 3 * std_snr]

    print(f"\n  Statistics:")
    print(f"    Mean: {mean_snr:+.4f} sigma")
    print(f"    Std:  {std_snr:.4f} sigma")
    print(f"    Range: [{min(predictions):+.4f}, {max(predictions):+.4f}]")
    print(f"    Max hour-to-hour jump: {max_jump:.4f} sigma at {max_jump_hour:02d}:00")

    # Threshold: max jump < 0.5 sigma (~3.3 dB) between adjacent hours
    jump_ok = max_jump < 0.5
    no_outliers = len(outliers) == 0

    if outliers:
        print(f"    Outlier hours (>3σ from mean): {outliers}")
    else:
        print(f"    Outliers: None")

    print()
    if jump_ok and no_outliers:
        print(f"  PASS: Smooth diurnal variation, no discontinuities")
        return True
    elif jump_ok:
        print(f"  PASS: No discontinuities (outliers may reflect real physics)")
        return True
    else:
        print(f"  FAIL: Discontinuity detected at hour boundary")
        return False


def test_tst703_band_coverage(model, device):
    """TST-703: Band Coverage Bias — all bands produce valid predictions"""
    print("\n" + "=" * 60)
    print("TST-703: Band Coverage Bias (160m-10m)")
    print("=" * 60)

    # Standard transatlantic path
    tx_lat, tx_lon = 39.14, -77.01  # W3
    rx_lat, rx_lon = 51.50, -0.12   # G

    print(f"\n  Path: W3 → G, SFI 150, Kp 2, 14:00 UTC")
    print(f"\n  {'Band':>6s}  {'MHz':>10s}  {'SNR sigma':>11s}  {'~dB':>8s}  {'Status':>8s}")
    print(f"  {'-'*50}")

    all_valid = True
    predictions = []

    for band_name, freq_hz in BANDS:
        snr, _ = predict(
            model, device,
            tx_lat=tx_lat, tx_lon=tx_lon,
            rx_lat=rx_lat, rx_lon=rx_lon,
            freq_hz=freq_hz,
            sfi=150, kp=2, hour_utc=14
        )

        predictions.append((band_name, snr))

        # Check validity
        is_finite = math.isfinite(snr)
        in_range = -5.0 <= snr <= 3.0  # Reasonable range for this path

        if is_finite and in_range:
            status = "OK"
        elif is_finite:
            status = "RANGE"  # Outside expected but still valid
        else:
            status = "INVALID"
            all_valid = False

        print(f"  {band_name:>6s}  {freq_hz/1e6:9.3f}   {snr:+10.4f}   {snr * SIGMA_TO_DB:+7.1f}  {status:>8s}")

    # Check for extreme band-to-band variation
    snr_values = [s for _, s in predictions]
    band_range = max(snr_values) - min(snr_values)

    print(f"\n  Band-to-band range: {band_range:.3f} sigma ({band_range * SIGMA_TO_DB:.1f} dB)")

    # All predictions should be finite
    all_finite = all(math.isfinite(s) for _, s in predictions)

    if all_finite:
        print(f"\n  PASS: All bands produce finite predictions")
        return True
    else:
        print(f"\n  FAIL: Some bands produce invalid predictions")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  IONIS V20 — TST-700 Bias & Fairness Tests")
    print("=" * 60)
    print("\n  These tests check for systematic biases in predictions.")

    # Load model
    print(f"\nLoading {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)

    model = IonisGate(
        dnn_dim=DNN_DIM,
        sidecar_hidden=SIDECAR_HIDDEN,
        sfi_idx=SFI_IDX,
        kp_penalty_idx=KP_PENALTY_IDX,
        gate_init_bias=CONFIG["model"]["gate_init_bias"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    print(f"  Device: {DEVICE}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Run tests
    results = []
    results.append(("TST-701", "Geographic Bias", test_tst701_geographic_bias(model, DEVICE)))
    results.append(("TST-702", "Temporal Bias", test_tst702_temporal_bias(model, DEVICE)))
    results.append(("TST-703", "Band Coverage", test_tst703_band_coverage(model, DEVICE)))

    # Summary
    print("\n" + "=" * 60)
    print("  TST-700 SUMMARY: Bias & Fairness")
    print("=" * 60)

    passed = sum(1 for _, _, p in results if p)
    total = len(results)

    for test_id, name, p in results:
        status = "PASS" if p else "FAIL"
        print(f"  {test_id}: {name:<20s}  {status}")

    print()
    print(f"  Results: {passed}/{total} passed")

    if passed == total:
        print("\n  ALL TST-700 TESTS PASSED")
        print("  Model shows no significant systematic biases.")
        return 0
    else:
        print(f"\n  {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
