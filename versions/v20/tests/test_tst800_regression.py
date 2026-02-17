#!/usr/bin/env python3
"""
test_tst800_regression.py — IONIS V20 Regression Tests

TST-800 Group: Baseline tests to catch silent model changes or degradation.

Tests:
  TST-801: Reference Prediction (fixed input → documented output)
  TST-802: RMSE Regression (checkpoint matches documented value)
  TST-803: Pearson Regression (checkpoint matches documented value)

These tests lock in IONIS V20 baselines. Any future model changes
should cause these tests to fail, forcing an explicit version bump.

V20 BASELINE VALUES (locked 2026-02-16):
  - Reference prediction (W3→G, 20m, SFI 150, Kp 2, 12 UTC): -0.328 sigma
  - RMSE: 0.8617 sigma
  - Pearson: +0.4879
"""

import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors

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

DEVICE = get_device()

# ── IONIS V20 Baselines ──────────────────────────────────────────────
# These values are locked for V20. Do NOT change unless intentionally
# releasing a new model version.

# Reference prediction for canonical path (W3→G, 20m, SFI 150, Kp 2, 12 UTC)
BASELINE_REF_PREDICTION_SIGMA = -0.328
BASELINE_REF_TOLERANCE_SIGMA = 0.05  # ±0.05 sigma (~0.3 dB)

# Training metrics from checkpoint (val_rmse, val_pearson in ionis_v20.pth)
BASELINE_RMSE = 0.8617
BASELINE_RMSE_TOLERANCE = 0.01

BASELINE_PEARSON = 0.4879
BASELINE_PEARSON_TOLERANCE = 0.005

# Conversion factor
SIGMA_TO_DB = 6.7


# ── Reference Input Builder ──────────────────────────────────────────────────

def make_reference_input(device):
    """
    Build the canonical V20 reference input vector.

    Path: W3 (Maryland) → G (London)
    Band: 20m (14.097 MHz)
    Conditions: SFI 150, Kp 2, 12:00 UTC, June
    Distance: ~5,900 km

    This exact input is used for regression testing.
    """
    # Path parameters
    tx_lat, tx_lon = 39.14, -77.01  # W3 area (Maryland)
    rx_lat, rx_lon = 51.50, -0.12   # G area (London)
    distance_km = 5900.0
    azimuth = 50.0
    freq_hz = 14_097_100  # 20m WSPR

    # Conditions
    sfi = 150.0
    kp = 2.0
    hour = 12
    month = 6

    # Build normalized features
    distance = distance_km / 20000.0
    freq_log = np.log10(freq_hz) / 8.0
    hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hour / 24.0)
    az_sin = np.sin(2.0 * np.pi * azimuth / 360.0)
    az_cos = np.cos(2.0 * np.pi * azimuth / 360.0)
    lat_diff = abs(tx_lat - rx_lat) / 180.0
    midpoint_lat = (tx_lat + rx_lat) / 2.0 / 90.0
    season_sin = np.sin(2.0 * np.pi * month / 12.0)
    season_cos = np.cos(2.0 * np.pi * month / 12.0)
    midpoint_lon = (tx_lon + rx_lon) / 2.0
    local_solar_h = hour + midpoint_lon / 15.0
    day_night_est = np.cos(2.0 * np.pi * local_solar_h / 24.0)
    sfi_norm = sfi / 300.0
    kp_penalty = 1.0 - kp / 9.0

    return torch.tensor(
        [[distance, freq_log, hour_sin, hour_cos,
          az_sin, az_cos, lat_diff, midpoint_lat,
          season_sin, season_cos, day_night_est,
          sfi_norm, kp_penalty]],
        dtype=torch.float32, device=device,
    )


# ── Test Functions ───────────────────────────────────────────────────────────

def test_tst801_reference_prediction(model, device):
    """TST-801: Reference Prediction — canonical path produces expected output."""
    print("\n" + "=" * 60)
    print("TST-801: Reference Prediction")
    print("=" * 60)

    model.eval()
    x = make_reference_input(device)

    with torch.no_grad():
        prediction = model(x).item()

    delta = abs(prediction - BASELINE_REF_PREDICTION_SIGMA)

    print(f"\n  Reference Path: W3 → G (Maryland to London)")
    print(f"  Band: 20m (14.097 MHz)")
    print(f"  Conditions: SFI 150, Kp 2, 12:00 UTC, June")
    print()
    print(f"  Prediction:      {prediction:+.4f} sigma ({prediction * SIGMA_TO_DB:+.1f} dB)")
    print(f"  Baseline:        {BASELINE_REF_PREDICTION_SIGMA:+.4f} sigma ({BASELINE_REF_PREDICTION_SIGMA * SIGMA_TO_DB:+.1f} dB)")
    print(f"  Tolerance:       ±{BASELINE_REF_TOLERANCE_SIGMA:.3f} sigma (±{BASELINE_REF_TOLERANCE_SIGMA * SIGMA_TO_DB:.1f} dB)")
    print(f"  Delta:           {delta:.4f} sigma")

    if delta <= BASELINE_REF_TOLERANCE_SIGMA:
        print(f"\n  PASS: Reference prediction within tolerance")
        return True
    else:
        print(f"\n  FAIL: Reference prediction outside tolerance")
        print(f"        Model weights may have changed without version bump")
        return False


def test_tst802_rmse_regression(metadata):
    """TST-802: RMSE Regression — metadata RMSE matches documented value."""
    print("\n" + "=" * 60)
    print("TST-802: RMSE Regression")
    print("=" * 60)

    loaded_rmse = metadata.get('val_rmse', None)

    if loaded_rmse is None:
        print("  FAIL: Metadata missing 'val_rmse' key")
        return False

    delta = abs(loaded_rmse - BASELINE_RMSE)

    print(f"\n  Checkpoint RMSE: {loaded_rmse:.4f} sigma")
    print(f"  Baseline RMSE:   {BASELINE_RMSE:.4f} sigma")
    print(f"  Tolerance:       ±{BASELINE_RMSE_TOLERANCE:.4f} sigma")
    print(f"  Delta:           {delta:.4f} sigma")

    if delta <= BASELINE_RMSE_TOLERANCE:
        print(f"\n  PASS: RMSE matches baseline")
        return True
    else:
        print(f"\n  FAIL: RMSE changed from baseline")
        print(f"        Checkpoint may have been overwritten or corrupted")
        return False


def test_tst803_pearson_regression(metadata):
    """TST-803: Pearson Regression — metadata Pearson matches documented value."""
    print("\n" + "=" * 60)
    print("TST-803: Pearson Regression")
    print("=" * 60)

    loaded_pearson = metadata.get('val_pearson', None)

    if loaded_pearson is None:
        print("  FAIL: Metadata missing 'val_pearson' key")
        return False

    delta = abs(loaded_pearson - BASELINE_PEARSON)

    print(f"\n  Checkpoint Pearson: {loaded_pearson:+.4f}")
    print(f"  Baseline Pearson:   {BASELINE_PEARSON:+.4f}")
    print(f"  Tolerance:          ±{BASELINE_PEARSON_TOLERANCE:.4f}")
    print(f"  Delta:              {delta:.4f}")

    if delta <= BASELINE_PEARSON_TOLERANCE:
        print(f"\n  PASS: Pearson matches baseline")
        return True
    else:
        print(f"\n  FAIL: Pearson changed from baseline")
        print(f"        Checkpoint may have been overwritten or corrupted")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  IONIS V20 — TST-800 Regression Tests")
    print("=" * 60)
    print("\n  These tests verify IONIS V20 baselines are unchanged.")
    print("  Failure indicates the model checkpoint was modified.")

    # Load model (safetensors — no pickle)
    print(f"\nLoading {MODEL_PATH}...")
    state_dict = load_safetensors(MODEL_PATH, device=str(DEVICE))

    # Load metadata from companion JSON
    meta_path = MODEL_PATH.replace(".safetensors", "_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    model = IonisGate(
        dnn_dim=DNN_DIM,
        sidecar_hidden=SIDECAR_HIDDEN,
        sfi_idx=SFI_IDX,
        kp_penalty_idx=KP_PENALTY_IDX,
        gate_init_bias=CONFIG["model"]["gate_init_bias"],
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"  Device: {DEVICE}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Run tests
    results = []
    results.append(("TST-801", "Reference Prediction", test_tst801_reference_prediction(model, DEVICE)))
    results.append(("TST-802", "RMSE Regression", test_tst802_rmse_regression(metadata)))
    results.append(("TST-803", "Pearson Regression", test_tst803_pearson_regression(metadata)))

    # Summary
    print("\n" + "=" * 60)
    print("  TST-800 SUMMARY: Regression Tests")
    print("=" * 60)

    passed = sum(1 for _, _, p in results if p)
    total = len(results)

    for test_id, name, p in results:
        status = "PASS" if p else "FAIL"
        print(f"  {test_id}: {name:<25s}  {status}")

    print()
    print(f"  V20 Baselines (locked 2026-02-16):")
    print(f"    Reference prediction: {BASELINE_REF_PREDICTION_SIGMA:+.3f} sigma")
    print(f"    RMSE:                 {BASELINE_RMSE:.3f} sigma")
    print(f"    Pearson:              {BASELINE_PEARSON:+.4f}")

    print()
    print(f"  Results: {passed}/{total} passed")

    if passed == total:
        print("\n  ALL TST-800 TESTS PASSED")
        print("  IONIS V20 checkpoint verified intact.")
        return 0
    else:
        print(f"\n  {total - passed} TEST(S) FAILED")
        print("  WARNING: Model checkpoint may have been modified!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
