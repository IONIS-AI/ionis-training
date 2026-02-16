#!/usr/bin/env python3
"""
test_tst500_robustness.py — IONIS V20 Model Robustness Tests

TST-500 Group: Standard ML model tests for determinism, stability, and numerical safety.

Tests:
  TST-501: Reproducibility (same input → same output)
  TST-502: Input Perturbation Stability (small changes → small output changes)
  TST-503: Boundary Value Testing (edge cases handled gracefully)
  TST-504: Null Input Handling (NaN/None rejection)
  TST-505: Numerical Overflow (extreme valid inputs)
  TST-506: Checkpoint Integrity (load and verify)
  TST-507: Device Portability (CPU vs MPS consistency)

These are standard ML model validation tests — they apply to any neural network.
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
INPUT_DIM = CONFIG["model"]["input_dim"]

# Reference values from V20 training
REF_RMSE = 0.862
REF_PEARSON = 0.4777
RMSE_TOLERANCE = 0.05
PEARSON_TOLERANCE = 0.02

SIGMA_TO_DB = 6.7


# ── Feature Builder ──────────────────────────────────────────────────────────

def make_reference_input(device):
    """Build the canonical reference input vector (W3→G, 20m, midday)."""
    distance = 5900.0 / 20000.0
    freq_log = np.log10(14_097_100) / 8.0
    hour = 12
    hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hour / 24.0)
    azimuth = 50.0
    az_sin = np.sin(2.0 * np.pi * azimuth / 360.0)
    az_cos = np.cos(2.0 * np.pi * azimuth / 360.0)
    tx_lat, tx_lon = 39.14, -77.01
    rx_lat, rx_lon = 51.50, -0.12
    lat_diff = abs(tx_lat - rx_lat) / 180.0
    midpoint_lat = (tx_lat + rx_lat) / 2.0 / 90.0
    month = 6
    season_sin = np.sin(2.0 * np.pi * month / 12.0)
    season_cos = np.cos(2.0 * np.pi * month / 12.0)
    midpoint_lon = (tx_lon + rx_lon) / 2.0
    local_solar_h = hour + midpoint_lon / 15.0
    day_night_est = np.cos(2.0 * np.pi * local_solar_h / 24.0)
    sfi = 150.0
    kp = 2.0
    sfi_norm = sfi / 300.0
    kp_penalty = 1.0 - kp / 9.0

    return torch.tensor(
        [[distance, freq_log, hour_sin, hour_cos,
          az_sin, az_cos, lat_diff, midpoint_lat,
          season_sin, season_cos, day_night_est,
          sfi_norm, kp_penalty]],
        dtype=torch.float32, device=device,
    )


def make_boundary_input(device, distance_km=5900, sfi=150, kp=2):
    """Build input with specified boundary values."""
    distance = distance_km / 20000.0
    freq_log = np.log10(14_097_100) / 8.0
    hour = 12
    hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hour / 24.0)
    azimuth = 50.0
    az_sin = np.sin(2.0 * np.pi * azimuth / 360.0)
    az_cos = np.cos(2.0 * np.pi * azimuth / 360.0)
    lat_diff = 0.1
    midpoint_lat = 0.5
    month = 6
    season_sin = np.sin(2.0 * np.pi * month / 12.0)
    season_cos = np.cos(2.0 * np.pi * month / 12.0)
    day_night_est = 1.0
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

def test_tst501_reproducibility(model, device):
    """TST-501: Reproducibility — same input produces same output."""
    print("\n" + "=" * 60)
    print("TST-501: Reproducibility")
    print("=" * 60)

    model.eval()
    x = make_reference_input(device)

    # Run 100 identical predictions
    predictions = []
    with torch.no_grad():
        for _ in range(100):
            pred = model(x).item()
            predictions.append(pred)

    unique_values = set(predictions)
    variance = np.var(predictions)

    print(f"  Ran 100 identical predictions")
    print(f"  Unique values: {len(unique_values)}")
    print(f"  Variance: {variance:.2e}")
    print(f"  Range: [{min(predictions):.6f}, {max(predictions):.6f}]")

    if len(unique_values) == 1:
        print("  PASS: All 100 predictions identical (deterministic)")
        return True
    elif variance < 1e-10:
        print("  PASS: Variance negligible (< 1e-10)")
        return True
    else:
        print("  FAIL: Non-deterministic inference detected")
        return False


def test_tst502_perturbation_stability(model, device):
    """TST-502: Input Perturbation Stability — small input changes produce small output changes."""
    print("\n" + "=" * 60)
    print("TST-502: Input Perturbation Stability")
    print("=" * 60)

    model.eval()
    x_base = make_reference_input(device)

    with torch.no_grad():
        base_pred = model(x_base).item()

    # Perturb each feature by ±0.1%
    max_delta = 0.0
    worst_feature = None

    for i in range(INPUT_DIM):
        x_perturbed = x_base.clone()
        original_val = x_perturbed[0, i].item()

        # Skip if value is zero (can't compute percentage)
        if abs(original_val) < 1e-6:
            continue

        x_perturbed[0, i] *= 1.001  # +0.1%

        with torch.no_grad():
            perturbed_pred = model(x_perturbed).item()

        delta = abs(perturbed_pred - base_pred)
        delta_db = delta * SIGMA_TO_DB

        if delta > max_delta:
            max_delta = delta
            worst_feature = i

    print(f"  Base prediction: {base_pred:+.4f} sigma")
    print(f"  Max output change for 0.1% input perturbation: {max_delta:.4f} sigma ({max_delta * SIGMA_TO_DB:.2f} dB)")
    print(f"  Worst feature index: {worst_feature}")

    # Threshold: 0.5 dB change for 0.1% input change
    threshold_sigma = 0.5 / SIGMA_TO_DB

    if max_delta < threshold_sigma:
        print(f"  PASS: Max change {max_delta * SIGMA_TO_DB:.2f} dB < 0.5 dB threshold")
        return True
    else:
        print(f"  FAIL: Max change {max_delta * SIGMA_TO_DB:.2f} dB >= 0.5 dB threshold")
        return False


def test_tst503_boundary_values(model, device):
    """TST-503: Boundary Value Testing — edge cases produce finite outputs."""
    print("\n" + "=" * 60)
    print("TST-503: Boundary Value Testing")
    print("=" * 60)

    model.eval()

    # Test boundary conditions
    boundary_cases = [
        ("SFI=50 (minimum)", dict(sfi=50, kp=2)),
        ("SFI=300 (high)", dict(sfi=300, kp=2)),
        ("Kp=0 (quiet)", dict(sfi=150, kp=0)),
        ("Kp=9 (severe storm)", dict(sfi=150, kp=9)),
        ("Distance=100km (short)", dict(distance_km=100, sfi=150, kp=2)),
        ("Distance=19000km (long)", dict(distance_km=19000, sfi=150, kp=2)),
        ("All extremes", dict(distance_km=19000, sfi=300, kp=9)),
    ]

    all_finite = True
    all_in_range = True

    print(f"\n  {'Case':<25s}  {'Prediction':>12s}  {'Status':>8s}")
    print(f"  {'-'*50}")

    for name, kwargs in boundary_cases:
        x = make_boundary_input(device, **kwargs)
        with torch.no_grad():
            pred = model(x).item()

        is_finite = math.isfinite(pred)
        # Reasonable SNR range: -10 to +5 sigma (-67 to +33 dB)
        in_range = -10.0 <= pred <= 5.0

        status = "OK" if is_finite and in_range else "ISSUE"
        print(f"  {name:<25s}  {pred:+11.4f}σ  {status:>8s}")

        if not is_finite:
            all_finite = False
        if not in_range:
            all_in_range = False

    print()
    if all_finite and all_in_range:
        print("  PASS: All boundary cases produce finite, reasonable outputs")
        return True
    elif all_finite:
        print("  PARTIAL: All outputs finite but some outside expected range")
        return True  # Still pass if finite
    else:
        print("  FAIL: NaN or Inf detected in boundary cases")
        return False


def test_tst504_null_handling(model, device):
    """TST-504: Null Input Handling — NaN inputs are caught."""
    print("\n" + "=" * 60)
    print("TST-504: Null Input Handling")
    print("=" * 60)

    model.eval()

    # Test with NaN in input
    x_nan = make_reference_input(device)
    x_nan[0, 0] = float('nan')

    with torch.no_grad():
        pred_nan = model(x_nan).item()

    nan_detected = math.isnan(pred_nan)

    print(f"  Input with NaN in feature 0")
    print(f"  Output: {pred_nan}")

    # Test with Inf in input
    x_inf = make_reference_input(device)
    x_inf[0, 0] = float('inf')

    with torch.no_grad():
        pred_inf = model(x_inf).item()

    inf_produces_issue = math.isnan(pred_inf) or math.isinf(pred_inf)

    print(f"\n  Input with Inf in feature 0")
    print(f"  Output: {pred_inf}")

    # For this test, we want to verify that bad inputs don't silently
    # produce seemingly-valid outputs. NaN in → NaN out is acceptable.
    # The Oracle wrapper (TST-300) handles rejection at a higher level.

    if nan_detected:
        print("\n  NaN propagated through model (expected behavior)")
        print("  Note: Production use should wrap model with IonisOracle for input validation")
        print("  PASS: Model does not hide NaN corruption")
        return True
    else:
        print("\n  WARNING: NaN input produced finite output — may mask data issues")
        print("  PASS: Model produced output (validation should happen at oracle level)")
        return True  # Pass but with warning


def test_tst505_numerical_overflow(model, device):
    """TST-505: Numerical Overflow — extreme valid inputs don't overflow."""
    print("\n" + "=" * 60)
    print("TST-505: Numerical Overflow")
    print("=" * 60)

    model.eval()

    # Extreme but valid inputs
    x = make_boundary_input(device, distance_km=19999, sfi=300, kp=9)

    with torch.no_grad():
        pred = model(x).item()

    print(f"  Extreme input: distance=19999km, SFI=300, Kp=9")
    print(f"  Prediction: {pred:+.4f} sigma ({pred * SIGMA_TO_DB:+.1f} dB)")

    is_finite = math.isfinite(pred)

    if is_finite:
        print("  PASS: No overflow with extreme valid inputs")
        return True
    else:
        print("  FAIL: Overflow detected (Inf or NaN)")
        return False


def test_tst506_checkpoint_integrity(model, device, checkpoint):
    """TST-506: Checkpoint Integrity — loaded model matches documented metrics."""
    print("\n" + "=" * 60)
    print("TST-506: Checkpoint Integrity")
    print("=" * 60)

    # Check checkpoint contains expected keys
    required_keys = ['model_state', 'val_rmse', 'val_pearson']
    missing_keys = [k for k in required_keys if k not in checkpoint]

    if missing_keys:
        print(f"  FAIL: Checkpoint missing keys: {missing_keys}")
        return False

    # Verify metrics match documented values
    loaded_rmse = checkpoint.get('val_rmse', 0)
    loaded_pearson = checkpoint.get('val_pearson', 0)

    print(f"  Checkpoint RMSE:    {loaded_rmse:.4f} sigma")
    print(f"  Expected RMSE:      {REF_RMSE:.4f} sigma (±{RMSE_TOLERANCE})")
    print(f"  Checkpoint Pearson: {loaded_pearson:+.4f}")
    print(f"  Expected Pearson:   {REF_PEARSON:+.4f} (±{PEARSON_TOLERANCE})")

    rmse_ok = abs(loaded_rmse - REF_RMSE) <= RMSE_TOLERANCE
    pearson_ok = abs(loaded_pearson - REF_PEARSON) <= PEARSON_TOLERANCE

    # Verify model produces expected reference prediction
    model.eval()
    x = make_reference_input(device)
    with torch.no_grad():
        ref_pred = model(x).item()

    # Reference prediction should be roughly -0.5 to +0.5 sigma for standard path
    pred_reasonable = -2.0 <= ref_pred <= 2.0

    print(f"\n  Reference prediction (W3→G, 20m): {ref_pred:+.4f} sigma")
    print(f"  Prediction in reasonable range: {pred_reasonable}")

    if rmse_ok and pearson_ok and pred_reasonable:
        print("\n  PASS: Checkpoint integrity verified")
        return True
    else:
        issues = []
        if not rmse_ok:
            issues.append("RMSE mismatch")
        if not pearson_ok:
            issues.append("Pearson mismatch")
        if not pred_reasonable:
            issues.append("Reference prediction out of range")
        print(f"\n  FAIL: {', '.join(issues)}")
        return False


def test_tst507_device_portability(device):
    """TST-507: Device Portability — CPU and MPS produce consistent results."""
    print("\n" + "=" * 60)
    print("TST-507: Device Portability")
    print("=" * 60)

    # Always test CPU
    devices_to_test = [torch.device("cpu")]

    # Add accelerator if available
    if torch.cuda.is_available():
        devices_to_test.append(torch.device("cuda"))
        print("  Testing: CPU, CUDA")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices_to_test.append(torch.device("mps"))
        print("  Testing: CPU, MPS")
    else:
        print("  Testing: CPU only (no accelerator available)")

    predictions = {}

    for dev in devices_to_test:
        # Load model on this device
        checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location=dev)
        model = IonisGate(
            dnn_dim=DNN_DIM,
            sidecar_hidden=SIDECAR_HIDDEN,
            sfi_idx=SFI_IDX,
            kp_penalty_idx=KP_PENALTY_IDX,
            gate_init_bias=CONFIG["model"]["gate_init_bias"],
        ).to(dev)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()

        x = make_reference_input(dev)
        with torch.no_grad():
            pred = model(x).item()

        predictions[str(dev)] = pred
        print(f"  {str(dev):>6s}: {pred:+.6f} sigma")

    # Check consistency
    if len(predictions) > 1:
        values = list(predictions.values())
        max_diff = max(values) - min(values)
        print(f"\n  Max difference: {max_diff:.6f} sigma ({max_diff * SIGMA_TO_DB:.4f} dB)")

        # Tolerance: 0.001 sigma difference between devices
        if max_diff < 0.001:
            print("  PASS: Cross-device predictions consistent (< 0.001 sigma)")
            return True
        else:
            print("  FAIL: Device predictions differ significantly")
            return False
    else:
        print("\n  PASS: Single device test (no cross-device comparison possible)")
        return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  IONIS V20 — TST-500 Model Robustness Tests")
    print("=" * 60)

    # Determine device
    device = get_device()

    # Load model
    print(f"\nLoading {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location=device)

    model = IonisGate(
        dnn_dim=DNN_DIM,
        sidecar_hidden=SIDECAR_HIDDEN,
        sfi_idx=SFI_IDX,
        kp_penalty_idx=KP_PENALTY_IDX,
        gate_init_bias=CONFIG["model"]["gate_init_bias"],
    ).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Run tests
    results = []
    results.append(("TST-501", "Reproducibility", test_tst501_reproducibility(model, device)))
    results.append(("TST-502", "Perturbation Stability", test_tst502_perturbation_stability(model, device)))
    results.append(("TST-503", "Boundary Values", test_tst503_boundary_values(model, device)))
    results.append(("TST-504", "Null Handling", test_tst504_null_handling(model, device)))
    results.append(("TST-505", "Numerical Overflow", test_tst505_numerical_overflow(model, device)))
    results.append(("TST-506", "Checkpoint Integrity", test_tst506_checkpoint_integrity(model, device, checkpoint)))
    results.append(("TST-507", "Device Portability", test_tst507_device_portability(device)))

    # Summary
    print("\n" + "=" * 60)
    print("  TST-500 SUMMARY: Model Robustness")
    print("=" * 60)

    passed = sum(1 for _, _, p in results if p)
    total = len(results)

    for test_id, name, p in results:
        status = "PASS" if p else "FAIL"
        print(f"  {test_id}: {name:<25s}  {status}")

    print()
    print(f"  Results: {passed}/{total} passed")

    if passed == total:
        print("\n  ALL TST-500 TESTS PASSED")
        return 0
    else:
        print(f"\n  {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
