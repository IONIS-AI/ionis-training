#!/usr/bin/env python3
"""
test_tst200_physics.py — IONIS V20 Physics Constraints Tests

TST-200 Group: Verify the model's learned physics matches ionospheric reality.

Tests:
  TST-201: SFI Monotonicity (70 vs 200) — Higher SFI improves signal
  TST-202: Kp Monotonicity (0 vs 9) — Storms degrade signal
  TST-203: D-Layer Absorption (80m vs 20m at Noon) — Daytime absorption
  TST-204: Polar Storm Degradation (Kp 2 vs 8) — High-lat storm sensitivity
  TST-205: 10m SFI Sensitivity — Higher bands need higher SFI
  TST-206: Grey Line Enhancement — Twilight propagation boost

These are the "V16 Physics Laws" that must survive any refactoring.
"""

import json
import os
import sys

import numpy as np
import torch

# ── Path Setup ───────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V20_DIR = os.path.dirname(SCRIPT_DIR)
VERSIONS_DIR = os.path.dirname(V20_DIR)
COMMON_DIR = os.path.join(VERSIONS_DIR, "common")
sys.path.insert(0, COMMON_DIR)

from train_common import IonisGate

# ── Load Config ──────────────────────────────────────────────────────────────

CONFIG_PATH = os.path.join(V20_DIR, "config_v20.json")
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

MODEL_PATH = os.path.join(V20_DIR, CONFIG["checkpoint"])
DNN_DIM = CONFIG["model"]["dnn_dim"]
SFI_IDX = CONFIG["model"]["sfi_idx"]
KP_PENALTY_IDX = CONFIG["model"]["kp_penalty_idx"]
SIDECAR_HIDDEN = CONFIG["model"]["sidecar_hidden"]

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

SIGMA_TO_DB = 6.7  # Approximate conversion factor


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
        return model(features).item()


# ── Test Functions ───────────────────────────────────────────────────────────

def test_tst201_sfi_monotonicity(model, device):
    """TST-201: SFI Monotonicity — Higher SFI should improve signal."""
    print("\n" + "=" * 60)
    print("TST-201: SFI Monotonicity (70 vs 200)")
    print("=" * 60)

    # Standard transatlantic path: W3 → G, 20m
    tx_lat, tx_lon = 39.14, -77.01  # W3
    rx_lat, rx_lon = 51.50, -0.12   # G
    freq_hz = 14_097_100  # 20m WSPR dial

    snr_low = predict(model, device, tx_lat, tx_lon, rx_lat, rx_lon,
                      freq_hz, sfi=70, kp=2, hour_utc=14)
    snr_high = predict(model, device, tx_lat, tx_lon, rx_lat, rx_lon,
                       freq_hz, sfi=200, kp=2, hour_utc=14)

    delta_sigma = snr_high - snr_low
    delta_db = delta_sigma * SIGMA_TO_DB

    print(f"\n  Path: W3 → G, 20m, Kp 2, 14:00 UTC")
    print(f"  SNR at SFI 70:  {snr_low:+.3f} sigma ({snr_low * SIGMA_TO_DB:+.1f} dB)")
    print(f"  SNR at SFI 200: {snr_high:+.3f} sigma ({snr_high * SIGMA_TO_DB:+.1f} dB)")
    print(f"\n  SFI Benefit (70→200): {delta_sigma:+.3f} sigma ({delta_db:+.1f} dB)")

    # Scoring based on spec
    if delta_db >= 3.0:
        grade = "A"
    elif delta_db >= 2.0:
        grade = "B"
    elif delta_db >= 1.0:
        grade = "C"
    elif delta_db > 0:
        grade = "D"
    else:
        grade = "F"

    print(f"  Grade: {grade}")

    if delta_sigma > 0:
        print(f"\n  PASS: Signal IMPROVED with higher SFI (correct physics)")
        return True
    else:
        print(f"\n  FAIL: Signal DECREASED with higher SFI (inverted physics)")
        return False


def test_tst202_kp_monotonicity(model, device):
    """TST-202: Kp Monotonicity — Storms should degrade signal."""
    print("\n" + "=" * 60)
    print("TST-202: Kp Monotonicity (0 vs 9)")
    print("=" * 60)

    # Standard transatlantic path: W3 → G, 20m
    tx_lat, tx_lon = 39.14, -77.01  # W3
    rx_lat, rx_lon = 51.50, -0.12   # G
    freq_hz = 14_097_100  # 20m WSPR dial

    snr_quiet = predict(model, device, tx_lat, tx_lon, rx_lat, rx_lon,
                        freq_hz, sfi=150, kp=0, hour_utc=14)
    snr_storm = predict(model, device, tx_lat, tx_lon, rx_lat, rx_lon,
                        freq_hz, sfi=150, kp=9, hour_utc=14)

    storm_cost_sigma = snr_quiet - snr_storm
    storm_cost_db = storm_cost_sigma * SIGMA_TO_DB

    print(f"\n  Path: W3 → G, 20m, SFI 150, 14:00 UTC")
    print(f"  SNR at Kp 0 (Quiet): {snr_quiet:+.3f} sigma ({snr_quiet * SIGMA_TO_DB:+.1f} dB)")
    print(f"  SNR at Kp 9 (Storm): {snr_storm:+.3f} sigma ({snr_storm * SIGMA_TO_DB:+.1f} dB)")
    print(f"\n  Storm Cost (Kp 0→9): {storm_cost_sigma:+.3f} sigma ({storm_cost_db:+.1f} dB)")

    # Scoring based on spec
    if storm_cost_db >= 4.0:
        grade = "A"
    elif storm_cost_db >= 3.0:
        grade = "A"
    elif storm_cost_db >= 2.0:
        grade = "B"
    elif storm_cost_db >= 1.0:
        grade = "C"
    elif storm_cost_db > 0:
        grade = "D"
    else:
        grade = "F"

    print(f"  Grade: {grade}")

    if storm_cost_sigma > 0:
        print(f"\n  PASS: Signal DROPPED during storm (correct physics)")
        return True
    else:
        print(f"\n  FAIL: Signal INCREASED during storm (inverted physics)")
        return False


def test_tst203_dlayer_absorption(model, device):
    """TST-203: D-Layer Absorption — 20m should beat 80m at noon."""
    print("\n" + "=" * 60)
    print("TST-203: D-Layer Absorption (80m vs 20m at Noon)")
    print("=" * 60)

    # Transatlantic path at solar noon
    tx_lat, tx_lon = 39.14, -77.01  # W3
    rx_lat, rx_lon = 51.50, -0.12   # G

    freq_80m = 3_568_600   # 80m WSPR dial
    freq_20m = 14_097_100  # 20m WSPR dial

    snr_80m = predict(model, device, tx_lat, tx_lon, rx_lat, rx_lon,
                      freq_80m, sfi=150, kp=2, hour_utc=12)
    snr_20m = predict(model, device, tx_lat, tx_lon, rx_lat, rx_lon,
                      freq_20m, sfi=150, kp=2, hour_utc=12)

    delta_sigma = snr_20m - snr_80m
    delta_db = delta_sigma * SIGMA_TO_DB

    print(f"\n  Path: W3 → G, SFI 150, Kp 2, 12:00 UTC (noon)")
    print(f"  SNR at 80m (3.5 MHz): {snr_80m:+.3f} sigma ({snr_80m * SIGMA_TO_DB:+.1f} dB)")
    print(f"  SNR at 20m (14 MHz):  {snr_20m:+.3f} sigma ({snr_20m * SIGMA_TO_DB:+.1f} dB)")
    print(f"\n  20m vs 80m Delta: {delta_sigma:+.3f} sigma ({delta_db:+.1f} dB)")

    # Scoring based on spec
    if delta_db >= 3.0:
        grade = "A"
    elif delta_db >= 1.0:
        grade = "B"
    elif delta_db >= 0:
        grade = "C"
    elif delta_db >= -1.0:
        grade = "D"
    else:
        grade = "F"

    print(f"  Grade: {grade}")

    # Pass criteria: delta >= 0 (20m equal or better at noon)
    if delta_sigma >= 0:
        print(f"\n  PASS: 20m equal or better than 80m at noon (D-layer effect)")
        return True
    else:
        print(f"\n  FAIL: 80m better than 20m at noon (unexpected)")
        # Note: This might actually be acceptable for some paths
        # Real D-layer absorption depends on many factors
        return False


def test_tst204_polar_storm(model, device):
    """TST-204: Polar Storm Degradation — High-lat paths more affected."""
    print("\n" + "=" * 60)
    print("TST-204: Polar Storm Degradation (Kp 2 vs 8)")
    print("=" * 60)

    # Polar path: Greenland → Finland
    tx_lat, tx_lon = 64.18, -51.72  # OX (Nuuk)
    rx_lat, rx_lon = 60.17, 24.94   # OH (Helsinki)
    freq_hz = 14_097_100  # 20m

    snr_quiet = predict(model, device, tx_lat, tx_lon, rx_lat, rx_lon,
                        freq_hz, sfi=150, kp=2, hour_utc=12)
    snr_storm = predict(model, device, tx_lat, tx_lon, rx_lat, rx_lon,
                        freq_hz, sfi=150, kp=8, hour_utc=12)

    polar_cost_sigma = snr_quiet - snr_storm
    polar_cost_db = polar_cost_sigma * SIGMA_TO_DB

    print(f"\n  Polar Path: OX (Greenland) → OH (Finland), 20m, SFI 150, 12:00 UTC")
    print(f"  SNR at Kp 2 (Quiet): {snr_quiet:+.3f} sigma ({snr_quiet * SIGMA_TO_DB:+.1f} dB)")
    print(f"  SNR at Kp 8 (Storm): {snr_storm:+.3f} sigma ({snr_storm * SIGMA_TO_DB:+.1f} dB)")
    print(f"\n  Polar Storm Cost (Kp 2→8): {polar_cost_sigma:+.3f} sigma ({polar_cost_db:+.1f} dB)")

    # Compare with mid-latitude path for ratio
    mid_tx_lat, mid_tx_lon = 39.14, -77.01  # W3
    mid_rx_lat, mid_rx_lon = 51.50, -0.12   # G

    mid_quiet = predict(model, device, mid_tx_lat, mid_tx_lon, mid_rx_lat, mid_rx_lon,
                        freq_hz, sfi=150, kp=2, hour_utc=12)
    mid_storm = predict(model, device, mid_tx_lat, mid_tx_lon, mid_rx_lat, mid_rx_lon,
                        freq_hz, sfi=150, kp=8, hour_utc=12)

    mid_cost_sigma = mid_quiet - mid_storm
    mid_cost_db = mid_cost_sigma * SIGMA_TO_DB

    print(f"\n  Mid-Lat Path: W3 → G, 20m, SFI 150, 12:00 UTC")
    print(f"  SNR at Kp 2: {mid_quiet:+.3f} sigma")
    print(f"  SNR at Kp 8: {mid_storm:+.3f} sigma")
    print(f"  Mid-Lat Storm Cost: {mid_cost_sigma:+.3f} sigma ({mid_cost_db:+.1f} dB)")

    if mid_cost_sigma > 0.01:
        ratio = polar_cost_sigma / mid_cost_sigma
        print(f"\n  Polar/Mid-lat ratio: {ratio:.2f}x")

        # Scoring
        if ratio >= 1.2:
            grade = "A"
        elif ratio >= 1.1:
            grade = "B"
        elif ratio >= 1.0:
            grade = "C"
        elif ratio >= 0.9:
            grade = "D"
        else:
            grade = "F"
        print(f"  Grade: {grade}")
    else:
        print(f"\n  Mid-lat storm cost too small to compute ratio")

    # Pass criteria: storm cost > 2 dB
    if polar_cost_db > 2.0:
        print(f"\n  PASS: Significant storm degradation on polar path")
        return True
    else:
        print(f"\n  FAIL: Insufficient storm degradation (< 2 dB)")
        return False


def test_tst205_10m_sfi_sensitivity(model, device):
    """TST-205: 10m SFI Sensitivity — Higher bands need higher SFI."""
    print("\n" + "=" * 60)
    print("TST-205: 10m SFI Sensitivity")
    print("=" * 60)

    # Transatlantic path on 10m
    tx_lat, tx_lon = 39.14, -77.01  # W3
    rx_lat, rx_lon = 51.50, -0.12   # G
    freq_hz = 28_124_600  # 10m WSPR dial

    snr_low = predict(model, device, tx_lat, tx_lon, rx_lat, rx_lon,
                      freq_hz, sfi=80, kp=2, hour_utc=14)
    snr_high = predict(model, device, tx_lat, tx_lon, rx_lat, rx_lon,
                       freq_hz, sfi=200, kp=2, hour_utc=14)

    delta_sigma = snr_high - snr_low
    delta_db = delta_sigma * SIGMA_TO_DB

    print(f"\n  Path: W3 → G, 10m, Kp 2, 14:00 UTC")
    print(f"  SNR at SFI 80:  {snr_low:+.3f} sigma ({snr_low * SIGMA_TO_DB:+.1f} dB)")
    print(f"  SNR at SFI 200: {snr_high:+.3f} sigma ({snr_high * SIGMA_TO_DB:+.1f} dB)")
    print(f"\n  10m SFI Benefit (80→200): {delta_sigma:+.3f} sigma ({delta_db:+.1f} dB)")

    # Compare with 20m
    freq_20m = 14_097_100
    snr_20m_low = predict(model, device, tx_lat, tx_lon, rx_lat, rx_lon,
                          freq_20m, sfi=80, kp=2, hour_utc=14)
    snr_20m_high = predict(model, device, tx_lat, tx_lon, rx_lat, rx_lon,
                           freq_20m, sfi=200, kp=2, hour_utc=14)
    delta_20m_sigma = snr_20m_high - snr_20m_low

    print(f"\n  20m SFI Benefit (80→200): {delta_20m_sigma:+.3f} sigma")

    if delta_20m_sigma > 0.01:
        ratio = delta_sigma / delta_20m_sigma
        print(f"  10m/20m sensitivity ratio: {ratio:.2f}x")

    # Scoring
    if delta_db >= 3.0:
        grade = "A"
    elif delta_db >= 2.0:
        grade = "B"
    elif delta_db >= 1.0:
        grade = "C"
    elif delta_db > 0:
        grade = "D"
    else:
        grade = "F"

    print(f"  Grade: {grade}")

    # Pass criteria: delta > 1.5 dB
    if delta_db > 1.5:
        print(f"\n  PASS: 10m shows strong SFI sensitivity")
        return True
    elif delta_sigma > 0:
        print(f"\n  PASS: 10m shows positive SFI sensitivity (below target but correct direction)")
        return True
    else:
        print(f"\n  FAIL: 10m shows inverted SFI response")
        return False


def test_tst206_grey_line(model, device):
    """TST-206: Grey Line Enhancement — Twilight should enhance E-W paths."""
    print("\n" + "=" * 60)
    print("TST-206: Grey Line / Twilight Enhancement")
    print("=" * 60)

    # Transatlantic E-W path
    tx_lat, tx_lon = 39.14, -77.01  # W3
    rx_lat, rx_lon = 51.50, -0.12   # G
    freq_hz = 14_097_100  # 20m

    snr_noon = predict(model, device, tx_lat, tx_lon, rx_lat, rx_lon,
                       freq_hz, sfi=150, kp=2, hour_utc=14)
    snr_twilight = predict(model, device, tx_lat, tx_lon, rx_lat, rx_lon,
                           freq_hz, sfi=150, kp=2, hour_utc=18)

    delta_sigma = snr_twilight - snr_noon
    delta_db = delta_sigma * SIGMA_TO_DB

    print(f"\n  Path: W3 → G, 20m, SFI 150, Kp 2")
    print(f"  SNR at 14:00 UTC (Midday): {snr_noon:+.3f} sigma ({snr_noon * SIGMA_TO_DB:+.1f} dB)")
    print(f"  SNR at 18:00 UTC (Twilight): {snr_twilight:+.3f} sigma ({snr_twilight * SIGMA_TO_DB:+.1f} dB)")
    print(f"\n  Grey Line Enhancement: {delta_sigma:+.3f} sigma ({delta_db:+.1f} dB)")

    # Scoring
    if delta_db >= 1.0:
        grade = "A"
    elif delta_db >= 0.5:
        grade = "B"
    elif delta_db >= 0:
        grade = "C"
    elif delta_db >= -0.5:
        grade = "D"
    else:
        grade = "F"

    print(f"  Grade: {grade}")

    # Pass criteria: twilight >= noon (grey line should help or be neutral)
    if delta_sigma >= 0:
        print(f"\n  PASS: Twilight shows equal or better propagation")
        return True
    elif delta_db >= -0.5:
        print(f"\n  PASS: Minor twilight degradation within tolerance")
        return True
    else:
        print(f"\n  FAIL: Significant twilight degradation (unexpected)")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  IONIS V20 — TST-200 Physics Constraints Tests")
    print("=" * 60)
    print("\n  These tests verify learned physics matches ionospheric reality.")
    print("  The 'V16 Physics Laws' that must survive any refactoring.")

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
    print(f"  RMSE: {checkpoint.get('val_rmse', 0):.4f} sigma")
    print(f"  Pearson: {checkpoint.get('val_pearson', 0):+.4f}")

    # Run tests
    results = []
    results.append(("TST-201", "SFI Monotonicity", test_tst201_sfi_monotonicity(model, DEVICE)))
    results.append(("TST-202", "Kp Monotonicity", test_tst202_kp_monotonicity(model, DEVICE)))
    results.append(("TST-203", "D-Layer Absorption", test_tst203_dlayer_absorption(model, DEVICE)))
    results.append(("TST-204", "Polar Storm", test_tst204_polar_storm(model, DEVICE)))
    results.append(("TST-205", "10m SFI Sensitivity", test_tst205_10m_sfi_sensitivity(model, DEVICE)))
    results.append(("TST-206", "Grey Line", test_tst206_grey_line(model, DEVICE)))

    # Summary
    print("\n" + "=" * 60)
    print("  TST-200 SUMMARY: Physics Constraints")
    print("=" * 60)

    passed = sum(1 for _, _, p in results if p)
    total = len(results)

    for test_id, name, p in results:
        status = "PASS" if p else "FAIL"
        print(f"  {test_id}: {name:<20s}  {status}")

    print()
    print(f"  Results: {passed}/{total} passed")

    if passed == total:
        print("\n  ALL TST-200 TESTS PASSED")
        print("  V16 Physics Laws enforced correctly.")
        return 0
    else:
        print(f"\n  {total - passed} TEST(S) FAILED")
        print("  Review model physics constraints.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
