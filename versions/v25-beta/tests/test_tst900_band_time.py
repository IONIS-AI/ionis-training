#!/usr/bin/env python3
"""
test_tst900_band_time.py — IONIS V25-beta TST-900 Tests

V25-beta: Sun sidecar receives SFI * freq_log as 1D product input.
The interaction is FORCED by multiplication before entering the sidecar.
"""

import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors

# ── Path Setup ───────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V25_BETA_DIR = os.path.dirname(SCRIPT_DIR)
VERSIONS_DIR = os.path.dirname(V25_BETA_DIR)
sys.path.insert(0, os.path.join(VERSIONS_DIR, "common"))
sys.path.insert(0, V25_BETA_DIR)

from model import get_device, build_features, BAND_FREQ_HZ
from train_v25_beta import IonisGateV25Beta

# ── Load Config ──────────────────────────────────────────────────────────────

CONFIG_PATH = os.path.join(V25_BETA_DIR, "config_v25_beta.json")
CKPT_PATH = os.path.join(V25_BETA_DIR, "ionis_v25_beta.safetensors")

with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

MODEL_VERSION = f"v25-{CONFIG.get('variant', 'beta')}"

DNN_DIM = CONFIG["model"]["dnn_dim"]
SFI_IDX = CONFIG["model"]["sfi_idx"]
KP_PENALTY_IDX = CONFIG["model"]["kp_penalty_idx"]
SIDECAR_HIDDEN = CONFIG["model"]["sidecar_hidden"]
SUN_SIDECAR_INPUT_DIM = CONFIG["model"]["sun_sidecar_input_dim"]

DEVICE = get_device()
SIGMA_TO_DB = 6.7


# ── Prediction Helper ────────────────────────────────────────────────────────

def predict(model, device, tx_lat, tx_lon, rx_lat, rx_lon, freq_hz,
            sfi, kp, hour_utc, month=6, day_of_year=172):
    features = build_features(
        tx_lat, tx_lon, rx_lat, rx_lon, freq_hz,
        sfi, kp, hour_utc, month,
        day_of_year=day_of_year,
        include_solar_depression=True,
    )
    tensor = torch.tensor([features], dtype=torch.float32, device=device)
    with torch.no_grad():
        return model(tensor).item()


def sigma_to_db(sigma, band_id=107, source="wspr"):
    std = CONFIG["norm_constants_per_band"].get(str(band_id), {}).get(source, {}).get("std", 6.7)
    mean = CONFIG["norm_constants_per_band"].get(str(band_id), {}).get(source, {}).get("mean", -18.0)
    return sigma * std + mean


# ── Standard Paths ───────────────────────────────────────────────────────────

W3_LAT, W3_LON = 39.14, -77.01
G_LAT, G_LON = 51.50, -0.12


# ── Test Functions ───────────────────────────────────────────────────────────

def test_tst901_10m_band_closure(model, device):
    """TST-901: 10m Band Closure — Should drop at midnight in WINTER."""
    print("\n" + "=" * 60)
    print("TST-901: 10m Band Closure (Winter, Midnight UTC)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["10m"]
    band_id = 111
    winter_doy = 355
    winter_month = 12

    snr_midday = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                         freq_hz, sfi=150, kp=2, hour_utc=14,
                         month=winter_month, day_of_year=winter_doy)
    db_midday = sigma_to_db(snr_midday, band_id)

    snr_midnight = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                           freq_hz, sfi=150, kp=2, hour_utc=0,
                           month=winter_month, day_of_year=winter_doy)
    db_midnight = sigma_to_db(snr_midnight, band_id)

    delta_db = db_midday - db_midnight

    print(f"\n  Path: W3 → G, 10m, SFI 150, Kp 2, Winter (Dec 21)")
    print(f"  SNR at 14:00 UTC (Midday):   {db_midday:+.1f} dB ({snr_midday:+.3f} sigma)")
    print(f"  SNR at 00:00 UTC (Midnight): {db_midnight:+.1f} dB ({snr_midnight:+.3f} sigma)")
    print(f"\n  Day/Night Delta: {delta_db:+.1f} dB")

    if delta_db >= 10.0:
        print(f"\n  PASS: 10m shows {delta_db:+.1f} dB day/night delta")
        return True
    else:
        print(f"\n  FAIL: Day/night delta {delta_db:+.1f} dB < 10 dB threshold")
        return False


def test_tst901b_10m_summer_twilight(model, device):
    """TST-901b: 10m Summer Twilight — Should stay open in summer."""
    print("\n" + "=" * 60)
    print("TST-901b: 10m Summer Twilight (Stays Open)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["10m"]
    band_id = 111
    summer_doy = 172
    summer_month = 6

    snr_midday = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                         freq_hz, sfi=150, kp=2, hour_utc=14,
                         month=summer_month, day_of_year=summer_doy)
    db_midday = sigma_to_db(snr_midday, band_id)

    snr_midnight = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                           freq_hz, sfi=150, kp=2, hour_utc=0,
                           month=summer_month, day_of_year=summer_doy)
    db_midnight = sigma_to_db(snr_midnight, band_id)

    delta_db = db_midday - db_midnight

    print(f"\n  Path: W3 → G, 10m, SFI 150, Kp 2, Summer (Jun 21)")
    print(f"  SNR at 14:00 UTC (Midday):   {db_midday:+.1f} dB ({snr_midday:+.3f} sigma)")
    print(f"  SNR at 00:00 UTC (Midnight): {db_midnight:+.1f} dB ({snr_midnight:+.3f} sigma)")
    print(f"\n  Day/Night Delta: {delta_db:+.1f} dB")

    if abs(delta_db) < 5.0:
        print(f"\n  PASS: 10m shows weak day/night delta ({delta_db:+.1f} dB) — correct summer physics")
        return True
    else:
        print(f"\n  FAIL: Day/night delta {delta_db:+.1f} dB too large for summer")
        return False


def test_tst902_15m_band_closure(model, device):
    """TST-902: 15m Band Closure — Should show day/night delta in WINTER."""
    print("\n" + "=" * 60)
    print("TST-902: 15m Band Closure (Winter)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["15m"]
    band_id = 109
    winter_doy = 355

    snr_midday = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                         freq_hz, sfi=150, kp=2, hour_utc=14,
                         month=12, day_of_year=winter_doy)
    snr_midnight = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                           freq_hz, sfi=150, kp=2, hour_utc=0,
                           month=12, day_of_year=winter_doy)

    delta_db = (snr_midday - snr_midnight) * SIGMA_TO_DB

    print(f"\n  Path: W3 → G, 15m, Winter")
    print(f"  Day/Night Delta: {delta_db:+.1f} dB")

    if delta_db >= 8.0:
        print(f"\n  PASS: 15m shows {delta_db:+.1f} dB day/night delta")
        return True
    else:
        print(f"\n  FAIL: Day/night delta {delta_db:+.1f} dB < 8 dB threshold")
        return False


def test_tst903_160m_mutual_darkness(model, device):
    """TST-903: 160m Mutual Darkness — DX requires both ends dark."""
    print("\n" + "=" * 60)
    print("TST-903: 160m Mutual Darkness (Winter)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["160m"]
    winter_doy = 355

    snr_daylight = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                           freq_hz, sfi=100, kp=2, hour_utc=14,
                           month=12, day_of_year=winter_doy)
    snr_dark = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                       freq_hz, sfi=100, kp=2, hour_utc=4,
                       month=12, day_of_year=winter_doy)

    delta_db = (snr_dark - snr_daylight) * SIGMA_TO_DB

    print(f"\n  Path: W3 → G, 160m, Winter")
    print(f"  Dark vs Day Delta: {delta_db:+.1f} dB")

    if delta_db >= 5.0:
        print(f"\n  PASS: 160m shows {delta_db:+.1f} dB improvement at night")
        return True
    else:
        print(f"\n  FAIL: 160m day/night difference < 5 dB ({delta_db:+.1f})")
        return False


def test_tst904_80m_mutual_darkness(model, device):
    """TST-904: 80m Mutual Darkness — DX requires both ends dark."""
    print("\n" + "=" * 60)
    print("TST-904: 80m Mutual Darkness (Winter)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["80m"]
    winter_doy = 355

    snr_daylight = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                           freq_hz, sfi=100, kp=2, hour_utc=14,
                           month=12, day_of_year=winter_doy)
    snr_dark = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                       freq_hz, sfi=100, kp=2, hour_utc=4,
                       month=12, day_of_year=winter_doy)

    delta_db = (snr_dark - snr_daylight) * SIGMA_TO_DB

    print(f"\n  Path: W3 → G, 80m, Winter")
    print(f"  Dark vs Day Delta: {delta_db:+.1f} dB")

    if delta_db >= 4.0:
        print(f"\n  PASS: 80m shows {delta_db:+.1f} dB improvement at night")
        return True
    else:
        print(f"\n  FAIL: 80m day/night difference < 4 dB ({delta_db:+.1f})")
        return False


def test_tst905_band_ordering_day(model, device):
    """TST-905: Band Ordering Day — High bands beat low bands at midday."""
    print("\n" + "=" * 60)
    print("TST-905: Band Ordering (Daytime)")
    print("=" * 60)

    hour_utc = 14
    sfi, kp = 150, 2

    snr_10m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ["10m"], sfi, kp, hour_utc)
    snr_15m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ["15m"], sfi, kp, hour_utc)
    snr_20m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ["20m"], sfi, kp, hour_utc)
    snr_80m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ["80m"], sfi, kp, hour_utc)
    snr_160m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                       BAND_FREQ_HZ["160m"], sfi, kp, hour_utc)

    high_band_avg = (snr_10m + snr_15m + snr_20m) / 3
    low_band_avg = (snr_80m + snr_160m) / 2
    delta = high_band_avg - low_band_avg

    print(f"\n  Path: W3 → G, SFI 150, Kp 2, 14:00 UTC")
    print(f"  High band avg: {high_band_avg:+.3f}σ")
    print(f"  Low band avg:  {low_band_avg:+.3f}σ")
    print(f"  Delta: {delta:+.3f}σ ({delta * SIGMA_TO_DB:+.1f} dB)")

    if delta > 0:
        print(f"\n  PASS: High bands better than low bands at midday")
        return True
    else:
        print(f"\n  FAIL: Low bands better than high bands at midday (wrong)")
        return False


def test_tst906_band_ordering_night(model, device):
    """TST-906: Band Ordering Night — Low bands beat high bands at midnight."""
    print("\n" + "=" * 60)
    print("TST-906: Band Ordering (Winter Nighttime)")
    print("=" * 60)

    hour_utc = 4
    sfi, kp = 100, 2
    winter_doy = 355

    snr_10m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ["10m"], sfi, kp, hour_utc,
                      month=12, day_of_year=winter_doy)
    snr_15m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ["15m"], sfi, kp, hour_utc,
                      month=12, day_of_year=winter_doy)
    snr_40m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ["40m"], sfi, kp, hour_utc,
                      month=12, day_of_year=winter_doy)
    snr_80m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ["80m"], sfi, kp, hour_utc,
                      month=12, day_of_year=winter_doy)

    high_band_avg = (snr_10m + snr_15m) / 2
    low_band_avg = (snr_40m + snr_80m) / 2
    delta = low_band_avg - high_band_avg

    print(f"\n  Path: W3 → G, SFI 100, Kp 2, 04:00 UTC, Winter")
    print(f"  High band avg (10m/15m): {high_band_avg:+.3f}σ")
    print(f"  Low band avg (40m/80m):  {low_band_avg:+.3f}σ")
    print(f"  Delta (low - high): {delta:+.3f}σ ({delta * SIGMA_TO_DB:+.1f} dB)")

    if delta > 0:
        print(f"\n  PASS: Low bands better than high bands at night")
        return True
    else:
        print(f"\n  FAIL: High bands still better than low bands at night (wrong)")
        return False


def test_tst907_time_sensitivity(model, device):
    """TST-907: Time Sensitivity — Same path >= 6 dB peak vs off-peak."""
    print("\n" + "=" * 60)
    print("TST-907: Time Sensitivity (Winter, Peak vs Off-Peak)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["20m"]
    sfi, kp = 150, 2
    winter_doy = 355

    snr_by_hour = {}
    for hour in range(24):
        snr = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      freq_hz, sfi, kp, hour, month=12, day_of_year=winter_doy)
        snr_by_hour[hour] = snr

    peak_hour = max(snr_by_hour, key=snr_by_hour.get)
    trough_hour = min(snr_by_hour, key=snr_by_hour.get)
    delta_sigma = snr_by_hour[peak_hour] - snr_by_hour[trough_hour]
    delta_db = delta_sigma * SIGMA_TO_DB

    print(f"\n  Path: W3 → G, 20m, SFI 150, Winter")
    print(f"  Peak:   {peak_hour:02d}:00 UTC")
    print(f"  Trough: {trough_hour:02d}:00 UTC")
    print(f"  Dynamic Range: {delta_db:+.1f} dB")

    if delta_db >= 6.0:
        print(f"\n  PASS: >= 6 dB time sensitivity")
        return True
    else:
        print(f"\n  FAIL: < 6 dB time sensitivity")
        return False


def test_tst908_10m_peak_hour(model, device):
    """TST-908: 10m Peak Hour — Should peak near local noon."""
    print("\n" + "=" * 60)
    print("TST-908: 10m Peak Hour (Winter, Should be near Noon)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["10m"]
    sfi, kp = 180, 2
    winter_doy = 355

    snr_by_hour = {}
    for hour in range(24):
        snr = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      freq_hz, sfi, kp, hour, month=12, day_of_year=winter_doy)
        snr_by_hour[hour] = snr

    peak_hour = max(snr_by_hour, key=snr_by_hour.get)

    print(f"\n  Path: W3 → G, 10m, SFI 180, Winter")
    print(f"  Peak Hour: {peak_hour:02d}:00 UTC")
    print(f"  Expected: 10-16 UTC")

    if 10 <= peak_hour <= 16:
        print(f"\n  PASS: 10m peaks during daylight hours")
        return True
    else:
        print(f"\n  FAIL: 10m peak at {peak_hour:02d}:00 UTC (outside expected range)")
        return False


def test_tst909_160m_peak_hour(model, device):
    """TST-909: 160m Peak Hour — Should peak near local midnight."""
    print("\n" + "=" * 60)
    print("TST-909: 160m Peak Hour (Winter, Should be near Midnight)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["160m"]
    sfi, kp = 80, 2
    winter_doy = 355

    snr_by_hour = {}
    for hour in range(24):
        snr = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      freq_hz, sfi, kp, hour, month=12, day_of_year=winter_doy)
        snr_by_hour[hour] = snr

    peak_hour = max(snr_by_hour, key=snr_by_hour.get)

    print(f"\n  Path: W3 → G, 160m, SFI 80, Winter")
    print(f"  Peak Hour: {peak_hour:02d}:00 UTC")
    print(f"  Expected: 00-06 or 22-24 UTC")

    if peak_hour <= 6 or peak_hour >= 22:
        print(f"\n  PASS: 160m peaks during night hours")
        return True
    else:
        print(f"\n  FAIL: 160m peak at {peak_hour:02d}:00 UTC (should be night)")
        return False


def test_tst910_40m_gray_line(model, device):
    """TST-910: 40m Gray Line Enhancement."""
    print("\n" + "=" * 60)
    print("TST-910: 40m Gray Line Enhancement")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["40m"]
    sfi, kp = 120, 2

    snr_noon = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                       freq_hz, sfi, kp, hour_utc=14)
    snr_twilight = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                           freq_hz, sfi, kp, hour_utc=0)

    twilight_boost = snr_twilight - snr_noon
    twilight_db = twilight_boost * SIGMA_TO_DB

    print(f"\n  Path: W3 → G, 40m, SFI 120")
    print(f"  Twilight vs Noon: {twilight_db:+.1f} dB")

    if twilight_boost >= -0.2:
        print(f"\n  PASS: 40m shows gray line enhancement or stability")
        return True
    else:
        print(f"\n  FAIL: 40m shows unexpected twilight degradation")
        return False


def test_v25_beta_sfi_differentiation(model, device):
    """V25-beta: Verify sun sidecar shows band-dependent SFI effects."""
    print("\n" + "=" * 60)
    print("V25-beta: Band-Dependent SFI Effects (Product Input)")
    print("=" * 60)

    sfi_effects = model.get_sun_effect_by_band(200.0 / 300.0, device)
    sfi_effects_low = model.get_sun_effect_by_band(70.0 / 300.0, device)

    print(f"\n  SFI=200 vs SFI=70 effect by band:")
    for band in ["160m", "80m", "40m", "30m", "20m", "15m", "10m"]:
        delta = sfi_effects[band] - sfi_effects_low[band]
        print(f"    {band:>4s}: {delta:+.3f}σ")

    sfi_10m = sfi_effects["10m"] - sfi_effects_low["10m"]
    sfi_160m = sfi_effects["160m"] - sfi_effects_low["160m"]
    differentiation = sfi_10m - sfi_160m

    print(f"\n  10m SFI benefit: {sfi_10m:+.3f}σ")
    print(f"  160m SFI benefit: {sfi_160m:+.3f}σ")
    print(f"  Differentiation: {differentiation:+.3f}σ")

    if abs(differentiation) > 0.1:
        print(f"\n  PASS: Sun sidecar is band-aware (delta > 0.1σ)")
        return True
    else:
        print(f"\n  FAIL: Product input did not produce band differentiation")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  IONIS V25-beta — TST-900 Band×Time Discrimination Tests")
    print("=" * 60)
    print(f"\n  Sun sidecar: SFI * freq_log (1D product input)")
    print("  Hypothesis: Forced interaction prevents optimizer from")
    print("  separating SFI from frequency.")

    if not os.path.exists(CKPT_PATH):
        print(f"\n  ERROR: Checkpoint not found: {CKPT_PATH}")
        return 1

    print(f"\nLoading {CKPT_PATH}...")
    state_dict = load_safetensors(CKPT_PATH, device=str(DEVICE))

    meta_path = CKPT_PATH.replace(".safetensors", "_meta.json")
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)

    model = IonisGateV25Beta(
        dnn_dim=DNN_DIM,
        sidecar_hidden=SIDECAR_HIDDEN,
        sfi_idx=SFI_IDX,
        kp_penalty_idx=KP_PENALTY_IDX,
        sun_sidecar_input_dim=SUN_SIDECAR_INPUT_DIM,
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"  Device: {DEVICE}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  RMSE: {metadata.get('val_rmse', 0):.4f}σ")
    print(f"  Pearson: {metadata.get('val_pearson', 0):+.4f}")
    print(f"  SFI 10m: {metadata.get('sfi_10m', 0):+.3f}σ")
    print(f"  SFI 160m: {metadata.get('sfi_160m', 0):+.3f}σ")

    # V25-beta specific test
    print("\n" + "=" * 60)
    print("  V25-BETA SIDECAR VERIFICATION")
    print("=" * 60)
    test_v25_beta_sfi_differentiation(model, DEVICE)

    # Standard TST-900 tests
    results = []
    results.append(("TST-901", "10m Band Closure (Winter)", test_tst901_10m_band_closure(model, DEVICE)))
    results.append(("TST-901b", "10m Summer Twilight", test_tst901b_10m_summer_twilight(model, DEVICE)))
    results.append(("TST-902", "15m Band Closure (Winter)", test_tst902_15m_band_closure(model, DEVICE)))
    results.append(("TST-903", "160m Mutual Darkness", test_tst903_160m_mutual_darkness(model, DEVICE)))
    results.append(("TST-904", "80m Mutual Darkness", test_tst904_80m_mutual_darkness(model, DEVICE)))
    results.append(("TST-905", "Band Order Day", test_tst905_band_ordering_day(model, DEVICE)))
    results.append(("TST-906", "Band Order Night", test_tst906_band_ordering_night(model, DEVICE)))
    results.append(("TST-907", "Time Sensitivity", test_tst907_time_sensitivity(model, DEVICE)))
    results.append(("TST-908", "10m Peak Hour", test_tst908_10m_peak_hour(model, DEVICE)))
    results.append(("TST-909", "160m Peak Hour", test_tst909_160m_peak_hour(model, DEVICE)))
    results.append(("TST-910", "40m Gray Line", test_tst910_40m_gray_line(model, DEVICE)))

    # Summary
    print("\n" + "=" * 60)
    print("  TST-900 SUMMARY: Band×Time Discrimination")
    print("=" * 60)

    passed = sum(1 for _, _, p in results if p)
    total = len(results)

    for test_id, name, p in results:
        status = "PASS" if p else "FAIL"
        print(f"  {test_id}: {name:<25s}  {status}")

    print()
    print(f"  Results: {passed}/{total} passed")

    if passed >= 10:
        print("\n  V25-beta TARGET ACHIEVED: TST-900 >= 10/11")
        return 0
    elif passed >= 9:
        print("\n  V25-beta MATCHES V22-gamma: TST-900 = 9/11")
        return 0
    else:
        print(f"\n  V25-beta REGRESSION: TST-900 = {passed}/11 (< 9/11)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
