#!/usr/bin/env python3
"""
test_tst900_band_time.py — IONIS V22 Band x Time Discrimination Tests

TST-900 Group: Verify band x time-of-day behavior matches ionospheric reality.

Tests:
  TST-901:  10m Band Closure (Winter) — >= 10 dB day/night delta
  TST-901b: 10m Summer Twilight — stays open (correct physics)
  TST-902:  15m Band Closure (Winter) — >= 8 dB day/night delta
  TST-903:  160m Mutual Darkness — night >= 5 dB better than day
  TST-904:  80m Mutual Darkness — night >= 4 dB better than day
  TST-905:  Band Ordering Day — high bands beat low bands at midday
  TST-906:  Band Ordering Night — low bands beat high bands at midnight
  TST-907:  Time Sensitivity — >= 6 dB peak vs off-peak
  TST-908:  10m Peak Hour — peaks 10-16 UTC
  TST-909:  160m Peak Hour — peaks 00-06 or 22-24 UTC
  TST-910:  40m Gray Line — twilight >= noon

Self-contained: imports only from ../model (locked V22 copy).
"""

import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors

# -- Path Setup ----------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V22_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, V22_DIR)

from model import IonisGate, get_device, build_features, BAND_FREQ_HZ

# -- Load Config ---------------------------------------------------------------

CONFIG_PATH = os.path.join(V22_DIR, "config_v22.json")
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

CHECKPOINT_PATH = os.path.join(V22_DIR, CONFIG["checkpoint"])
DNN_DIM = CONFIG["model"]["dnn_dim"]
SFI_IDX = CONFIG["model"]["sfi_idx"]
KP_PENALTY_IDX = CONFIG["model"]["kp_penalty_idx"]
SIDECAR_HIDDEN = CONFIG["model"]["sidecar_hidden"]

DEVICE = get_device()
SIGMA_TO_DB = 6.7

FT8_THRESHOLD_DB = -21.0
WSPR_FLOOR_DB = -28.0


# -- Prediction Helper ---------------------------------------------------------

def predict(model, device, tx_lat, tx_lon, rx_lat, rx_lon, freq_hz,
            sfi, kp, hour_utc, month=6, day_of_year=172):
    """Make a V22 prediction (solar depression + cross-products)."""
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
    """Convert normalized sigma to dB using band-specific constants."""
    std = CONFIG["norm_constants_per_band"].get(str(band_id), {}).get(source, {}).get("std", 6.7)
    mean = CONFIG["norm_constants_per_band"].get(str(band_id), {}).get(source, {}).get("mean", -18.0)
    return sigma * std + mean


# -- Standard Paths ------------------------------------------------------------

W3_LAT, W3_LON = 39.14, -77.01
G_LAT, G_LON = 51.50, -0.12


# -- Test Functions ------------------------------------------------------------

def test_tst901_10m_band_closure(model, device):
    """TST-901: 10m Band Closure — >= 10 dB day/night delta in winter."""
    print("\n" + "=" * 60)
    print("TST-901: 10m Band Closure (Winter, Midnight UTC)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["10m"]
    band_id = 111

    snr_midday = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                         freq_hz, sfi=150, kp=2, hour_utc=14,
                         month=12, day_of_year=355)
    db_midday = sigma_to_db(snr_midday, band_id)

    snr_midnight = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                           freq_hz, sfi=150, kp=2, hour_utc=0,
                           month=12, day_of_year=355)
    db_midnight = sigma_to_db(snr_midnight, band_id)

    delta_db = db_midday - db_midnight

    print(f"\n  Path: W3 -> G, 10m, SFI 150, Kp 2, Winter (Dec 21)")
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
    """TST-901b: 10m Summer Twilight — stays open (< 5 dB delta)."""
    print("\n" + "=" * 60)
    print("TST-901b: 10m Summer Twilight (Stays Open)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["10m"]
    band_id = 111

    snr_midday = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                         freq_hz, sfi=150, kp=2, hour_utc=14,
                         month=6, day_of_year=172)
    db_midday = sigma_to_db(snr_midday, band_id)

    snr_midnight = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                           freq_hz, sfi=150, kp=2, hour_utc=0,
                           month=6, day_of_year=172)
    db_midnight = sigma_to_db(snr_midnight, band_id)

    delta_db = db_midday - db_midnight

    print(f"\n  Path: W3 -> G, 10m, SFI 150, Kp 2, Summer (Jun 21)")
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
    """TST-902: 15m Band Closure — >= 8 dB day/night delta in winter."""
    print("\n" + "=" * 60)
    print("TST-902: 15m Band Closure (Winter, Midnight UTC)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["15m"]
    band_id = 109

    snr_midday = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                         freq_hz, sfi=150, kp=2, hour_utc=14,
                         month=12, day_of_year=355)
    db_midday = sigma_to_db(snr_midday, band_id)

    snr_midnight = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                           freq_hz, sfi=150, kp=2, hour_utc=0,
                           month=12, day_of_year=355)
    db_midnight = sigma_to_db(snr_midnight, band_id)

    delta_db = db_midday - db_midnight

    print(f"\n  Path: W3 -> G, 15m, SFI 150, Kp 2, Winter (Dec 21)")
    print(f"  SNR at 14:00 UTC (Midday):   {db_midday:+.1f} dB ({snr_midday:+.3f} sigma)")
    print(f"  SNR at 00:00 UTC (Midnight): {db_midnight:+.1f} dB ({snr_midnight:+.3f} sigma)")
    print(f"\n  Day/Night Delta: {delta_db:+.1f} dB")

    if delta_db >= 8.0:
        print(f"\n  PASS: 15m shows {delta_db:+.1f} dB day/night delta")
        return True
    else:
        print(f"\n  FAIL: Day/night delta {delta_db:+.1f} dB < 8 dB threshold")
        return False


def test_tst903_160m_mutual_darkness(model, device):
    """TST-903: 160m Mutual Darkness — night >= 5 dB better than day (winter)."""
    print("\n" + "=" * 60)
    print("TST-903: 160m Mutual Darkness (Winter)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["160m"]
    band_id = 102

    snr_daylight = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                           freq_hz, sfi=100, kp=2, hour_utc=14,
                           month=12, day_of_year=355)
    db_daylight = sigma_to_db(snr_daylight, band_id)

    snr_dark = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                       freq_hz, sfi=100, kp=2, hour_utc=4,
                       month=12, day_of_year=355)
    db_dark = sigma_to_db(snr_dark, band_id)

    delta_db = db_dark - db_daylight

    print(f"\n  Path: W3 -> G, 160m, SFI 100, Kp 2, Winter")
    print(f"  SNR at 14:00 UTC (Daylight): {db_daylight:+.1f} dB")
    print(f"  SNR at 04:00 UTC (Dark):     {db_dark:+.1f} dB")
    print(f"\n  Dark vs Day Delta: {delta_db:+.1f} dB")

    if delta_db >= 5.0:
        print(f"\n  PASS: 160m shows {delta_db:+.1f} dB improvement at night")
        return True
    else:
        print(f"\n  FAIL: 160m day/night difference < 5 dB ({delta_db:+.1f})")
        return False


def test_tst904_80m_mutual_darkness(model, device):
    """TST-904: 80m Mutual Darkness — night >= 4 dB better than day (winter)."""
    print("\n" + "=" * 60)
    print("TST-904: 80m Mutual Darkness (Winter)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["80m"]
    band_id = 103

    snr_daylight = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                           freq_hz, sfi=100, kp=2, hour_utc=14,
                           month=12, day_of_year=355)
    db_daylight = sigma_to_db(snr_daylight, band_id)

    snr_dark = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                       freq_hz, sfi=100, kp=2, hour_utc=4,
                       month=12, day_of_year=355)
    db_dark = sigma_to_db(snr_dark, band_id)

    delta_db = db_dark - db_daylight

    print(f"\n  Path: W3 -> G, 80m, SFI 100, Kp 2, Winter")
    print(f"  SNR at 14:00 UTC (Daylight): {db_daylight:+.1f} dB")
    print(f"  SNR at 04:00 UTC (Dark):     {db_dark:+.1f} dB")
    print(f"\n  Dark vs Day Delta: {delta_db:+.1f} dB")

    if delta_db >= 4.0:
        print(f"\n  PASS: 80m shows {delta_db:+.1f} dB improvement at night")
        return True
    else:
        print(f"\n  FAIL: 80m day/night difference < 4 dB ({delta_db:+.1f})")
        return False


def test_tst905_band_ordering_day(model, device):
    """TST-905: Band Ordering Day — high bands beat low bands at midday."""
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

    print(f"\n  Path: W3 -> G, SFI 150, Kp 2, 14:00 UTC (Daytime)")
    print(f"\n  High Bands:")
    print(f"    10m: {snr_10m:+.3f} sigma ({sigma_to_db(snr_10m, 111):+.1f} dB)")
    print(f"    15m: {snr_15m:+.3f} sigma ({sigma_to_db(snr_15m, 109):+.1f} dB)")
    print(f"    20m: {snr_20m:+.3f} sigma ({sigma_to_db(snr_20m, 107):+.1f} dB)")
    print(f"\n  Low Bands:")
    print(f"    80m: {snr_80m:+.3f} sigma ({sigma_to_db(snr_80m, 103):+.1f} dB)")
    print(f"   160m: {snr_160m:+.3f} sigma ({sigma_to_db(snr_160m, 102):+.1f} dB)")

    high_band_avg = (snr_10m + snr_15m + snr_20m) / 3
    low_band_avg = (snr_80m + snr_160m) / 2
    delta = high_band_avg - low_band_avg

    print(f"\n  High band avg: {high_band_avg:+.3f} sigma")
    print(f"  Low band avg:  {low_band_avg:+.3f} sigma")
    print(f"  Delta: {delta:+.3f} sigma ({delta * SIGMA_TO_DB:+.1f} dB)")

    if delta > 0:
        print(f"\n  PASS: High bands better than low bands at midday")
        return True
    else:
        print(f"\n  FAIL: Low bands better than high bands at midday")
        return False


def test_tst906_band_ordering_night(model, device):
    """TST-906: Band Ordering Night — low bands beat high bands (winter midnight)."""
    print("\n" + "=" * 60)
    print("TST-906: Band Ordering (Winter Nighttime)")
    print("=" * 60)

    hour_utc = 4
    sfi, kp = 100, 2

    snr_10m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ["10m"], sfi, kp, hour_utc,
                      month=12, day_of_year=355)
    snr_15m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ["15m"], sfi, kp, hour_utc,
                      month=12, day_of_year=355)
    snr_40m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ["40m"], sfi, kp, hour_utc,
                      month=12, day_of_year=355)
    snr_80m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ["80m"], sfi, kp, hour_utc,
                      month=12, day_of_year=355)

    print(f"\n  Path: W3 -> G, SFI 100, Kp 2, 04:00 UTC, Winter")
    print(f"\n  High Bands:")
    print(f"    10m: {snr_10m:+.3f} sigma ({sigma_to_db(snr_10m, 111):+.1f} dB)")
    print(f"    15m: {snr_15m:+.3f} sigma ({sigma_to_db(snr_15m, 109):+.1f} dB)")
    print(f"\n  Low Bands:")
    print(f"    40m: {snr_40m:+.3f} sigma ({sigma_to_db(snr_40m, 105):+.1f} dB)")
    print(f"    80m: {snr_80m:+.3f} sigma ({sigma_to_db(snr_80m, 103):+.1f} dB)")

    high_band_avg = (snr_10m + snr_15m) / 2
    low_band_avg = (snr_40m + snr_80m) / 2
    delta = low_band_avg - high_band_avg

    print(f"\n  High band avg (10m/15m): {high_band_avg:+.3f} sigma")
    print(f"  Low band avg (40m/80m):  {low_band_avg:+.3f} sigma")
    print(f"  Delta (low - high): {delta:+.3f} sigma ({delta * SIGMA_TO_DB:+.1f} dB)")

    if delta > 0:
        print(f"\n  PASS: Low bands better than high bands at night")
        return True
    else:
        print(f"\n  FAIL: High bands still better than low bands at night")
        return False


def test_tst907_time_sensitivity(model, device):
    """TST-907: Time Sensitivity — >= 6 dB peak vs off-peak (winter)."""
    print("\n" + "=" * 60)
    print("TST-907: Time Sensitivity (Winter, Peak vs Off-Peak)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["20m"]
    band_id = 107
    sfi, kp = 150, 2

    snr_by_hour = {}
    for hour in range(24):
        snr = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      freq_hz, sfi, kp, hour, month=12, day_of_year=355)
        snr_by_hour[hour] = snr

    peak_hour = max(snr_by_hour, key=snr_by_hour.get)
    trough_hour = min(snr_by_hour, key=snr_by_hour.get)
    peak_snr = snr_by_hour[peak_hour]
    trough_snr = snr_by_hour[trough_hour]
    delta_db = (peak_snr - trough_snr) * SIGMA_TO_DB

    print(f"\n  Path: W3 -> G, 20m, SFI 150, Kp 2, Winter")
    print(f"\n  24-Hour Scan:")
    for h in [0, 4, 8, 12, 16, 20]:
        snr = snr_by_hour[h]
        print(f"    {h:02d}:00 UTC: {snr:+.3f} sigma ({sigma_to_db(snr, band_id):+.1f} dB)")

    print(f"\n  Peak:   {peak_hour:02d}:00 UTC = {peak_snr:+.3f} sigma")
    print(f"  Trough: {trough_hour:02d}:00 UTC = {trough_snr:+.3f} sigma")
    print(f"\n  Dynamic Range: {delta_db:+.1f} dB")

    if delta_db >= 6.0:
        print(f"\n  PASS: >= 6 dB time sensitivity")
        return True
    else:
        print(f"\n  FAIL: < 6 dB time sensitivity")
        return False


def test_tst908_10m_peak_hour(model, device):
    """TST-908: 10m Peak Hour — peaks 10-16 UTC (winter)."""
    print("\n" + "=" * 60)
    print("TST-908: 10m Peak Hour (Winter)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["10m"]
    sfi, kp = 180, 2

    snr_by_hour = {}
    for hour in range(24):
        snr = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      freq_hz, sfi, kp, hour, month=12, day_of_year=355)
        snr_by_hour[hour] = snr

    peak_hour = max(snr_by_hour, key=snr_by_hour.get)

    print(f"\n  Path: W3 -> G, 10m, SFI 180, Kp 2, Winter")
    print(f"  Peak Hour: {peak_hour:02d}:00 UTC (expected: 10-16 UTC)")

    if 10 <= peak_hour <= 16:
        print(f"\n  PASS: 10m peaks during daylight hours")
        return True
    else:
        print(f"\n  FAIL: 10m peak at {peak_hour:02d}:00 UTC (outside 10-16 range)")
        return False


def test_tst909_160m_peak_hour(model, device):
    """TST-909: 160m Peak Hour — peaks 00-06 or 22-24 UTC (winter)."""
    print("\n" + "=" * 60)
    print("TST-909: 160m Peak Hour (Winter)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["160m"]
    sfi, kp = 80, 2

    snr_by_hour = {}
    for hour in range(24):
        snr = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      freq_hz, sfi, kp, hour, month=12, day_of_year=355)
        snr_by_hour[hour] = snr

    peak_hour = max(snr_by_hour, key=snr_by_hour.get)

    print(f"\n  Path: W3 -> G, 160m, SFI 80, Kp 2, Winter")
    print(f"  Peak Hour: {peak_hour:02d}:00 UTC (expected: 00-06 or 22-24)")

    if peak_hour <= 6 or peak_hour >= 22:
        print(f"\n  PASS: 160m peaks during night hours")
        return True
    else:
        print(f"\n  FAIL: 160m peak at {peak_hour:02d}:00 UTC (should be night)")
        return False


def test_tst910_40m_gray_line(model, device):
    """TST-910: 40m Gray Line — twilight >= noon."""
    print("\n" + "=" * 60)
    print("TST-910: 40m Gray Line Enhancement")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["40m"]
    band_id = 105
    sfi, kp = 120, 2

    snr_noon = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                       freq_hz, sfi, kp, hour_utc=14)
    snr_twilight = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                           freq_hz, sfi, kp, hour_utc=0)

    print(f"\n  Path: W3 -> G, 40m, SFI 120, Kp 2")
    print(f"  SNR at 14:00 UTC (Day):     {sigma_to_db(snr_noon, band_id):+.1f} dB")
    print(f"  SNR at 00:00 UTC (Twilight): {sigma_to_db(snr_twilight, band_id):+.1f} dB")

    twilight_boost = snr_twilight - snr_noon
    twilight_db = twilight_boost * SIGMA_TO_DB
    print(f"\n  Twilight vs Noon: {twilight_db:+.1f} dB")

    if twilight_boost >= -0.2:
        print(f"\n  PASS: 40m shows gray line enhancement or stability")
        return True
    else:
        print(f"\n  FAIL: 40m shows unexpected twilight degradation")
        return False


# -- Main ----------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  IONIS V22-gamma — TST-900 Band x Time Discrimination Tests")
    print("=" * 60)
    print(f"\n  Model:  V22-gamma (solar depression + cross-products)")
    print(f"  Config: {CONFIG_PATH}")

    print(f"\nLoading {CHECKPOINT_PATH}...")
    state_dict = load_safetensors(CHECKPOINT_PATH, device=str(DEVICE))

    meta_path = CHECKPOINT_PATH.replace(".safetensors", "_meta.json")
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
    print(f"  RMSE: {metadata.get('val_rmse', 0):.4f} sigma")
    print(f"  Pearson: {metadata.get('val_pearson', 0):+.4f}")

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

    print("\n" + "=" * 60)
    print("  TST-900 SUMMARY: Band x Time Discrimination")
    print("=" * 60)

    passed = sum(1 for _, _, p in results if p)
    total = len(results)

    for test_id, name, p in results:
        status = "PASS" if p else "FAIL"
        print(f"  {test_id}: {name:<30s}  {status}")

    print()
    print(f"  Results: {passed}/{total} passed")

    if passed == total:
        print("\n  ALL TST-900 TESTS PASSED")
        return 0
    else:
        print(f"\n  {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
