#!/usr/bin/env python3
"""
test_tst900_band_time.py — IONIS V21 Band×Time Discrimination Tests

TST-900 Group: Verify band×time-of-day behavior matches ionospheric reality.

Tests:
  TST-901: Band Closure — 10m should drop below FT8 at midnight UTC
  TST-902: Band Closure — 15m should drop below FT8 at midnight UTC
  TST-903: Mutual Darkness — 160m DX requires both ends dark
  TST-904: Mutual Darkness — 80m DX requires both ends dark
  TST-905: Band Ordering Day — High bands beat low bands at midday
  TST-906: Band Ordering Night — Low bands beat high bands at midnight
  TST-907: Time Sensitivity — Same path shows >= 10 dB day/night difference
  TST-908: Peak Hour — 10m peaks near local noon
  TST-909: Peak Hour — 160m peaks near local midnight
  TST-910: Gray Line — 40m shows enhanced propagation at twilight

These tests document the band×time sensitivity issues discovered in V20.
V21 should pass these after time sensitivity improvements.

Discovery: 2026-02-20, ham-stats.com showed V20 predicting CW to EU on 10m at
midnight UTC. The model learned distance but not band×time behavior.
"""

import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors

# ── Path Setup ───────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V21_DIR = os.path.dirname(SCRIPT_DIR)
VERSIONS_DIR = os.path.dirname(V21_DIR)
sys.path.insert(0, os.path.join(VERSIONS_DIR, "common"))

from model import IonisGate, get_device, build_features, BAND_FREQ_HZ

# ── Load Config ──────────────────────────────────────────────────────────────

# Initially test against V20 to establish baseline
V20_DIR = os.path.join(VERSIONS_DIR, "v20")
CONFIG_PATH = os.path.join(V20_DIR, "config_v20.json")

with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

MODEL_PATH = os.path.join(V20_DIR, CONFIG["checkpoint"])
DNN_DIM = CONFIG["model"]["dnn_dim"]
SFI_IDX = CONFIG["model"]["sfi_idx"]
KP_PENALTY_IDX = CONFIG["model"]["kp_penalty_idx"]
SIDECAR_HIDDEN = CONFIG["model"]["sidecar_hidden"]

DEVICE = get_device()

SIGMA_TO_DB = 6.7  # Approximate conversion factor

# Thresholds
FT8_THRESHOLD_DB = -21.0  # FT8 decode limit
WSPR_FLOOR_DB = -28.0  # WSPR absolute floor


# ── Prediction Helper ────────────────────────────────────────────────────────

def predict(model, device, tx_lat, tx_lon, rx_lat, rx_lon, freq_hz,
            sfi, kp, hour_utc, month=6):
    """Make a prediction for given parameters."""
    features = build_features(
        tx_lat, tx_lon, rx_lat, rx_lon, freq_hz,
        sfi, kp, hour_utc, month,
    )
    tensor = torch.tensor([features], dtype=torch.float32, device=device)
    with torch.no_grad():
        return model(tensor).item()


def sigma_to_db(sigma, band_id=107, source="wspr"):
    """Convert normalized sigma to dB using band-specific constants."""
    std = CONFIG["norm_constants_per_band"].get(str(band_id), {}).get(source, {}).get("std", 6.7)
    mean = CONFIG["norm_constants_per_band"].get(str(band_id), {}).get(source, {}).get("mean", -18.0)
    return sigma * std + mean


# ── Standard Paths ───────────────────────────────────────────────────────────

# W3 (Maryland) to G (London) — transatlantic, ~5700 km
W3_LAT, W3_LON = 39.14, -77.01
G_LAT, G_LON = 51.50, -0.12

# KH6 (Hawaii) to JA (Tokyo) — transpacific, ~6200 km
KH6_LAT, KH6_LON = 21.31, -157.86
JA_LAT, JA_LON = 35.68, 139.69

# W6 (California) to VK (Sydney) — long path, ~12000 km
W6_LAT, W6_LON = 37.77, -122.42
VK_LAT, VK_LON = -33.87, 151.21


# ── Test Functions ───────────────────────────────────────────────────────────

def test_tst901_10m_band_closure(model, device):
    """TST-901: 10m Band Closure — Should drop below FT8 at midnight UTC."""
    print("\n" + "=" * 60)
    print("TST-901: 10m Band Closure (Midnight UTC)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["10m"]
    band_id = 111

    # Midday prediction
    snr_midday = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                         freq_hz, sfi=150, kp=2, hour_utc=14)
    db_midday = sigma_to_db(snr_midday, band_id)

    # Midnight prediction
    snr_midnight = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                           freq_hz, sfi=150, kp=2, hour_utc=0)
    db_midnight = sigma_to_db(snr_midnight, band_id)

    delta_db = db_midday - db_midnight

    print(f"\n  Path: W3 → G, 10m, SFI 150, Kp 2")
    print(f"  SNR at 14:00 UTC (Midday):   {db_midday:+.1f} dB ({snr_midday:+.3f} sigma)")
    print(f"  SNR at 00:00 UTC (Midnight): {db_midnight:+.1f} dB ({snr_midnight:+.3f} sigma)")
    print(f"\n  Day/Night Delta: {delta_db:+.1f} dB")
    print(f"  FT8 Threshold: {FT8_THRESHOLD_DB} dB")

    # Pass criteria: midnight should be below FT8 threshold (-21 dB)
    if db_midnight < FT8_THRESHOLD_DB:
        print(f"\n  PASS: 10m below FT8 threshold at midnight (band closed)")
        return True
    else:
        print(f"\n  FAIL: 10m still above FT8 at midnight ({db_midnight:+.1f} > {FT8_THRESHOLD_DB})")
        print(f"  Expected: Band should be dead at midnight UTC for transatlantic")
        return False


def test_tst902_15m_band_closure(model, device):
    """TST-902: 15m Band Closure — Should drop below FT8 at midnight UTC."""
    print("\n" + "=" * 60)
    print("TST-902: 15m Band Closure (Midnight UTC)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["15m"]
    band_id = 109

    # Midday prediction
    snr_midday = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                         freq_hz, sfi=150, kp=2, hour_utc=14)
    db_midday = sigma_to_db(snr_midday, band_id)

    # Midnight prediction
    snr_midnight = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                           freq_hz, sfi=150, kp=2, hour_utc=0)
    db_midnight = sigma_to_db(snr_midnight, band_id)

    delta_db = db_midday - db_midnight

    print(f"\n  Path: W3 → G, 15m, SFI 150, Kp 2")
    print(f"  SNR at 14:00 UTC (Midday):   {db_midday:+.1f} dB ({snr_midday:+.3f} sigma)")
    print(f"  SNR at 00:00 UTC (Midnight): {db_midnight:+.1f} dB ({snr_midnight:+.3f} sigma)")
    print(f"\n  Day/Night Delta: {delta_db:+.1f} dB")
    print(f"  FT8 Threshold: {FT8_THRESHOLD_DB} dB")

    # Pass criteria: midnight should be below FT8 threshold
    if db_midnight < FT8_THRESHOLD_DB:
        print(f"\n  PASS: 15m below FT8 threshold at midnight (band closed)")
        return True
    else:
        print(f"\n  FAIL: 15m still above FT8 at midnight ({db_midnight:+.1f} > {FT8_THRESHOLD_DB})")
        return False


def test_tst903_160m_mutual_darkness(model, device):
    """TST-903: 160m Mutual Darkness — DX requires both ends dark."""
    print("\n" + "=" * 60)
    print("TST-903: 160m Mutual Darkness Requirement")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["160m"]
    band_id = 102

    # Both ends daylight (10:00 UTC = midday in Atlantic)
    snr_daylight = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                           freq_hz, sfi=100, kp=2, hour_utc=10)
    db_daylight = sigma_to_db(snr_daylight, band_id)

    # Both ends dark (04:00 UTC = night in Atlantic)
    snr_dark = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                       freq_hz, sfi=100, kp=2, hour_utc=4)
    db_dark = sigma_to_db(snr_dark, band_id)

    delta_db = db_dark - db_daylight

    print(f"\n  Path: W3 → G, 160m (~5700 km), SFI 100, Kp 2")
    print(f"  SNR at 10:00 UTC (Both ends daylight): {db_daylight:+.1f} dB")
    print(f"  SNR at 04:00 UTC (Both ends dark):     {db_dark:+.1f} dB")
    print(f"\n  Dark vs Day Delta: {delta_db:+.1f} dB")
    print(f"  WSPR Floor: {WSPR_FLOOR_DB} dB")

    # Pass criteria: daylight should be near WSPR floor, dark should be workable
    if db_daylight < -24.0 and db_dark > -20.0:
        print(f"\n  PASS: 160m shows proper mutual darkness dependency")
        return True
    elif delta_db >= 6.0:
        print(f"\n  PASS: 160m shows >= 6 dB day/night difference (acceptable)")
        return True
    else:
        print(f"\n  FAIL: 160m not showing expected mutual darkness behavior")
        print(f"  Expected: Daylight << Dark (need >= 6 dB difference)")
        return False


def test_tst904_80m_mutual_darkness(model, device):
    """TST-904: 80m Mutual Darkness — DX requires both ends dark."""
    print("\n" + "=" * 60)
    print("TST-904: 80m Mutual Darkness Requirement")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["80m"]
    band_id = 103

    # Both ends daylight
    snr_daylight = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                           freq_hz, sfi=100, kp=2, hour_utc=14)
    db_daylight = sigma_to_db(snr_daylight, band_id)

    # Both ends dark
    snr_dark = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                       freq_hz, sfi=100, kp=2, hour_utc=4)
    db_dark = sigma_to_db(snr_dark, band_id)

    delta_db = db_dark - db_daylight

    print(f"\n  Path: W3 → G, 80m (~5700 km), SFI 100, Kp 2")
    print(f"  SNR at 14:00 UTC (Both ends daylight): {db_daylight:+.1f} dB")
    print(f"  SNR at 04:00 UTC (Both ends dark):     {db_dark:+.1f} dB")
    print(f"\n  Dark vs Day Delta: {delta_db:+.1f} dB")

    # Pass criteria: >= 5 dB improvement at night
    if delta_db >= 5.0:
        print(f"\n  PASS: 80m shows >= 5 dB day/night difference")
        return True
    else:
        print(f"\n  FAIL: 80m day/night difference < 5 dB ({delta_db:+.1f})")
        return False


def test_tst905_band_ordering_day(model, device):
    """TST-905: Band Ordering Day — High bands beat low bands at midday."""
    print("\n" + "=" * 60)
    print("TST-905: Band Ordering (Daytime)")
    print("=" * 60)

    # Midday scenario
    hour_utc = 14
    sfi, kp = 150, 2

    snr_10m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ["10m"], sfi, kp, hour_utc)
    snr_15m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ["15m"], sfi, kp, hour_utc)
    snr_20m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ["20m"], sfi, kp, hour_utc)
    snr_40m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ["40m"], sfi, kp, hour_utc)
    snr_80m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ["80m"], sfi, kp, hour_utc)
    snr_160m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                       BAND_FREQ_HZ["160m"], sfi, kp, hour_utc)

    print(f"\n  Path: W3 → G, SFI 150, Kp 2, 14:00 UTC (Daytime)")
    print(f"\n  High Bands:")
    print(f"    10m: {snr_10m:+.3f} sigma ({sigma_to_db(snr_10m, 111):+.1f} dB)")
    print(f"    15m: {snr_15m:+.3f} sigma ({sigma_to_db(snr_15m, 109):+.1f} dB)")
    print(f"    20m: {snr_20m:+.3f} sigma ({sigma_to_db(snr_20m, 107):+.1f} dB)")
    print(f"\n  Low Bands:")
    print(f"    40m: {snr_40m:+.3f} sigma ({sigma_to_db(snr_40m, 105):+.1f} dB)")
    print(f"    80m: {snr_80m:+.3f} sigma ({sigma_to_db(snr_80m, 103):+.1f} dB)")
    print(f"   160m: {snr_160m:+.3f} sigma ({sigma_to_db(snr_160m, 102):+.1f} dB)")

    # Pass criteria: average high bands > average low bands
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
        print(f"\n  FAIL: Low bands better than high bands at midday (wrong)")
        return False


def test_tst906_band_ordering_night(model, device):
    """TST-906: Band Ordering Night — Low bands beat high bands at midnight."""
    print("\n" + "=" * 60)
    print("TST-906: Band Ordering (Nighttime)")
    print("=" * 60)

    # Midnight scenario
    hour_utc = 4  # Both ends dark
    sfi, kp = 100, 2

    snr_10m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ["10m"], sfi, kp, hour_utc)
    snr_15m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ["15m"], sfi, kp, hour_utc)
    snr_20m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ["20m"], sfi, kp, hour_utc)
    snr_40m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ["40m"], sfi, kp, hour_utc)
    snr_80m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ["80m"], sfi, kp, hour_utc)
    snr_160m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                       BAND_FREQ_HZ["160m"], sfi, kp, hour_utc)

    print(f"\n  Path: W3 → G, SFI 100, Kp 2, 04:00 UTC (Nighttime)")
    print(f"\n  High Bands:")
    print(f"    10m: {snr_10m:+.3f} sigma ({sigma_to_db(snr_10m, 111):+.1f} dB)")
    print(f"    15m: {snr_15m:+.3f} sigma ({sigma_to_db(snr_15m, 109):+.1f} dB)")
    print(f"    20m: {snr_20m:+.3f} sigma ({sigma_to_db(snr_20m, 107):+.1f} dB)")
    print(f"\n  Low Bands:")
    print(f"    40m: {snr_40m:+.3f} sigma ({sigma_to_db(snr_40m, 105):+.1f} dB)")
    print(f"    80m: {snr_80m:+.3f} sigma ({sigma_to_db(snr_80m, 103):+.1f} dB)")
    print(f"   160m: {snr_160m:+.3f} sigma ({sigma_to_db(snr_160m, 102):+.1f} dB)")

    # Pass criteria: 40m/80m > 10m/15m at night
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
        print(f"\n  FAIL: High bands still better than low bands at night (wrong)")
        return False


def test_tst907_time_sensitivity(model, device):
    """TST-907: Time Sensitivity — Same path >= 10 dB peak vs off-peak."""
    print("\n" + "=" * 60)
    print("TST-907: Time Sensitivity (Peak vs Off-Peak)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["20m"]
    band_id = 107
    sfi, kp = 150, 2

    # Scan all hours
    snr_by_hour = {}
    for hour in range(24):
        snr = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      freq_hz, sfi, kp, hour)
        snr_by_hour[hour] = snr

    peak_hour = max(snr_by_hour, key=snr_by_hour.get)
    trough_hour = min(snr_by_hour, key=snr_by_hour.get)
    peak_snr = snr_by_hour[peak_hour]
    trough_snr = snr_by_hour[trough_hour]
    delta_sigma = peak_snr - trough_snr
    delta_db = delta_sigma * SIGMA_TO_DB

    print(f"\n  Path: W3 → G, 20m, SFI 150, Kp 2")
    print(f"\n  24-Hour Scan:")
    for h in [0, 4, 8, 12, 16, 20]:
        snr = snr_by_hour[h]
        print(f"    {h:02d}:00 UTC: {snr:+.3f} sigma ({sigma_to_db(snr, band_id):+.1f} dB)")

    print(f"\n  Peak:   {peak_hour:02d}:00 UTC = {peak_snr:+.3f} sigma ({sigma_to_db(peak_snr, band_id):+.1f} dB)")
    print(f"  Trough: {trough_hour:02d}:00 UTC = {trough_snr:+.3f} sigma ({sigma_to_db(trough_snr, band_id):+.1f} dB)")
    print(f"\n  Dynamic Range: {delta_db:+.1f} dB")

    # Pass criteria: >= 10 dB difference peak to trough
    if delta_db >= 10.0:
        print(f"\n  PASS: >= 10 dB time sensitivity")
        return True
    elif delta_db >= 6.0:
        print(f"\n  MARGINAL: 6-10 dB time sensitivity (needs improvement)")
        return False
    else:
        print(f"\n  FAIL: < 6 dB time sensitivity (model not learning time)")
        return False


def test_tst908_10m_peak_hour(model, device):
    """TST-908: 10m Peak Hour — Should peak near local noon."""
    print("\n" + "=" * 60)
    print("TST-908: 10m Peak Hour (Should be near Noon)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["10m"]
    sfi, kp = 180, 2  # Good conditions for 10m

    # For W3 → G, midpoint is roughly -40 lon, so local noon ~ 12-14 UTC
    snr_by_hour = {}
    for hour in range(24):
        snr = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      freq_hz, sfi, kp, hour)
        snr_by_hour[hour] = snr

    peak_hour = max(snr_by_hour, key=snr_by_hour.get)

    print(f"\n  Path: W3 → G, 10m, SFI 180, Kp 2")
    print(f"\n  Peak Hour: {peak_hour:02d}:00 UTC")
    print(f"  Expected: 10-16 UTC (local noon for transatlantic)")

    # Pass criteria: peak between 10-16 UTC (daylight hours for transatlantic)
    if 10 <= peak_hour <= 16:
        print(f"\n  PASS: 10m peaks during daylight hours")
        return True
    else:
        print(f"\n  FAIL: 10m peak at {peak_hour:02d}:00 UTC (outside expected range)")
        return False


def test_tst909_160m_peak_hour(model, device):
    """TST-909: 160m Peak Hour — Should peak near local midnight."""
    print("\n" + "=" * 60)
    print("TST-909: 160m Peak Hour (Should be near Midnight)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["160m"]
    sfi, kp = 80, 2  # Solar conditions less relevant for 160m

    snr_by_hour = {}
    for hour in range(24):
        snr = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      freq_hz, sfi, kp, hour)
        snr_by_hour[hour] = snr

    peak_hour = max(snr_by_hour, key=snr_by_hour.get)

    print(f"\n  Path: W3 → G, 160m, SFI 80, Kp 2")
    print(f"\n  Peak Hour: {peak_hour:02d}:00 UTC")
    print(f"  Expected: 00-06 or 22-24 UTC (mutual darkness)")

    # Pass criteria: peak during night hours (0-6 or 22-24 UTC)
    if peak_hour <= 6 or peak_hour >= 22:
        print(f"\n  PASS: 160m peaks during night hours")
        return True
    else:
        print(f"\n  FAIL: 160m peak at {peak_hour:02d}:00 UTC (should be night)")
        return False


def test_tst910_40m_gray_line(model, device):
    """TST-910: 40m Gray Line Enhancement — Twilight boost for E-W paths."""
    print("\n" + "=" * 60)
    print("TST-910: 40m Gray Line Enhancement")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["40m"]
    band_id = 105
    sfi, kp = 120, 2

    # For transatlantic E-W path, gray line is around 18-20 UTC (sunset on US side)
    snr_noon = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                       freq_hz, sfi, kp, hour_utc=14)
    snr_twilight = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                           freq_hz, sfi, kp, hour_utc=0)  # Sunrise gray line
    snr_night = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                        freq_hz, sfi, kp, hour_utc=4)

    print(f"\n  Path: W3 → G, 40m, SFI 120, Kp 2")
    print(f"\n  SNR at 14:00 UTC (Day):     {sigma_to_db(snr_noon, band_id):+.1f} dB")
    print(f"  SNR at 00:00 UTC (Twilight): {sigma_to_db(snr_twilight, band_id):+.1f} dB")
    print(f"  SNR at 04:00 UTC (Night):   {sigma_to_db(snr_night, band_id):+.1f} dB")

    # 40m should show good propagation at twilight
    twilight_boost = snr_twilight - snr_noon
    twilight_db = twilight_boost * SIGMA_TO_DB

    print(f"\n  Twilight vs Noon: {twilight_db:+.1f} dB")

    # Pass criteria: twilight should be at least as good as noon
    if twilight_boost >= 0:
        print(f"\n  PASS: 40m shows gray line enhancement or stability")
        return True
    elif twilight_boost >= -0.2:
        print(f"\n  PASS: 40m within tolerance at twilight")
        return True
    else:
        print(f"\n  FAIL: 40m shows unexpected twilight degradation")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  IONIS V21 — TST-900 Band×Time Discrimination Tests")
    print("=" * 60)
    print("\n  Testing V20 model to establish baseline.")
    print("  These tests document issues discovered on ham-stats.com.")
    print("  V21 should pass these after time sensitivity improvements.")

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
    print(f"  RMSE: {metadata.get('val_rmse', 0):.4f} sigma")
    print(f"  Pearson: {metadata.get('val_pearson', 0):+.4f}")

    # Run tests
    results = []
    results.append(("TST-901", "10m Band Closure", test_tst901_10m_band_closure(model, DEVICE)))
    results.append(("TST-902", "15m Band Closure", test_tst902_15m_band_closure(model, DEVICE)))
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
        print(f"  {test_id}: {name:<20s}  {status}")

    print()
    print(f"  Results: {passed}/{total} passed")

    if passed == total:
        print("\n  ALL TST-900 TESTS PASSED")
        print("  Model shows correct band×time discrimination.")
        return 0
    else:
        print(f"\n  {total - passed} TEST(S) FAILED")
        print("  V21 improvements needed for band×time sensitivity.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
