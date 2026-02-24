#!/usr/bin/env python3
"""
test_tst900_band_time.py — IONIS V24 Band×Time Discrimination Tests

TST-900 Group: Verify band×time-of-day behavior matches ionospheric reality.

V24 Architecture Change:
    Sun sidecar REMOVED — model now has only storm sidecar.
    Hypothesis: Trunk freed from +0.48σ forced error → better band×time physics.

Tests:
  TST-901: Band Closure — 10m should show >= 10 dB day/night delta in winter
  TST-901b: Summer Twilight — 10m shows weak delta in summer (correct physics)
  TST-902: Band Closure — 15m should show >= 8 dB day/night delta in winter
  TST-903: Mutual Darkness — 160m DX requires both ends dark (>= 5 dB)
  TST-904: Mutual Darkness — 80m DX requires both ends dark (>= 4 dB)
  TST-905: Band Ordering Day — High bands beat low bands at midday
  TST-906: Band Ordering Night — Low bands beat high bands at night
  TST-907: Time Sensitivity — 20m shows >= 6 dB day/night difference
  TST-908: Peak Hour — 10m peaks during daylight hours
  TST-909: Peak Hour — 160m peaks during mutual darkness
  TST-910: Gray Line — 40m shows enhancement at twilight

Pass criteria: >= 10/11 with all mandatory tests passing (TST-901, 903, 904, 905, 906)
"""

import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors

# ── Path Setup ───────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V24_DIR = os.path.dirname(SCRIPT_DIR)
VERSIONS_DIR = os.path.dirname(V24_DIR)
sys.path.insert(0, V24_DIR)
sys.path.insert(0, os.path.join(VERSIONS_DIR, "common"))

from model import get_device, build_features, BAND_FREQ_HZ, MonotonicMLP
from train_v24 import IonisGateV24

# ── Load Config ──────────────────────────────────────────────────────────────

CONFIG_PATH = os.path.join(V24_DIR, "config_v24.json")
MODEL_PATH = os.path.join(V24_DIR, "ionis_v24.safetensors")

with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

DNN_DIM = CONFIG["model"]["dnn_dim"]
KP_PENALTY_IDX = CONFIG["model"]["kp_penalty_idx"]
SIDECAR_HIDDEN = CONFIG["model"]["sidecar_hidden"]

DEVICE = get_device()

SIGMA_TO_DB = 6.7  # Approximate conversion factor


# ── Prediction Helper ────────────────────────────────────────────────────────

def predict(model, device, tx_lat, tx_lon, rx_lat, rx_lon, freq_hz,
            sfi, kp, hour_utc, month=6, day_of_year=172):
    """Make a prediction for given parameters."""
    features = build_features(
        tx_lat, tx_lon, rx_lat, rx_lon, freq_hz,
        sfi, kp, hour_utc, month,
        day_of_year=day_of_year,
        include_solar_depression=True,  # V24 uses V22-gamma recipe
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


# ── Test Functions ───────────────────────────────────────────────────────────

def test_tst901_10m_band_closure(model, device):
    """TST-901: 10m Band Closure — Should show >= 10 dB day/night delta in winter."""
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

    print(f"\n  Path: W3 -> G, 10m, SFI 150, Kp 2, Winter (Dec 21)")
    print(f"  SNR at 14:00 UTC (Midday):   {db_midday:.1f} dB ({snr_midday:+.3f} sigma)")
    print(f"  SNR at 00:00 UTC (Midnight): {db_midnight:.1f} dB ({snr_midnight:+.3f} sigma)")
    print(f"\n  Day/Night Delta: {delta_db:+.1f} dB")

    if delta_db >= 10.0:
        print(f"\n  PASS: Day/night delta {delta_db:+.1f} dB >= 10 dB threshold")
        return True
    else:
        print(f"\n  FAIL: Day/night delta {delta_db:+.1f} dB < 10 dB threshold")
        return False


def test_tst901b_10m_summer_twilight(model, device):
    """TST-901b: 10m Summer Twilight — Should show weak delta in summer."""
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

    print(f"\n  Path: W3 -> G, 10m, SFI 150, Kp 2, Summer (Jun 21)")
    print(f"  SNR at 14:00 UTC (Midday):   {db_midday:.1f} dB ({snr_midday:+.3f} sigma)")
    print(f"  SNR at 00:00 UTC (Midnight): {db_midnight:.1f} dB ({snr_midnight:+.3f} sigma)")
    print(f"\n  Day/Night Delta: {delta_db:+.1f} dB")

    if abs(delta_db) < 5.0:
        print(f"\n  PASS: 10m shows weak day/night delta — correct summer physics")
        return True
    else:
        print(f"\n  FAIL: Day/night delta {delta_db:+.1f} dB too large for summer")
        return False


def test_tst902_15m_band_closure(model, device):
    """TST-902: 15m Band Closure — Should show >= 8 dB day/night delta in winter."""
    print("\n" + "=" * 60)
    print("TST-902: 15m Band Closure (Winter, Midnight UTC)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["15m"]
    band_id = 109

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

    print(f"\n  Path: W3 -> G, 15m, SFI 150, Kp 2, Winter (Dec 21)")
    print(f"  SNR at 14:00 UTC (Midday):   {db_midday:.1f} dB ({snr_midday:+.3f} sigma)")
    print(f"  SNR at 00:00 UTC (Midnight): {db_midnight:.1f} dB ({snr_midnight:+.3f} sigma)")
    print(f"\n  Day/Night Delta: {delta_db:+.1f} dB")

    if delta_db >= 8.0:
        print(f"\n  PASS: Day/night delta {delta_db:+.1f} dB >= 8 dB threshold")
        return True
    else:
        print(f"\n  FAIL: Day/night delta {delta_db:+.1f} dB < 8 dB threshold")
        return False


def test_tst903_160m_mutual_darkness(model, device):
    """TST-903: 160m Mutual Darkness — Should show >= 5 dB day/night delta."""
    print("\n" + "=" * 60)
    print("TST-903: 160m Mutual Darkness (Winter)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["160m"]
    band_id = 102

    winter_doy = 355
    winter_month = 12

    # 14:00 UTC = daylight for both W3 and G
    snr_day = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      freq_hz, sfi=100, kp=2, hour_utc=14,
                      month=winter_month, day_of_year=winter_doy)
    db_day = sigma_to_db(snr_day, band_id)

    # 04:00 UTC = both ends dark (mutual darkness)
    snr_dark = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                       freq_hz, sfi=100, kp=2, hour_utc=4,
                       month=winter_month, day_of_year=winter_doy)
    db_dark = sigma_to_db(snr_dark, band_id)

    delta_db = db_dark - db_day

    print(f"\n  Path: W3 -> G, 160m (~5700 km), SFI 100, Kp 2, Winter")
    print(f"  SNR at 14:00 UTC (Both ends daylight): {db_day:.1f} dB")
    print(f"  SNR at 04:00 UTC (Both ends dark):     {db_dark:.1f} dB")
    print(f"\n  Dark vs Day Delta: {delta_db:+.1f} dB")

    if delta_db >= 5.0:
        print(f"\n  PASS: 160m shows {delta_db:+.1f} dB improvement in mutual darkness")
        return True
    else:
        print(f"\n  FAIL: 160m day/night difference < 5 dB ({delta_db:+.1f})")
        return False


def test_tst904_80m_mutual_darkness(model, device):
    """TST-904: 80m Mutual Darkness — Should show >= 4 dB day/night delta."""
    print("\n" + "=" * 60)
    print("TST-904: 80m Mutual Darkness (Winter)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["80m"]
    band_id = 103

    winter_doy = 355
    winter_month = 12

    snr_day = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      freq_hz, sfi=100, kp=2, hour_utc=14,
                      month=winter_month, day_of_year=winter_doy)
    db_day = sigma_to_db(snr_day, band_id)

    snr_dark = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                       freq_hz, sfi=100, kp=2, hour_utc=4,
                       month=winter_month, day_of_year=winter_doy)
    db_dark = sigma_to_db(snr_dark, band_id)

    delta_db = db_dark - db_day

    print(f"\n  Path: W3 -> G, 80m (~5700 km), SFI 100, Kp 2, Winter")
    print(f"  SNR at 14:00 UTC (Both ends daylight): {db_day:.1f} dB")
    print(f"  SNR at 04:00 UTC (Both ends dark):     {db_dark:.1f} dB")
    print(f"\n  Dark vs Day Delta: {delta_db:+.1f} dB")

    if delta_db >= 4.0:
        print(f"\n  PASS: 80m shows {delta_db:+.1f} dB improvement in mutual darkness")
        return True
    else:
        print(f"\n  FAIL: 80m day/night difference < 4 dB ({delta_db:+.1f})")
        return False


def test_tst905_band_order_day(model, device):
    """TST-905: Band Ordering Day — High bands should beat low bands at midday."""
    print("\n" + "=" * 60)
    print("TST-905: Band Ordering (Daytime)")
    print("=" * 60)

    summer_doy = 172
    summer_month = 6

    high_bands = ["10m", "15m", "20m"]
    low_bands = ["80m", "160m"]
    high_ids = [111, 109, 107]
    low_ids = [103, 102]

    high_snrs = []
    low_snrs = []

    print(f"\n  Path: W3 -> G, SFI 150, Kp 2, 14:00 UTC (Daytime)")
    print(f"\n  High Bands:")
    for band, band_id in zip(high_bands, high_ids):
        snr = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ[band], sfi=150, kp=2, hour_utc=14,
                      month=summer_month, day_of_year=summer_doy)
        db = sigma_to_db(snr, band_id)
        high_snrs.append(snr)
        print(f"    {band}: {snr:+.3f} sigma ({db:.1f} dB)")

    print(f"\n  Low Bands:")
    for band, band_id in zip(low_bands, low_ids):
        snr = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ[band], sfi=150, kp=2, hour_utc=14,
                      month=summer_month, day_of_year=summer_doy)
        db = sigma_to_db(snr, band_id)
        low_snrs.append(snr)
        print(f"   {band}: {snr:+.3f} sigma ({db:.1f} dB)")

    high_avg = np.mean(high_snrs)
    low_avg = np.mean(low_snrs)
    delta = high_avg - low_avg

    print(f"\n  High band avg: {high_avg:+.3f} sigma")
    print(f"  Low band avg:  {low_avg:+.3f} sigma")
    print(f"  Delta: {delta:+.3f} sigma ({delta * SIGMA_TO_DB:+.1f} dB)")

    if delta > 0:
        print(f"\n  PASS: High bands beat low bands at midday")
        return True
    else:
        print(f"\n  FAIL: Low bands better than high bands at midday (wrong)")
        return False


def test_tst906_band_order_night(model, device):
    """TST-906: Band Ordering Night — Low bands should beat high bands at night."""
    print("\n" + "=" * 60)
    print("TST-906: Band Ordering (Winter Nighttime)")
    print("=" * 60)

    winter_doy = 355
    winter_month = 12

    high_bands = ["10m", "15m"]
    low_bands = ["40m", "80m"]
    high_ids = [111, 109]
    low_ids = [105, 103]

    high_snrs = []
    low_snrs = []

    print(f"\n  Path: W3 -> G, SFI 100, Kp 2, 04:00 UTC, Winter")
    print(f"\n  High Bands:")
    for band, band_id in zip(high_bands, high_ids):
        snr = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ[band], sfi=100, kp=2, hour_utc=4,
                      month=winter_month, day_of_year=winter_doy)
        db = sigma_to_db(snr, band_id)
        high_snrs.append(snr)
        print(f"    {band}: {snr:+.3f} sigma ({db:.1f} dB)")

    print(f"\n  Low Bands:")
    for band, band_id in zip(low_bands, low_ids):
        snr = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ[band], sfi=100, kp=2, hour_utc=4,
                      month=winter_month, day_of_year=winter_doy)
        db = sigma_to_db(snr, band_id)
        low_snrs.append(snr)
        print(f"    {band}: {snr:+.3f} sigma ({db:.1f} dB)")

    high_avg = np.mean(high_snrs)
    low_avg = np.mean(low_snrs)
    delta = low_avg - high_avg

    print(f"\n  High band avg (10m/15m): {high_avg:+.3f} sigma")
    print(f"  Low band avg (40m/80m):  {low_avg:+.3f} sigma")
    print(f"  Delta (low - high): {delta:+.3f} sigma ({delta * SIGMA_TO_DB:+.1f} dB)")

    if delta > 0:
        print(f"\n  PASS: Low bands better than high bands at night")
        return True
    else:
        print(f"\n  FAIL: High bands still better than low bands at night (wrong)")
        return False


def test_tst907_time_sensitivity(model, device):
    """TST-907: Time Sensitivity — 20m should show >= 6 dB day/night delta."""
    print("\n" + "=" * 60)
    print("TST-907: Time Sensitivity (Winter, Peak vs Off-Peak)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["20m"]
    band_id = 107

    winter_doy = 355
    winter_month = 12

    print(f"\n  Path: W3 -> G, 20m, SFI 150, Kp 2, Winter")
    print(f"\n  24-Hour Scan:")

    hours = [0, 4, 8, 12, 16, 20]
    snrs = []
    for h in hours:
        snr = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      freq_hz, sfi=150, kp=2, hour_utc=h,
                      month=winter_month, day_of_year=winter_doy)
        db = sigma_to_db(snr, band_id)
        snrs.append((h, snr, db))
        print(f"    {h:02d}:00 UTC: {snr:+.3f} sigma ({db:.1f} dB)")

    # Find peak and trough
    all_snrs = []
    for h in range(24):
        snr = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      freq_hz, sfi=150, kp=2, hour_utc=h,
                      month=winter_month, day_of_year=winter_doy)
        all_snrs.append((h, snr))

    peak_h, peak_snr = max(all_snrs, key=lambda x: x[1])
    trough_h, trough_snr = min(all_snrs, key=lambda x: x[1])

    peak_db = sigma_to_db(peak_snr, band_id)
    trough_db = sigma_to_db(trough_snr, band_id)
    dynamic_range = peak_db - trough_db

    print(f"\n  Peak:   {peak_h:02d}:00 UTC = {peak_snr:+.3f} sigma")
    print(f"  Trough: {trough_h:02d}:00 UTC = {trough_snr:+.3f} sigma")
    print(f"\n  Dynamic Range: {dynamic_range:+.1f} dB")

    if dynamic_range >= 6.0:
        print(f"\n  PASS: >= 6 dB time sensitivity")
        return True
    else:
        print(f"\n  FAIL: Dynamic range {dynamic_range:.1f} dB < 6 dB")
        return False


def test_tst908_10m_peak_hour(model, device):
    """TST-908: 10m Peak Hour — Should peak during daylight hours (10-16 UTC)."""
    print("\n" + "=" * 60)
    print("TST-908: 10m Peak Hour (Winter, Should be near Noon)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["10m"]
    band_id = 111

    winter_doy = 355
    winter_month = 12

    snrs = []
    for h in range(24):
        snr = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      freq_hz, sfi=180, kp=2, hour_utc=h,
                      month=winter_month, day_of_year=winter_doy)
        snrs.append((h, snr))

    peak_h, _ = max(snrs, key=lambda x: x[1])

    print(f"\n  Path: W3 -> G, 10m, SFI 180, Kp 2, Winter")
    print(f"\n  Peak Hour: {peak_h:02d}:00 UTC")
    print(f"  Expected: 10-16 UTC (local noon for transatlantic)")

    if 10 <= peak_h <= 16:
        print(f"\n  PASS: 10m peaks during daylight hours")
        return True
    else:
        print(f"\n  FAIL: 10m peak at {peak_h:02d}:00 UTC (should be daytime)")
        return False


def test_tst909_160m_peak_hour(model, device):
    """TST-909: 160m Peak Hour — Should peak during mutual darkness (00-06 or 22-24 UTC)."""
    print("\n" + "=" * 60)
    print("TST-909: 160m Peak Hour (Winter, Should be near Midnight)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["160m"]
    band_id = 102

    winter_doy = 355
    winter_month = 12

    snrs = []
    for h in range(24):
        snr = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      freq_hz, sfi=80, kp=2, hour_utc=h,
                      month=winter_month, day_of_year=winter_doy)
        snrs.append((h, snr))

    peak_h, _ = max(snrs, key=lambda x: x[1])

    print(f"\n  Path: W3 -> G, 160m, SFI 80, Kp 2, Winter")
    print(f"\n  Peak Hour: {peak_h:02d}:00 UTC")
    print(f"  Expected: 00-06 or 22-24 UTC (mutual darkness)")

    if peak_h <= 6 or peak_h >= 22:
        print(f"\n  PASS: 160m peaks during mutual darkness")
        return True
    else:
        print(f"\n  FAIL: 160m peak at {peak_h:02d}:00 UTC (should be night)")
        return False


def test_tst910_40m_gray_line(model, device):
    """TST-910: 40m Gray Line — Should show enhancement at twilight vs midday."""
    print("\n" + "=" * 60)
    print("TST-910: 40m Gray Line Enhancement")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["40m"]
    band_id = 105

    winter_doy = 355
    winter_month = 12

    # Midday (14:00 UTC)
    snr_noon = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                       freq_hz, sfi=120, kp=2, hour_utc=14,
                       month=winter_month, day_of_year=winter_doy)
    db_noon = sigma_to_db(snr_noon, band_id)

    # Twilight / mutual darkness (00:00 UTC)
    snr_twilight = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                           freq_hz, sfi=120, kp=2, hour_utc=0,
                           month=winter_month, day_of_year=winter_doy)
    db_twilight = sigma_to_db(snr_twilight, band_id)

    delta_db = db_twilight - db_noon

    print(f"\n  Path: W3 -> G, 40m, SFI 120, Kp 2")
    print(f"\n  SNR at 14:00 UTC (Day):     {db_noon:.1f} dB")
    print(f"  SNR at 00:00 UTC (Twilight): {db_twilight:.1f} dB")
    print(f"\n  Twilight vs Noon: {delta_db:+.1f} dB")

    # 40m should be better at twilight/night than midday (when D-layer absorbs)
    if delta_db >= 0:
        print(f"\n  PASS: 40m shows gray line enhancement or stability")
        return True
    else:
        print(f"\n  FAIL: 40m worse at twilight than noon (wrong physics)")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  IONIS V24 — TST-900 Band x Time Discrimination Tests")
    print("=" * 60)
    print()
    print("  Model: V24 (sun sidecar REMOVED)")
    print("  Architecture: IonisGateV24 (storm sidecar only)")
    print("  Recipe: V22-gamma (no IRI features)")
    print("  Hypothesis: Trunk freed from +0.48σ forced error")
    print()

    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Checkpoint not found: {MODEL_PATH}")
        print("Run train_v24.py first to generate the checkpoint.")
        sys.exit(1)

    print(f"Loading {MODEL_PATH}...")
    state_dict = load_safetensors(MODEL_PATH, device=str(DEVICE))

    model = IonisGateV24(
        dnn_dim=DNN_DIM,
        sidecar_hidden=SIDECAR_HIDDEN,
        kp_penalty_idx=KP_PENALTY_IDX,
        gate_init_bias=CONFIG["model"].get("gate_init_bias"),
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Device: {DEVICE}")
    print(f"  Parameters: {n_params:,}")

    # Load metadata if available
    meta_path = MODEL_PATH.replace('.safetensors', '_meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  RMSE: {meta.get('val_rmse', 0):.4f} sigma")
        print(f"  Pearson: {meta.get('val_pearson', 0):+.4f}")
        print(f"  Storm Cost: {meta.get('storm_cost', 0):+.3f} sigma")

    # Run tests
    results = {}

    results["TST-901"] = test_tst901_10m_band_closure(model, DEVICE)
    results["TST-901b"] = test_tst901b_10m_summer_twilight(model, DEVICE)
    results["TST-902"] = test_tst902_15m_band_closure(model, DEVICE)
    results["TST-903"] = test_tst903_160m_mutual_darkness(model, DEVICE)
    results["TST-904"] = test_tst904_80m_mutual_darkness(model, DEVICE)
    results["TST-905"] = test_tst905_band_order_day(model, DEVICE)
    results["TST-906"] = test_tst906_band_order_night(model, DEVICE)
    results["TST-907"] = test_tst907_time_sensitivity(model, DEVICE)
    results["TST-908"] = test_tst908_10m_peak_hour(model, DEVICE)
    results["TST-909"] = test_tst909_160m_peak_hour(model, DEVICE)
    results["TST-910"] = test_tst910_40m_gray_line(model, DEVICE)

    # Summary
    print()
    print("=" * 60)
    print("  TST-900 SUMMARY: Band x Time Discrimination")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for test_id, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test_id}: {status}")

    print()
    print(f"  Results: {passed}/{total} passed")
    print()
    print(f"  V22-gamma baseline: 9/11 TST-900")
    print(f"  V24 target: >= 10/11")
    print()

    # Mandatory tests check
    mandatory = ["TST-901", "TST-903", "TST-904", "TST-905", "TST-906"]
    mandatory_passed = all(results[t] for t in mandatory)

    if passed >= 10 and mandatory_passed:
        print("  V24 GATE: PASS — Physics improvement confirmed")
    elif passed >= 9:
        print("  V24: MAINTAINED — No regression from V22-gamma")
    else:
        print("  V24: REGRESSION — Escape valve needed")
        print("  Action: Consider lower clamp floor [0.1, 2.0]")

    print()

    return 0 if passed >= 10 else 1


if __name__ == "__main__":
    sys.exit(main())
