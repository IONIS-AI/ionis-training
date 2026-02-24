#!/usr/bin/env python3
"""
test_tst900_band_time.py — IONIS AUDIT-03 Band×Time Discrimination Tests

TST-900 Group: Verify band×time-of-day behavior matches ionospheric reality.

AUDIT-03: V23 recipe with 35-bucket IRI atlas (5-unit SFI steps)
  - dnn_dim=18 (V22's 15 + 3 IRI features)
  - IRI features: foF2_freq_ratio, foE_mid, hmF2_mid
  - 35-bucket atlas: SFI 70, 75, 80...240 (5-unit steps)

Comparison target: V23-alpha (4/11 TST-900)
"""

import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors

# ── Path Setup ───────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIT03_DIR = os.path.dirname(SCRIPT_DIR)
VERSIONS_DIR = os.path.dirname(AUDIT03_DIR)
TRAINING_DIR = os.path.dirname(VERSIONS_DIR)
sys.path.insert(0, os.path.join(VERSIONS_DIR, "common"))

from model import (
    IonisGate, get_device, BAND_FREQ_HZ,
    haversine_km, azimuth_deg, vertex_lat_deg, solar_elevation_deg,
    latlon_to_grid4,
)

# ── Load Config ──────────────────────────────────────────────────────────────

AUDIT03_CONFIG = os.path.join(AUDIT03_DIR, "config_audit03.json")
AUDIT03_CKPT = os.path.join(AUDIT03_DIR, "ionis_audit03.safetensors")

with open(AUDIT03_CONFIG) as f:
    CONFIG = json.load(f)

MODEL_PATH = AUDIT03_CKPT
MODEL_VERSION = "audit-03"

DNN_DIM = CONFIG["model"]["dnn_dim"]
SFI_IDX = CONFIG["model"]["sfi_idx"]
KP_PENALTY_IDX = CONFIG["model"]["kp_penalty_idx"]
SIDECAR_HIDDEN = CONFIG["model"]["sidecar_hidden"]

DEVICE = get_device()

SIGMA_TO_DB = 6.7  # Approximate conversion factor

# Thresholds
FT8_THRESHOLD_DB = -21.0  # FT8 decode limit
WSPR_FLOOR_DB = -28.0  # WSPR absolute floor


# ── 35-Bucket IRI Atlas Loading ──────────────────────────────────────────────

def sfi_bucket_to_index_35(sfi):
    """Convert SFI to 35-bucket index (5-unit steps: 70, 75, 80...240)."""
    return int(np.clip((sfi - 70) / 5, 0, 34))


def load_iri_atlas():
    """Load the 35-bucket IRI-2020 atlas (CoW-safe .npz)."""
    iri_path = os.path.join(TRAINING_DIR, CONFIG["iri_atlas"]["path"])
    print(f"  Loading 35-bucket IRI atlas from {iri_path}...")
    npz = np.load(iri_path, allow_pickle=True)
    data = npz["data"]                      # (31692, 24, 12, 35, 3) float32
    grid_index = npz["grid_index"].item()   # dict: grid_4 -> int
    sfi_buckets = npz["sfi_buckets"]        # [70, 75, 80, ..., 240]
    print(f"  IRI atlas: {data.shape}, {data.nbytes / 1e9:.2f} GB")
    print(f"  SFI buckets: {len(sfi_buckets)} (5-unit steps)")
    return data, grid_index, sfi_buckets


# Global IRI atlas (loaded once)
IRI_DATA, IRI_GRID_INDEX, IRI_SFI_BUCKETS = None, None, None


def get_iri_params(lat, lon, hour_utc, month, sfi):
    """Look up IRI parameters for a single point using 35-bucket indexing."""
    global IRI_DATA, IRI_GRID_INDEX

    if IRI_DATA is None:
        raise RuntimeError("IRI atlas not loaded")

    grid4 = latlon_to_grid4(lat, lon)
    grid_idx = IRI_GRID_INDEX.get(grid4, -1)

    if grid_idx == -1:
        # Unknown grid — return neutral values
        return 1.0, 3.0, 300.0  # foF2/freq=1 (neutral), foE=3 MHz, hmF2=300 km

    hour_idx = int(hour_utc) % 24
    month_idx = (int(month) - 1) % 12
    sfi_idx = sfi_bucket_to_index_35(sfi)  # 35-bucket indexing

    # IRI_DATA shape: (grids, hours, months, sfi_buckets, 3)
    # 3 params: [foF2, hmF2, foE]
    iri_vec = IRI_DATA[grid_idx, hour_idx, month_idx, sfi_idx, :]
    foF2 = float(iri_vec[0])
    hmF2 = float(iri_vec[1])
    foE = float(iri_vec[2])

    return foF2, foE, hmF2


# ── Feature Builder ──────────────────────────────────────────────────────────

def build_features(tx_lat, tx_lon, rx_lat, rx_lon, freq_hz, sfi, kp,
                   hour_utc, month, day_of_year):
    """Build an AUDIT-03 feature vector (18 DNN + SFI + Kp = 20 total)."""
    distance_km = haversine_km(tx_lat, tx_lon, rx_lat, rx_lon)
    az = azimuth_deg(tx_lat, tx_lon, rx_lat, rx_lon)
    midpoint_lat = (tx_lat + rx_lat) / 2.0
    midpoint_lon = (tx_lon + rx_lon) / 2.0

    # V22 features (indices 0-14)
    v_lat = vertex_lat_deg(tx_lat, tx_lon, rx_lat, rx_lon)
    tx_solar = solar_elevation_deg(tx_lat, tx_lon, hour_utc, day_of_year)
    rx_solar = solar_elevation_deg(rx_lat, rx_lon, hour_utc, day_of_year)
    tx_solar_norm = tx_solar / 90.0
    rx_solar_norm = rx_solar / 90.0

    freq_mhz = freq_hz / 1e6
    if freq_mhz >= 10.0:
        freq_centered = (freq_mhz - 10.0) / 18.0
    else:
        freq_centered = (freq_mhz - 10.0) / 8.2

    features = [
        distance_km / 20000.0,                              # 0: distance
        np.log10(freq_hz) / 8.0,                            # 1: freq_log
        np.sin(2.0 * np.pi * hour_utc / 24.0),              # 2: hour_sin
        np.cos(2.0 * np.pi * hour_utc / 24.0),              # 3: hour_cos
        np.sin(2.0 * np.pi * az / 360.0),                   # 4: az_sin
        np.cos(2.0 * np.pi * az / 360.0),                   # 5: az_cos
        abs(tx_lat - rx_lat) / 180.0,                       # 6: lat_diff
        midpoint_lat / 90.0,                                # 7: midpoint_lat
        np.sin(2.0 * np.pi * month / 12.0),                 # 8: season_sin
        np.cos(2.0 * np.pi * month / 12.0),                 # 9: season_cos
        v_lat / 90.0,                                       # 10: vertex_lat
        tx_solar_norm,                                      # 11: tx_solar_dep
        rx_solar_norm,                                      # 12: rx_solar_dep
        freq_centered * tx_solar_norm,                      # 13: freq_x_tx_dark
        freq_centered * rx_solar_norm,                      # 14: freq_x_rx_dark
    ]

    # IRI features (indices 15-17) using 35-bucket lookup
    foF2, foE, hmF2 = get_iri_params(midpoint_lat, midpoint_lon, hour_utc, month, sfi)

    features.append(foF2 / freq_mhz)                        # 15: foF2_freq_ratio
    features.append(foE / 10.0)                             # 16: foE_mid (normalized)
    features.append(hmF2 / 500.0)                           # 17: hmF2_mid (normalized)

    # Sidecar inputs (indices 18-19)
    features.append(sfi / 300.0)                            # 18: sfi
    features.append(1.0 - kp / 9.0)                         # 19: kp_penalty

    return np.array(features, dtype=np.float32)


# ── Prediction Helper ────────────────────────────────────────────────────────

def predict(model, device, tx_lat, tx_lon, rx_lat, rx_lon, freq_hz,
            sfi, kp, hour_utc, month=6, day_of_year=172):
    """Make a prediction for given parameters."""
    features = build_features(
        tx_lat, tx_lon, rx_lat, rx_lon, freq_hz,
        sfi, kp, hour_utc, month, day_of_year,
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
    """TST-901: 10m Band Closure — Should drop below FT8 at midnight in WINTER."""
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

    print(f"\n  Path: W3 -> G, 10m, SFI 150, Kp 2, Summer (Jun 21)")
    print(f"  SNR at 14:00 UTC (Midday):   {db_midday:+.1f} dB ({snr_midday:+.3f} sigma)")
    print(f"  SNR at 00:00 UTC (Midnight): {db_midnight:+.1f} dB ({snr_midnight:+.3f} sigma)")
    print(f"\n  Day/Night Delta: {delta_db:+.1f} dB")

    if abs(delta_db) < 5.0:
        print(f"\n  PASS: 10m shows weak day/night delta — correct summer physics")
        return True
    else:
        print(f"\n  FAIL: Day/night delta {delta_db:+.1f} dB too large for summer")
        return False


def test_tst902_15m_band_closure(model, device):
    """TST-902: 15m Band Closure — Should show significant day/night delta in WINTER."""
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
    """TST-903: 160m Mutual Darkness — DX requires both ends dark (WINTER)."""
    print("\n" + "=" * 60)
    print("TST-903: 160m Mutual Darkness (Winter)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["160m"]
    band_id = 102
    winter_doy = 355
    winter_month = 12

    snr_daylight = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                           freq_hz, sfi=100, kp=2, hour_utc=14,
                           month=winter_month, day_of_year=winter_doy)
    db_daylight = sigma_to_db(snr_daylight, band_id)

    snr_dark = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                       freq_hz, sfi=100, kp=2, hour_utc=4,
                       month=winter_month, day_of_year=winter_doy)
    db_dark = sigma_to_db(snr_dark, band_id)

    delta_db = db_dark - db_daylight

    print(f"\n  Path: W3 -> G, 160m (~5700 km), SFI 100, Kp 2, Winter")
    print(f"  SNR at 14:00 UTC (Both ends daylight): {db_daylight:+.1f} dB")
    print(f"  SNR at 04:00 UTC (Both ends dark):     {db_dark:+.1f} dB")
    print(f"\n  Dark vs Day Delta: {delta_db:+.1f} dB")

    if delta_db >= 5.0:
        print(f"\n  PASS: 160m shows {delta_db:+.1f} dB improvement at night")
        return True
    else:
        print(f"\n  FAIL: 160m day/night difference < 5 dB ({delta_db:+.1f})")
        return False


def test_tst904_80m_mutual_darkness(model, device):
    """TST-904: 80m Mutual Darkness — DX requires both ends dark (WINTER)."""
    print("\n" + "=" * 60)
    print("TST-904: 80m Mutual Darkness (Winter)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["80m"]
    band_id = 103
    winter_doy = 355
    winter_month = 12

    snr_daylight = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                           freq_hz, sfi=100, kp=2, hour_utc=14,
                           month=winter_month, day_of_year=winter_doy)
    db_daylight = sigma_to_db(snr_daylight, band_id)

    snr_dark = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                       freq_hz, sfi=100, kp=2, hour_utc=4,
                       month=winter_month, day_of_year=winter_doy)
    db_dark = sigma_to_db(snr_dark, band_id)

    delta_db = db_dark - db_daylight

    print(f"\n  Path: W3 -> G, 80m (~5700 km), SFI 100, Kp 2, Winter")
    print(f"  SNR at 14:00 UTC (Both ends daylight): {db_daylight:+.1f} dB")
    print(f"  SNR at 04:00 UTC (Both ends dark):     {db_dark:+.1f} dB")
    print(f"\n  Dark vs Day Delta: {delta_db:+.1f} dB")

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
        print(f"\n  FAIL: Low bands better than high bands at midday (wrong)")
        return False


def test_tst906_band_ordering_night(model, device):
    """TST-906: Band Ordering Night — Low bands beat high bands at midnight (WINTER)."""
    print("\n" + "=" * 60)
    print("TST-906: Band Ordering (Winter Nighttime)")
    print("=" * 60)

    hour_utc = 4
    sfi, kp = 100, 2
    winter_doy = 355
    winter_month = 12

    snr_10m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ["10m"], sfi, kp, hour_utc,
                      month=winter_month, day_of_year=winter_doy)
    snr_15m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ["15m"], sfi, kp, hour_utc,
                      month=winter_month, day_of_year=winter_doy)
    snr_40m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ["40m"], sfi, kp, hour_utc,
                      month=winter_month, day_of_year=winter_doy)
    snr_80m = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      BAND_FREQ_HZ["80m"], sfi, kp, hour_utc,
                      month=winter_month, day_of_year=winter_doy)

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
        print(f"\n  FAIL: High bands still better than low bands at night (wrong)")
        return False


def test_tst907_time_sensitivity(model, device):
    """TST-907: Time Sensitivity — Same path >= 6 dB peak vs off-peak (WINTER)."""
    print("\n" + "=" * 60)
    print("TST-907: Time Sensitivity (Winter, Peak vs Off-Peak)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["20m"]
    band_id = 107
    sfi, kp = 150, 2
    winter_doy = 355
    winter_month = 12

    snr_by_hour = {}
    for hour in range(24):
        snr = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      freq_hz, sfi, kp, hour,
                      month=winter_month, day_of_year=winter_doy)
        snr_by_hour[hour] = snr

    peak_hour = max(snr_by_hour, key=snr_by_hour.get)
    trough_hour = min(snr_by_hour, key=snr_by_hour.get)
    peak_snr = snr_by_hour[peak_hour]
    trough_snr = snr_by_hour[trough_hour]
    delta_sigma = peak_snr - trough_snr
    delta_db = delta_sigma * SIGMA_TO_DB

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
    """TST-908: 10m Peak Hour — Should peak near local noon (WINTER)."""
    print("\n" + "=" * 60)
    print("TST-908: 10m Peak Hour (Winter, Should be near Noon)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["10m"]
    sfi, kp = 180, 2
    winter_doy = 355
    winter_month = 12

    snr_by_hour = {}
    for hour in range(24):
        snr = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      freq_hz, sfi, kp, hour,
                      month=winter_month, day_of_year=winter_doy)
        snr_by_hour[hour] = snr

    peak_hour = max(snr_by_hour, key=snr_by_hour.get)

    print(f"\n  Path: W3 -> G, 10m, SFI 180, Kp 2, Winter")
    print(f"\n  Peak Hour: {peak_hour:02d}:00 UTC")
    print(f"  Expected: 10-16 UTC (local noon for transatlantic)")

    if 10 <= peak_hour <= 16:
        print(f"\n  PASS: 10m peaks during daylight hours")
        return True
    else:
        print(f"\n  FAIL: 10m peak at {peak_hour:02d}:00 UTC (outside expected range)")
        return False


def test_tst909_160m_peak_hour(model, device):
    """TST-909: 160m Peak Hour — Should peak near local midnight (WINTER)."""
    print("\n" + "=" * 60)
    print("TST-909: 160m Peak Hour (Winter, Should be near Midnight)")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["160m"]
    sfi, kp = 80, 2
    winter_doy = 355
    winter_month = 12

    snr_by_hour = {}
    for hour in range(24):
        snr = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                      freq_hz, sfi, kp, hour,
                      month=winter_month, day_of_year=winter_doy)
        snr_by_hour[hour] = snr

    peak_hour = max(snr_by_hour, key=snr_by_hour.get)

    print(f"\n  Path: W3 -> G, 160m, SFI 80, Kp 2, Winter")
    print(f"\n  Peak Hour: {peak_hour:02d}:00 UTC")
    print(f"  Expected: 00-06 or 22-24 UTC (mutual darkness)")

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

    snr_noon = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                       freq_hz, sfi, kp, hour_utc=14)
    snr_twilight = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                           freq_hz, sfi, kp, hour_utc=0)

    print(f"\n  Path: W3 -> G, 40m, SFI 120, Kp 2")
    print(f"\n  SNR at 14:00 UTC (Day):     {sigma_to_db(snr_noon, band_id):+.1f} dB")
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global IRI_DATA, IRI_GRID_INDEX, IRI_SFI_BUCKETS

    print("=" * 60)
    print("  IONIS AUDIT-03 — TST-900 Band x Time Discrimination Tests")
    print("=" * 60)
    print(f"\n  Model Version: {MODEL_VERSION}")
    print(f"  DNN Dim: {DNN_DIM} (V23 recipe)")
    print(f"  IRI Atlas: 35-bucket (5-unit SFI steps)")
    print("  Comparison: V23-alpha (4/11 TST-900)")

    # Load IRI atlas
    print()
    IRI_DATA, IRI_GRID_INDEX, IRI_SFI_BUCKETS = load_iri_atlas()

    # Load model (safetensors)
    print(f"\nLoading {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"  ERROR: Checkpoint not found: {MODEL_PATH}")
        return 1

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
    print(f"  SFI Benefit: {metadata.get('sfi_benefit', 0):+.3f} sigma")
    print(f"  Storm Cost: {metadata.get('storm_cost', 0):+.3f} sigma")

    # Run tests
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
    print("  TST-900 SUMMARY: Band x Time Discrimination")
    print("=" * 60)

    passed = sum(1 for _, _, p in results if p)
    total = len(results)

    for test_id, name, p in results:
        status = "PASS" if p else "FAIL"
        print(f"  {test_id}: {name:<25s}  {status}")

    print()
    print(f"  Results: {passed}/{total} passed")
    print()
    print(f"  V23-alpha baseline: 4/11 TST-900")
    print(f"  V22-gamma baseline: 9/11 TST-900")

    if passed >= 8:
        print(f"\n  AUDIT-03: PASS — Matches or exceeds V22-gamma")
        return 0
    elif passed > 4:
        print(f"\n  AUDIT-03: PARTIAL — Better than V23-alpha but needs work")
        return 1
    else:
        print(f"\n  AUDIT-03: FAIL — Same or worse than V23-alpha")
        return 1


if __name__ == "__main__":
    sys.exit(main())
