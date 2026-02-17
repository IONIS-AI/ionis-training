#!/usr/bin/env python3
"""
test_tst100_canonical.py — IONIS V20 Canonical Path Tests

TST-100 Group: Verify the model produces reasonable predictions for
well-known HF propagation paths across all major regions.

Test Categories:
  A. North America ↔ Europe (TST-101 to TST-105)
  B. Trans-Pacific (TST-110 to TST-113)
  C. Europe ↔ Asia (TST-120 to TST-122)
  D. Africa Paths (TST-130 to TST-132)
  E. South America Paths (TST-140 to TST-142)
  F. Oceania Paths (TST-150 to TST-152)
  G. Regional/NVIS (TST-160 to TST-162)
  H. Band-Specific Physics (TST-170 to TST-175)

Total: 30 canonical path tests covering all continents and major bands.

Output is in sigma (Z-normalized). Approximate conversion: 1 sigma ≈ 6.7 dB.
Threshold for "path OPEN": > -2.5 sigma (within WSPR decode range).
"""

import json
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

DEVICE = get_device()

# ── Constants ────────────────────────────────────────────────────────────────

SIGMA_TO_DB = 6.7
PATH_OPEN_THRESHOLD = -2.5  # sigma

# Band frequencies (Hz) - WSPR dial frequencies
FREQ_160M = 1_836_600
FREQ_80M = 3_568_600
FREQ_60M = 5_287_200
FREQ_40M = 7_038_600
FREQ_30M = 10_138_700
FREQ_20M = 14_097_100
FREQ_17M = 18_104_600
FREQ_15M = 21_094_600
FREQ_12M = 24_924_600
FREQ_10M = 28_124_600

# ── Location Database ────────────────────────────────────────────────────────
# Format: (lat, lon, name, prefix)

LOCATIONS = {
    # North America
    'W3':  (39.14, -77.01, 'Maryland', 'W3'),
    'W6':  (34.05, -118.24, 'Los Angeles', 'W6'),
    'VE3': (43.65, -79.38, 'Toronto', 'VE3'),
    'KH6': (21.31, -157.86, 'Hawaii', 'KH6'),

    # Europe
    'G':   (51.50, -0.12, 'London', 'G'),
    'DL':  (52.52, 13.40, 'Berlin', 'DL'),
    'OH':  (60.17, 24.94, 'Helsinki', 'OH'),
    'I':   (41.90, 12.50, 'Rome', 'I'),

    # Asia
    'JA':  (35.68, 139.69, 'Tokyo', 'JA'),
    'HL':  (37.57, 126.98, 'Seoul', 'HL'),
    'VU':  (12.97, 77.59, 'Bangalore', 'VU'),
    'BY':  (39.90, 116.40, 'Beijing', 'BY'),

    # Oceania
    'VK':  (-33.87, 151.21, 'Sydney', 'VK'),
    'ZL':  (-41.29, 174.78, 'Wellington', 'ZL'),

    # Africa
    'ZS':  (-33.93, 18.42, 'Cape Town', 'ZS'),
    '5H':  (-6.17, 35.74, 'Tanzania', '5H'),
    'EA8': (28.29, -16.63, 'Canary Islands', 'EA8'),

    # South America
    'PY':  (-23.55, -46.63, 'São Paulo', 'PY'),
    'LU':  (-34.60, -58.38, 'Buenos Aires', 'LU'),
    'CE':  (-33.45, -70.67, 'Santiago', 'CE'),

    # Polar
    'OX':  (64.18, -51.72, 'Greenland', 'OX'),
}


# ── Prediction Helper ────────────────────────────────────────────────────────

def predict(model, device, tx_loc, rx_loc, freq_hz, sfi, kp, hour_utc, month=6):
    """Make a prediction for a given path and conditions."""
    tx_lat, tx_lon = LOCATIONS[tx_loc][:2]
    rx_lat, rx_lon = LOCATIONS[rx_loc][:2]

    # Calculate distance (haversine)
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
        snr = model(features).item()

    return snr, distance_km


def run_path_test(model, device, test_id, tx, rx, band_name, freq_hz,
                  sfi, kp, hour_utc, month=6, threshold=PATH_OPEN_THRESHOLD,
                  description=None):
    """Generic path test runner."""
    tx_name = f"{LOCATIONS[tx][3]} ({LOCATIONS[tx][2]})"
    rx_name = f"{LOCATIONS[rx][3]} ({LOCATIONS[rx][2]})"

    snr, dist = predict(model, device, tx, rx, freq_hz, sfi, kp, hour_utc, month)

    print(f"\n  Path: {tx_name} → {rx_name}")
    print(f"  Distance: {dist:,.0f} km | Band: {band_name}")
    print(f"  Conditions: SFI {sfi}, Kp {kp}, {hour_utc:02d}:00 UTC")
    print(f"  Prediction: {snr:+.3f}σ ({snr * SIGMA_TO_DB:+.1f} dB)")

    passed = snr > threshold
    return passed, snr, dist


# ══════════════════════════════════════════════════════════════════════════════
#  CATEGORY A: NORTH AMERICA ↔ EUROPE
# ══════════════════════════════════════════════════════════════════════════════

def test_tst101(model, device):
    """TST-101: W3 → G (20m Day) — Classic transatlantic"""
    print("\n" + "=" * 65)
    print("TST-101: W3 → G — North America to Europe (20m Day)")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-101", 'W3', 'G', '20m', FREQ_20M,
        sfi=150, kp=2, hour_utc=14
    )
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


def test_tst102(model, device):
    """TST-102: W3 → G (20m Night) — Grey line propagation"""
    print("\n" + "=" * 65)
    print("TST-102: W3 → G — North America to Europe (20m Grey Line)")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-102", 'W3', 'G', '20m', FREQ_20M,
        sfi=150, kp=2, hour_utc=4
    )
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


def test_tst103(model, device):
    """TST-103: G → W6 (20m) — Europe to US West Coast"""
    print("\n" + "=" * 65)
    print("TST-103: G → W6 — Europe to US West Coast (20m)")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-103", 'G', 'W6', '20m', FREQ_20M,
        sfi=150, kp=2, hour_utc=18
    )
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


def test_tst104(model, device):
    """TST-104: W3 → G (40m) — Transatlantic on 40m"""
    print("\n" + "=" * 65)
    print("TST-104: W3 → G — North America to Europe (40m)")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-104", 'W3', 'G', '40m', FREQ_40M,
        sfi=150, kp=2, hour_utc=22  # Evening for 40m
    )
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


def test_tst105(model, device):
    """TST-105: VE3 → DL (20m) — Canada to Germany"""
    print("\n" + "=" * 65)
    print("TST-105: VE3 → DL — Canada to Germany (20m)")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-105", 'VE3', 'DL', '20m', FREQ_20M,
        sfi=150, kp=2, hour_utc=14
    )
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


# ══════════════════════════════════════════════════════════════════════════════
#  CATEGORY B: TRANS-PACIFIC
# ══════════════════════════════════════════════════════════════════════════════

def test_tst110(model, device):
    """TST-110: W6 → JA (20m) — US West Coast to Japan"""
    print("\n" + "=" * 65)
    print("TST-110: W6 → JA — US West Coast to Japan (20m)")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-110", 'W6', 'JA', '20m', FREQ_20M,
        sfi=150, kp=2, hour_utc=16
    )
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


def test_tst111(model, device):
    """TST-111: JA → W3 (20m) — Japan to US East Coast"""
    print("\n" + "=" * 65)
    print("TST-111: JA → W3 — Japan to US East Coast (20m)")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-111", 'JA', 'W3', '20m', FREQ_20M,
        sfi=150, kp=2, hour_utc=23  # Long path timing
    )
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


def test_tst112(model, device):
    """TST-112: KH6 → JA (20m) — Hawaii to Japan"""
    print("\n" + "=" * 65)
    print("TST-112: KH6 → JA — Hawaii to Japan (20m)")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-112", 'KH6', 'JA', '20m', FREQ_20M,
        sfi=150, kp=2, hour_utc=6
    )
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


def test_tst113(model, device):
    """TST-113: VK → W6 (20m) — Australia to US West Coast"""
    print("\n" + "=" * 65)
    print("TST-113: VK → W6 — Australia to US West Coast (20m)")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-113", 'VK', 'W6', '20m', FREQ_20M,
        sfi=150, kp=2, hour_utc=5
    )
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


# ══════════════════════════════════════════════════════════════════════════════
#  CATEGORY C: EUROPE ↔ ASIA
# ══════════════════════════════════════════════════════════════════════════════

def test_tst120(model, device):
    """TST-120: G → JA (20m) — Europe to Japan"""
    print("\n" + "=" * 65)
    print("TST-120: G → JA — Europe to Japan (20m)")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-120", 'G', 'JA', '20m', FREQ_20M,
        sfi=150, kp=2, hour_utc=8
    )
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


def test_tst121(model, device):
    """TST-121: DL → VU (20m) — Germany to India"""
    print("\n" + "=" * 65)
    print("TST-121: DL → VU — Germany to India (20m)")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-121", 'DL', 'VU', '20m', FREQ_20M,
        sfi=150, kp=2, hour_utc=12
    )
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


def test_tst122(model, device):
    """TST-122: JA → OH (20m) — Japan to Finland (polar-ish)"""
    print("\n" + "=" * 65)
    print("TST-122: JA → OH — Japan to Finland (20m)")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-122", 'JA', 'OH', '20m', FREQ_20M,
        sfi=150, kp=2, hour_utc=10
    )
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


# ══════════════════════════════════════════════════════════════════════════════
#  CATEGORY D: AFRICA PATHS
# ══════════════════════════════════════════════════════════════════════════════

def test_tst130(model, device):
    """TST-130: ZS → G (20m) — South Africa to Europe"""
    print("\n" + "=" * 65)
    print("TST-130: ZS → G — South Africa to Europe (20m)")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-130", 'ZS', 'G', '20m', FREQ_20M,
        sfi=150, kp=2, hour_utc=14
    )
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


def test_tst131(model, device):
    """TST-131: ZS → W3 (20m) — South Africa to US East Coast"""
    print("\n" + "=" * 65)
    print("TST-131: ZS → W3 — South Africa to US East Coast (20m)")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-131", 'ZS', 'W3', '20m', FREQ_20M,
        sfi=150, kp=2, hour_utc=16
    )
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


def test_tst132(model, device):
    """TST-132: 5H → DL (20m) — Tanzania to Germany"""
    print("\n" + "=" * 65)
    print("TST-132: 5H → DL — Tanzania to Germany (20m)")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-132", '5H', 'DL', '20m', FREQ_20M,
        sfi=150, kp=2, hour_utc=14
    )
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


# ══════════════════════════════════════════════════════════════════════════════
#  CATEGORY E: SOUTH AMERICA PATHS
# ══════════════════════════════════════════════════════════════════════════════

def test_tst140(model, device):
    """TST-140: PY → W3 (20m) — Brazil to US East Coast"""
    print("\n" + "=" * 65)
    print("TST-140: PY → W3 — Brazil to US East Coast (20m)")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-140", 'PY', 'W3', '20m', FREQ_20M,
        sfi=150, kp=2, hour_utc=18
    )
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


def test_tst141(model, device):
    """TST-141: LU → G (20m) — Argentina to Europe"""
    print("\n" + "=" * 65)
    print("TST-141: LU → G — Argentina to Europe (20m)")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-141", 'LU', 'G', '20m', FREQ_20M,
        sfi=150, kp=2, hour_utc=16
    )
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


def test_tst142(model, device):
    """TST-142: PY → VU (20m) — Brazil to India (equatorial long-haul)"""
    print("\n" + "=" * 65)
    print("TST-142: PY → VU — Brazil to India (20m, Equatorial)")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-142", 'PY', 'VU', '20m', FREQ_20M,
        sfi=150, kp=2, hour_utc=14
    )
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


# ══════════════════════════════════════════════════════════════════════════════
#  CATEGORY F: OCEANIA PATHS
# ══════════════════════════════════════════════════════════════════════════════

def test_tst150(model, device):
    """TST-150: VK → G (20m) — Australia to Europe"""
    print("\n" + "=" * 65)
    print("TST-150: VK → G — Australia to Europe (20m)")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-150", 'VK', 'G', '20m', FREQ_20M,
        sfi=150, kp=2, hour_utc=8
    )
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


def test_tst151(model, device):
    """TST-151: ZL → JA (20m) — New Zealand to Japan"""
    print("\n" + "=" * 65)
    print("TST-151: ZL → JA — New Zealand to Japan (20m)")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-151", 'ZL', 'JA', '20m', FREQ_20M,
        sfi=150, kp=2, hour_utc=4
    )
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


def test_tst152(model, device):
    """TST-152: VK → ZS (20m) — Australia to South Africa"""
    print("\n" + "=" * 65)
    print("TST-152: VK → ZS — Australia to South Africa (20m)")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-152", 'VK', 'ZS', '20m', FREQ_20M,
        sfi=150, kp=2, hour_utc=10
    )
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


# ══════════════════════════════════════════════════════════════════════════════
#  CATEGORY G: REGIONAL / NVIS
# ══════════════════════════════════════════════════════════════════════════════

def test_tst160(model, device):
    """TST-160: G → DL (20m) — Intra-Europe short path"""
    print("\n" + "=" * 65)
    print("TST-160: G → DL — Intra-Europe (20m)")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-160", 'G', 'DL', '20m', FREQ_20M,
        sfi=150, kp=2, hour_utc=12
    )
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


def test_tst161(model, device):
    """TST-161: JA → HL (20m) — Intra-Asia (Japan to Korea)"""
    print("\n" + "=" * 65)
    print("TST-161: JA → HL — Intra-Asia Japan to Korea (20m)")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-161", 'JA', 'HL', '20m', FREQ_20M,
        sfi=150, kp=2, hour_utc=6
    )
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


def test_tst162(model, device):
    """TST-162: NVIS 80m — Central US regional"""
    print("\n" + "=" * 65)
    print("TST-162: NVIS 80m — Central US Regional")
    print("=" * 65)

    # Direct coordinates for NVIS (not in LOCATIONS dict)
    tx_lat, tx_lon = 40.0, -100.0
    rx_lat, rx_lon = 42.0, -98.0

    # Manual prediction for non-standard path
    R = 6371.0
    lat1_r, lat2_r = np.radians(tx_lat), np.radians(rx_lat)
    dlat = np.radians(rx_lat - tx_lat)
    dlon = np.radians(rx_lon - tx_lon)
    a = np.sin(dlat/2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2
    distance_km = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    x = np.sin(dlon) * np.cos(lat2_r)
    y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)
    azimuth = (np.degrees(np.arctan2(x, y)) + 360) % 360

    distance = distance_km / 20000.0
    freq_log = np.log10(FREQ_80M) / 8.0
    hour_utc = 2
    month = 6
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
    sfi_norm = 100.0 / 300.0
    kp_penalty = 1.0 - 2.0 / 9.0

    features = torch.tensor(
        [[distance, freq_log, hour_sin, hour_cos,
          az_sin, az_cos, lat_diff, midpoint_lat,
          season_sin, season_cos, day_night_est,
          sfi_norm, kp_penalty]],
        dtype=torch.float32, device=DEVICE,
    )

    with torch.no_grad():
        snr = model(features).item()

    print(f"\n  Path: Central US NVIS (~280 km)")
    print(f"  Distance: {distance_km:,.0f} km | Band: 80m")
    print(f"  Conditions: SFI 100, Kp 2, 02:00 UTC (night)")
    print(f"  Prediction: {snr:+.3f}σ ({snr * SIGMA_TO_DB:+.1f} dB)")

    passed = snr > -2.0  # NVIS should be stronger
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


# ══════════════════════════════════════════════════════════════════════════════
#  CATEGORY H: BAND-SPECIFIC PHYSICS
# ══════════════════════════════════════════════════════════════════════════════

def test_tst170(model, device):
    """TST-170: Polar path quiet (OX → OH, Kp 2)"""
    print("\n" + "=" * 65)
    print("TST-170: OX → OH — Polar Path Quiet (Kp 2)")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-170", 'OX', 'OH', '20m', FREQ_20M,
        sfi=150, kp=2, hour_utc=12
    )
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed, snr


def test_tst171(model, device, quiet_snr):
    """TST-171: Polar path storm (OX → OH, Kp 8)"""
    print("\n" + "=" * 65)
    print("TST-171: OX → OH — Polar Path Storm (Kp 8)")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-171", 'OX', 'OH', '20m', FREQ_20M,
        sfi=150, kp=8, hour_utc=12
    )
    degradation = quiet_snr - snr
    print(f"  Storm degradation: {degradation:+.3f}σ ({degradation * SIGMA_TO_DB:+.1f} dB)")

    passed = degradation > 1.0  # Should be > 1 sigma degradation
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


def test_tst172(model, device):
    """TST-172: 10m low SFI (W3 → G)"""
    print("\n" + "=" * 65)
    print("TST-172: W3 → G — 10m Low SFI (SFI 80)")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-172", 'W3', 'G', '10m', FREQ_10M,
        sfi=80, kp=2, hour_utc=14
    )
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed, snr


def test_tst173(model, device, low_sfi_snr):
    """TST-173: 10m high SFI (W3 → G)"""
    print("\n" + "=" * 65)
    print("TST-173: W3 → G — 10m High SFI (SFI 200)")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-173", 'W3', 'G', '10m', FREQ_10M,
        sfi=200, kp=2, hour_utc=14
    )
    improvement = snr - low_sfi_snr
    print(f"  SFI improvement: {improvement:+.3f}σ ({improvement * SIGMA_TO_DB:+.1f} dB)")

    passed = improvement > 0.3  # Should improve with higher SFI
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


def test_tst174(model, device):
    """TST-174: 40m night path (W3 → G)"""
    print("\n" + "=" * 65)
    print("TST-174: W3 → G — 40m Night Path")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-174", 'W3', 'G', '40m', FREQ_40M,
        sfi=150, kp=2, hour_utc=2  # Night
    )
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


def test_tst175(model, device):
    """TST-175: 160m regional night (VE3 → W3)"""
    print("\n" + "=" * 65)
    print("TST-175: VE3 → W3 — 160m Regional Night")
    print("=" * 65)
    passed, snr, _ = run_path_test(
        model, device, "TST-175", 'VE3', 'W3', '160m', FREQ_160M,
        sfi=100, kp=2, hour_utc=4  # Night
    )
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  IONIS V20 — TST-100 Canonical Path Tests (Expanded)")
    print("=" * 65)
    print("\n  Comprehensive coverage: All continents + major bands")
    print(f"  Threshold for path OPEN: > {PATH_OPEN_THRESHOLD}σ")

    # Load model
    print(f"\nLoading {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, weights_only=True, map_location=DEVICE)

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

    results = []

    # Category A: North America ↔ Europe
    print("\n" + "─" * 65)
    print("  CATEGORY A: NORTH AMERICA ↔ EUROPE")
    print("─" * 65)
    results.append(("TST-101", "W3→G 20m Day", test_tst101(model, DEVICE)))
    results.append(("TST-102", "W3→G 20m Night", test_tst102(model, DEVICE)))
    results.append(("TST-103", "G→W6 20m", test_tst103(model, DEVICE)))
    results.append(("TST-104", "W3→G 40m", test_tst104(model, DEVICE)))
    results.append(("TST-105", "VE3→DL 20m", test_tst105(model, DEVICE)))

    # Category B: Trans-Pacific
    print("\n" + "─" * 65)
    print("  CATEGORY B: TRANS-PACIFIC")
    print("─" * 65)
    results.append(("TST-110", "W6→JA 20m", test_tst110(model, DEVICE)))
    results.append(("TST-111", "JA→W3 20m", test_tst111(model, DEVICE)))
    results.append(("TST-112", "KH6→JA 20m", test_tst112(model, DEVICE)))
    results.append(("TST-113", "VK→W6 20m", test_tst113(model, DEVICE)))

    # Category C: Europe ↔ Asia
    print("\n" + "─" * 65)
    print("  CATEGORY C: EUROPE ↔ ASIA")
    print("─" * 65)
    results.append(("TST-120", "G→JA 20m", test_tst120(model, DEVICE)))
    results.append(("TST-121", "DL→VU 20m", test_tst121(model, DEVICE)))
    results.append(("TST-122", "JA→OH 20m", test_tst122(model, DEVICE)))

    # Category D: Africa
    print("\n" + "─" * 65)
    print("  CATEGORY D: AFRICA PATHS")
    print("─" * 65)
    results.append(("TST-130", "ZS→G 20m", test_tst130(model, DEVICE)))
    results.append(("TST-131", "ZS→W3 20m", test_tst131(model, DEVICE)))
    results.append(("TST-132", "5H→DL 20m", test_tst132(model, DEVICE)))

    # Category E: South America
    print("\n" + "─" * 65)
    print("  CATEGORY E: SOUTH AMERICA PATHS")
    print("─" * 65)
    results.append(("TST-140", "PY→W3 20m", test_tst140(model, DEVICE)))
    results.append(("TST-141", "LU→G 20m", test_tst141(model, DEVICE)))
    results.append(("TST-142", "PY→VU 20m", test_tst142(model, DEVICE)))

    # Category F: Oceania
    print("\n" + "─" * 65)
    print("  CATEGORY F: OCEANIA PATHS")
    print("─" * 65)
    results.append(("TST-150", "VK→G 20m", test_tst150(model, DEVICE)))
    results.append(("TST-151", "ZL→JA 20m", test_tst151(model, DEVICE)))
    results.append(("TST-152", "VK→ZS 20m", test_tst152(model, DEVICE)))

    # Category G: Regional/NVIS
    print("\n" + "─" * 65)
    print("  CATEGORY G: REGIONAL / NVIS")
    print("─" * 65)
    results.append(("TST-160", "G→DL 20m", test_tst160(model, DEVICE)))
    results.append(("TST-161", "JA→HL 20m", test_tst161(model, DEVICE)))
    results.append(("TST-162", "NVIS 80m", test_tst162(model, DEVICE)))

    # Category H: Band-Specific Physics
    print("\n" + "─" * 65)
    print("  CATEGORY H: BAND-SPECIFIC PHYSICS")
    print("─" * 65)
    tst170_result, quiet_snr = test_tst170(model, DEVICE)
    results.append(("TST-170", "Polar Quiet", tst170_result))
    results.append(("TST-171", "Polar Storm", test_tst171(model, DEVICE, quiet_snr)))

    tst172_result, low_sfi_snr = test_tst172(model, DEVICE)
    results.append(("TST-172", "10m Low SFI", tst172_result))
    results.append(("TST-173", "10m High SFI", test_tst173(model, DEVICE, low_sfi_snr)))

    results.append(("TST-174", "40m Night", test_tst174(model, DEVICE)))
    results.append(("TST-175", "160m Regional", test_tst175(model, DEVICE)))

    # Summary
    print("\n" + "=" * 65)
    print("  TST-100 SUMMARY: Canonical Paths (Expanded)")
    print("=" * 65)

    passed = sum(1 for _, _, p in results if p)
    total = len(results)

    categories = [
        ("A: NA↔EU", results[0:5]),
        ("B: Trans-Pac", results[5:9]),
        ("C: EU↔Asia", results[9:12]),
        ("D: Africa", results[12:15]),
        ("E: S.America", results[15:18]),
        ("F: Oceania", results[18:21]),
        ("G: Regional", results[21:24]),
        ("H: Physics", results[24:30]),
    ]

    for cat_name, cat_results in categories:
        cat_passed = sum(1 for _, _, p in cat_results if p)
        cat_total = len(cat_results)
        print(f"\n  {cat_name}: {cat_passed}/{cat_total}")
        for test_id, name, p in cat_results:
            status = "PASS" if p else "FAIL"
            print(f"    {test_id}: {name:<18s} {status}")

    print(f"\n  {'─'*40}")
    print(f"  TOTAL: {passed}/{total} passed")

    if passed == total:
        print("\n  ALL TST-100 TESTS PASSED")
        print("  Global HF propagation coverage verified.")
        return 0
    else:
        print(f"\n  {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
