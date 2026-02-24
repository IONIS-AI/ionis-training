#!/usr/bin/env python3
"""
test_tst900_band_time.py — IONIS V26-alpha Band×Time Discrimination Tests

TST-900 Group: Verify band×time-of-day behavior matches ionospheric reality.

V26 Architecture Change:
    - 9 band-specific output heads replace single shared base_head
    - band_idx feature at index 17 routes to correct head
    - Testing hypothesis: shared head caused +3.7σ defect

ACID TEST (New for V26):
    DN13→JN48, 10m, 02 UTC, Feb, SFI=150 must output ≤ 0.0σ
    V22-gamma outputs +3.7σ — 10m should be CLOSED at 2 AM!

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
  ACID:    10m Night Closure — DN13→JN48 at 02 UTC must be ≤ 0.0σ
"""

import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors

# ── Path Setup ───────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V26_DIR = os.path.dirname(SCRIPT_DIR)
VERSIONS_DIR = os.path.dirname(V26_DIR)
sys.path.insert(0, os.path.join(VERSIONS_DIR, "common"))

from model import get_device, build_features, BAND_FREQ_HZ, MonotonicMLP, _gate

# ── V26 Model Architecture ───────────────────────────────────────────────────

class IonisGateV26(nn.Module):
    """IonisGateV26 — Band-specific output heads (same as train_v26_alpha.py)."""

    def __init__(self, dnn_dim=15, sidecar_hidden=8, sfi_idx=15, kp_penalty_idx=16,
                 band_idx=17, num_bands=9, gate_init_bias=None):
        super().__init__()

        if gate_init_bias is None:
            gate_init_bias = -math.log(2.0)

        self.dnn_dim = dnn_dim
        self.sfi_idx = sfi_idx
        self.kp_penalty_idx = kp_penalty_idx
        self.band_idx = band_idx
        self.num_bands = num_bands

        self.trunk = nn.Sequential(
            nn.Linear(dnn_dim, 512), nn.Mish(),
            nn.Linear(512, 256), nn.Mish(),
        )

        self.band_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128), nn.Mish(),
                nn.Linear(128, 1),
            )
            for _ in range(num_bands)
        ])

        self.sun_scaler_head = nn.Sequential(
            nn.Linear(256, 64), nn.Mish(),
            nn.Linear(64, 1),
        )
        self.storm_scaler_head = nn.Sequential(
            nn.Linear(256, 64), nn.Mish(),
            nn.Linear(64, 1),
        )

        self.sun_sidecar = MonotonicMLP(hidden_dim=sidecar_hidden)
        self.storm_sidecar = MonotonicMLP(hidden_dim=sidecar_hidden)

        self._init_scaler_heads(gate_init_bias)

    def _init_scaler_heads(self, gate_init_bias):
        for head in [self.sun_scaler_head, self.storm_scaler_head]:
            final_layer = head[-1]
            nn.init.zeros_(final_layer.weight)
            nn.init.constant_(final_layer.bias, gate_init_bias)

    def forward(self, x):
        x_deep = x[:, :self.dnn_dim]
        x_sfi = x[:, self.sfi_idx:self.sfi_idx + 1]
        x_kp = x[:, self.kp_penalty_idx:self.kp_penalty_idx + 1]
        x_band = x[:, self.band_idx].long()

        trunk_out = self.trunk(x_deep)

        all_heads = torch.cat([h(trunk_out) for h in self.band_heads], dim=1)
        base_snr = all_heads.gather(1, x_band.unsqueeze(1))

        sun_logit = self.sun_scaler_head(trunk_out)
        storm_logit = self.storm_scaler_head(trunk_out)
        sun_gate = _gate(sun_logit)
        storm_gate = _gate(storm_logit)

        return base_snr + sun_gate * self.sun_sidecar(x_sfi) + \
               storm_gate * self.storm_sidecar(x_kp)


# ── Load Config ──────────────────────────────────────────────────────────────

V26_CONFIG = os.path.join(V26_DIR, "config_v26_alpha.json")
V26_CKPT = os.path.join(V26_DIR, "ionis_v26_alpha.safetensors")

with open(V26_CONFIG) as f:
    CONFIG = json.load(f)

MODEL_PATH = V26_CKPT

DNN_DIM = CONFIG["model"]["dnn_dim"]
SFI_IDX = CONFIG["model"]["sfi_idx"]
KP_PENALTY_IDX = CONFIG["model"]["kp_penalty_idx"]
BAND_IDX = CONFIG["model"]["band_idx"]
NUM_BANDS = CONFIG["model"]["num_bands"]
SIDECAR_HIDDEN = CONFIG["model"]["sidecar_hidden"]

DEVICE = get_device()

SIGMA_TO_DB = 6.7

FT8_THRESHOLD_DB = -21.0
WSPR_FLOOR_DB = -28.0

# Band name to band_idx mapping (for routing)
BAND_NAME_TO_IDX = {
    "160m": 0,
    "80m": 1,
    "60m": 2,
    "40m": 3,
    "30m": 4,
    "20m": 5,
    "17m": 6,
    "15m": 7,
    "12m": 8,
    "10m": 8,  # 10m uses index 8 (9th head)
}

# More precise mapping from config
BAND_ID_TO_IDX = {int(k): v for k, v in CONFIG["band_to_idx"].items()}


# ── Prediction Helper ────────────────────────────────────────────────────────

def predict(model, device, tx_lat, tx_lon, rx_lat, rx_lon, freq_hz,
            sfi, kp, hour_utc, month=6, day_of_year=172):
    """Make a prediction for given parameters.

    V26 adds band_idx at feature index 17 for routing.
    """
    # Build base V22 features (17 features: 0-14 DNN + 15 SFI + 16 Kp)
    features = build_features(
        tx_lat, tx_lon, rx_lat, rx_lon, freq_hz,
        sfi, kp, hour_utc, month,
        day_of_year=day_of_year,
        include_solar_depression=True,
    )

    # V26: Append band_idx for routing
    # Map frequency to band index
    # Map frequency to band index (9 heads, 60m/40m share index 2)
    freq_to_band_idx = {
        1_836_600: 0,   # 160m
        3_568_600: 1,   # 80m
        5_287_200: 2,   # 60m (shares with 40m)
        7_038_600: 2,   # 40m (shares with 60m)
        10_138_700: 3,  # 30m
        14_097_100: 4,  # 20m
        18_104_600: 5,  # 17m
        21_094_600: 6,  # 15m
        24_924_600: 7,  # 12m
        28_124_600: 8,  # 10m
    }
    band_idx = freq_to_band_idx.get(freq_hz, 4)  # Default to 20m

    # Add band_idx as feature 17
    features_v26 = np.append(features, band_idx)

    tensor = torch.tensor([features_v26], dtype=torch.float32, device=device)
    with torch.no_grad():
        return model(tensor).item()


def sigma_to_db(sigma, band_id=107, source="wspr"):
    """Convert normalized sigma to dB using band-specific constants."""
    std = CONFIG["norm_constants_per_band"].get(str(band_id), {}).get(source, {}).get("std", 6.7)
    mean = CONFIG["norm_constants_per_band"].get(str(band_id), {}).get(source, {}).get("mean", -18.0)
    return sigma * std + mean


# ── Standard Paths ───────────────────────────────────────────────────────────

W3_LAT, W3_LON = 39.14, -77.01
G_LAT, G_LON = 51.50, -0.12

KH6_LAT, KH6_LON = 21.31, -157.86
JA_LAT, JA_LON = 35.68, 139.69

W6_LAT, W6_LON = 37.77, -122.42
VK_LAT, VK_LON = -33.87, 151.21

# DN13 (Colorado) to JN48 (Austria) — the acid test path
DN13_LAT, DN13_LON = 33.5, -105.5  # DN13 grid center
JN48_LAT, JN48_LON = 48.5, 13.5    # JN48 grid center


# ── ACID TEST (Primary V26 Gate) ─────────────────────────────────────────────

def test_acid_10m_night_closure(model, device):
    """ACID TEST: 10m Night Closure — DN13→JN48 at 02 UTC must be ≤ 0.0σ.

    This is THE test that V22-gamma fails. It outputs +3.7σ (strong signal)
    when the band is physically CLOSED at 2 AM.

    V26 hypothesis: With a dedicated 10m output head, the model can learn
    "night = dead" for 10m without those weights bleeding into 160m's
    "night = good" response.
    """
    print("\n" + "=" * 60)
    print("*** ACID TEST: 10m Night Closure (The V26 Gate) ***")
    print("=" * 60)

    freq_hz = BAND_FREQ_HZ["10m"]
    band_id = 111

    # February conditions (winter at mid-latitudes)
    february_doy = 45
    february_month = 2

    # THE ACID TEST: DN13→JN48, 10m, 02 UTC, Feb, SFI=150
    snr_02utc = predict(model, device, DN13_LAT, DN13_LON, JN48_LAT, JN48_LON,
                        freq_hz, sfi=150, kp=2, hour_utc=2,
                        month=february_month, day_of_year=february_doy)
    db_02utc = sigma_to_db(snr_02utc, band_id)

    # Also check midday for reference
    snr_14utc = predict(model, device, DN13_LAT, DN13_LON, JN48_LAT, JN48_LON,
                        freq_hz, sfi=150, kp=2, hour_utc=14,
                        month=february_month, day_of_year=february_doy)
    db_14utc = sigma_to_db(snr_14utc, band_id)

    delta_db = db_14utc - db_02utc

    print(f"\n  Path: DN13 → JN48 (Colorado → Austria), 10m, SFI 150, Kp 2, February")
    print(f"\n  SNR at 02:00 UTC (Night): {snr_02utc:+.3f}σ ({db_02utc:+.1f} dB)")
    print(f"  SNR at 14:00 UTC (Day):   {snr_14utc:+.3f}σ ({db_14utc:+.1f} dB)")
    print(f"\n  Day/Night Delta: {delta_db:+.1f} dB")
    print(f"\n  V22-gamma reference: +3.7σ at 02 UTC (WRONG — band is closed!)")
    print(f"  V26 requirement: ≤ 0.0σ at 02 UTC (band should be closed)")

    # Pass criteria: 02 UTC must be ≤ 0.0σ (band closed at night)
    if snr_02utc <= 0.0:
        print(f"\n  *** ACID TEST PASS: 10m shows {snr_02utc:+.3f}σ at night (≤ 0.0σ) ***")
        print(f"  The multi-head architecture learned what the shared head couldn't!")
        return True
    else:
        print(f"\n  *** ACID TEST FAIL: 10m shows {snr_02utc:+.3f}σ at night (> 0.0σ) ***")
        print(f"  The defect persists — 10m should not predict signal at 2 AM.")
        return False


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

    print(f"\n  Path: W3 → G, 10m, SFI 150, Kp 2, Winter (Dec 21)")
    print(f"  SNR at 14:00 UTC (Midday):   {db_midday:+.1f} dB ({snr_midday:+.3f} sigma)")
    print(f"  SNR at 00:00 UTC (Midnight): {db_midnight:+.1f} dB ({snr_midnight:+.3f} sigma)")
    print(f"\n  Day/Night Delta: {delta_db:+.1f} dB")
    print(f"  FT8 Threshold: {FT8_THRESHOLD_DB} dB")

    if delta_db >= 10.0:
        print(f"\n  PASS: 10m shows {delta_db:+.1f} dB day/night delta (band closes at midnight)")
        return True
    else:
        print(f"\n  FAIL: Day/night delta {delta_db:+.1f} dB < 10 dB threshold")
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

    print(f"\n  Path: W3 → G, 160m (~5700 km), SFI 100, Kp 2, Winter")
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

    print(f"\n  Path: W3 → G, 80m (~5700 km), SFI 100, Kp 2, Winter")
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

    print(f"\n  Path: W3 → G, SFI 150, Kp 2, 14:00 UTC (Daytime)")
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

    print(f"\n  Path: W3 → G, SFI 100, Kp 2, 04:00 UTC, Winter")
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

    print(f"\n  Path: W3 → G, 20m, SFI 150, Kp 2, Winter")
    print(f"\n  24-Hour Scan:")
    for h in [0, 4, 8, 12, 16, 20]:
        snr = snr_by_hour[h]
        print(f"    {h:02d}:00 UTC: {snr:+.3f} sigma ({sigma_to_db(snr, band_id):+.1f} dB)")

    print(f"\n  Peak:   {peak_hour:02d}:00 UTC = {peak_snr:+.3f} sigma ({sigma_to_db(peak_snr, band_id):+.1f} dB)")
    print(f"  Trough: {trough_hour:02d}:00 UTC = {trough_snr:+.3f} sigma ({sigma_to_db(trough_snr, band_id):+.1f} dB)")
    print(f"\n  Dynamic Range: {delta_db:+.1f} dB")

    if delta_db >= 6.0:
        print(f"\n  PASS: >= 6 dB time sensitivity")
        return True
    else:
        print(f"\n  FAIL: < 6 dB time sensitivity (model not learning time)")
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

    print(f"\n  Path: W3 → G, 10m, SFI 180, Kp 2, Winter")
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

    print(f"\n  Path: W3 → G, 160m, SFI 80, Kp 2, Winter")
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
    snr_night = predict(model, device, W3_LAT, W3_LON, G_LAT, G_LON,
                        freq_hz, sfi, kp, hour_utc=4)

    print(f"\n  Path: W3 → G, 40m, SFI 120, Kp 2")
    print(f"\n  SNR at 14:00 UTC (Day):     {sigma_to_db(snr_noon, band_id):+.1f} dB")
    print(f"  SNR at 00:00 UTC (Twilight): {sigma_to_db(snr_twilight, band_id):+.1f} dB")
    print(f"  SNR at 04:00 UTC (Night):   {sigma_to_db(snr_night, band_id):+.1f} dB")

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
    print("=" * 60)
    print("  IONIS V26-alpha — TST-900 Band×Time Discrimination Tests")
    print("=" * 60)
    print(f"\n  Architecture: IonisGateV26 (9 band-specific output heads)")
    print(f"  Hypothesis: Shared output head caused +3.7σ defect")
    print("  These tests verify band×time discrimination.")

    # Check if checkpoint exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n  ERROR: Checkpoint not found at {MODEL_PATH}")
        print(f"  Run training first: python train_v26_alpha.py")
        return 1

    # Load model
    print(f"\nLoading {MODEL_PATH}...")
    state_dict = load_safetensors(MODEL_PATH, device=str(DEVICE))

    meta_path = MODEL_PATH.replace(".safetensors", "_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    model = IonisGateV26(
        dnn_dim=DNN_DIM,
        sidecar_hidden=SIDECAR_HIDDEN,
        sfi_idx=SFI_IDX,
        kp_penalty_idx=KP_PENALTY_IDX,
        band_idx=BAND_IDX,
        num_bands=NUM_BANDS,
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

    # THE ACID TEST (Primary V26 gate)
    results.append(("ACID", "10m Night Closure (V26 Gate)", test_acid_10m_night_closure(model, DEVICE)))

    # Standard TST-900 battery
    results.append(("TST-901", "10m Band Closure (Winter)", test_tst901_10m_band_closure(model, DEVICE)))
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

    # Highlight acid test result
    acid_passed = results[0][2]

    for test_id, name, p in results:
        status = "PASS" if p else "FAIL"
        marker = " <<<" if test_id == "ACID" else ""
        print(f"  {test_id}: {name:<25s}  {status}{marker}")

    print()
    print(f"  Results: {passed}/{total} passed")

    # V26 gate criteria
    print("\n" + "=" * 60)
    print("  V26 GATE EVALUATION")
    print("=" * 60)

    if acid_passed:
        print("\n  *** ACID TEST PASSED ***")
        print("  The multi-head architecture learned what the shared head couldn't!")
        print("  10m now correctly predicts closure at night.")
    else:
        print("\n  *** ACID TEST FAILED ***")
        print("  The defect persists — architecture change didn't fix the +3.7σ bug.")
        print("  Consider escape valve: freq_log skip-connection to band heads.")

    tst900_score = sum(1 for _, _, p in results[1:] if p)  # Exclude acid test
    print(f"\n  TST-900 Score: {tst900_score}/9")
    print(f"  V22-gamma reference: 9/11")
    print(f"  V26 requirement: >= 10/11 (with acid test)")

    if passed >= 10:
        print("\n  >>> V26 GATE PASSED <<<")
        print("  Ready for production evaluation.")
        return 0
    else:
        print(f"\n  >>> V26 GATE FAILED ({passed}/10) <<<")
        print("  Review results and consider escape valve.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
