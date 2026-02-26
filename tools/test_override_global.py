#!/usr/bin/env python3
"""
test_override_global.py — Global Validation Test Battery for PhysicsOverrideLayer

Tests the override logic directly using solar_elevation_deg and the override
functions. No model needed — pure physics rule validation.

Defects under test:
  1. TX in deep night (-55°), RX in daylight (+17°) — override did NOT fire
     because old rule required BOTH endpoints dark. Result: ham-stats.com shows
     10m "CW" from Idaho to Europe at 1:20 AM local. (Fixed: Rule B)
  2. Old clamp at -1.0σ = -24.2 dB, above WSPR -28 dB threshold.
     Display showed "WSPR" instead of "—" (closed). (Fixed: -2.0σ clamp)
  3. Low bands (160m/80m/40m) showing "CW"/"FT8" for DX paths at midday.
     D-layer absorption at 1/f² makes this physically impossible beyond
     NVIS range (~1500 km). (Fixed: Rule C)

Override rules:
  Rule A: freq >= 21 MHz AND tx_solar < -6° AND rx_solar < -6° → clamp
  Rule B: freq >= 21 MHz AND tx_solar < -18° → clamp
  Rule C: freq <= 7.5 MHz AND tx_solar > 0° AND rx_solar > 0°
          AND distance > 1500 km → clamp
  CLAMP = -2.0σ (≈ -30.9 dB, below WSPR -28 dB decode floor)

Seven test categories:
  1. TX-Dark Closure — TX deep night, RX daylight (Rule B defect)
  2. Both-Dark Closure — both endpoints in darkness (Rule A)
  3. Should-Be-Open — both endpoints in daylight, high bands, no override
  4. Greyline Edge — one endpoint in twilight, no override
  5. Low-Band Guard — short-range/nighttime low bands, no override
  6. D-Layer Daytime — low bands, both daylight, DX distance → override fires
  7. (Reserved for KI7MT regression — run separately via validate_physics_override.py)

Usage:
    python tools/test_override_global.py
    python tools/test_override_global.py --verbose

Exit code:
    0 = all tests passed
    1 = failures detected
"""

import os
import sys

# ── Path Setup ────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(SCRIPT_DIR)
COMMON_DIR = os.path.join(TRAINING_DIR, "versions", "common")
sys.path.insert(0, COMMON_DIR)

from model import solar_elevation_deg, grid4_to_latlon, haversine_km, BAND_FREQ_HZ
from physics_override import (
    apply_override_to_prediction,
    FREQ_THRESHOLD_MHZ,
    LOW_FREQ_THRESHOLD_MHZ,
    DLAYER_DISTANCE_KM,
    CLAMP_SIGMA,
)


# ── Constants ─────────────────────────────────────────────────────────────────

# V22-gamma normalization: mean = -17.53 dB, std = 6.70 dB
WSPR_MEAN_DB = -17.53
WSPR_STD_DB = 6.70
WSPR_DECODE_FLOOR_DB = -28.0

def sigma_to_db(sigma):
    return sigma * WSPR_STD_DB + WSPR_MEAN_DB

def db_to_sigma(db):
    return (db - WSPR_MEAN_DB) / WSPR_STD_DB


# ── Test Grid Definitions ────────────────────────────────────────────────────

GRIDS = {
    "BP51": ("Alaska", 61.5, -149.0),
    "KP20": ("Finland", 60.5, 25.0),
    "DN13": ("Idaho", 43.5, -117.0),
    "FN20": ("NE US", 40.5, -75.0),
    "IO91": ("England", 51.5, -1.0),
    "JN48": ("C. Europe", 48.5, 9.0),
    "EL96": ("Florida", 26.5, -81.0),
    "PM95": ("Japan", 35.5, 139.0),
    "OK03": ("Thailand", 13.5, 101.0),
    "GH64": ("Brazil", -15.5, -47.0),
    "KG33": ("S. Africa", -26.5, 27.0),
    "QF56": ("Australia", -33.5, 151.0),
}


# ── Helper: Build a test case ────────────────────────────────────────────────

def make_test(test_id, category, tx_grid, rx_grid, band, hour_utc, month,
              day_of_year, sfi, kp, expect_override, expect_closed, physics):
    """Build a test case dict with computed solar elevations and distance."""
    tx_lat, tx_lon = grid4_to_latlon(tx_grid)
    rx_lat, rx_lon = grid4_to_latlon(rx_grid)
    tx_solar = solar_elevation_deg(tx_lat, tx_lon, hour_utc, day_of_year)
    rx_solar = solar_elevation_deg(rx_lat, rx_lon, hour_utc, day_of_year)
    freq_mhz = BAND_FREQ_HZ.get(band, 0) / 1e6
    distance_km = haversine_km(tx_lat, tx_lon, rx_lat, rx_lon)

    return {
        "id": test_id,
        "category": category,
        "tx_grid": tx_grid,
        "rx_grid": rx_grid,
        "band": band,
        "freq_mhz": freq_mhz,
        "hour_utc": hour_utc,
        "month": month,
        "day_of_year": day_of_year,
        "sfi": sfi,
        "kp": kp,
        "tx_solar": tx_solar,
        "rx_solar": rx_solar,
        "distance_km": distance_km,
        "expect_override": expect_override,
        "expect_closed": expect_closed,
        "physics": physics,
    }


# ── Category 1: TX-Dark High-Band Closure ────────────────────────────────────
# TX solar < -18° (full night), RX in daylight — override MUST fire

def build_cat1_tests():
    tests = []
    n = 1

    # DN13 (Idaho) night → JN48 (Europe) day, 10m, Feb
    # 09z: Idaho=-52.6°, Europe=+23.9°
    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "DN13", "JN48", "10m",
        9, 2, 57, 120, 3, True, True,
        "TX Idaho at -52.6° (deep night), RX Europe at +23.9° (day). 28 MHz dead above Idaho."))
    n += 1

    # DN13 night → JN48 day, 15m, Feb
    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "DN13", "JN48", "15m",
        9, 2, 57, 120, 3, True, True,
        "TX Idaho at -52.6° (deep night), RX Europe at +23.9° (day). 21 MHz dead above Idaho."))
    n += 1

    # DN13 night → JN48 day, 12m, Feb
    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "DN13", "JN48", "12m",
        9, 2, 57, 120, 3, True, True,
        "TX Idaho at -52.6°, RX Europe +23.9°. 24 MHz dead above Idaho."))
    n += 1

    # DN13 night → IO91 (England) day, 10m
    # 09z: Idaho=-52.6°, England=+17.3°
    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "DN13", "IO91", "10m",
        9, 2, 57, 150, 2, True, True,
        "TX Idaho at -52.6°, RX England at +17.3°. 28 MHz dead."))
    n += 1

    # FN20 (NE US) night → JN48 day, 10m
    # 06z: NEUS=-56.3°, Europe=-1.2° (just below horizon)
    # Use 07z for better contrast: compute inline
    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "FN20", "JN48", "10m",
        6, 2, 57, 120, 3, True, True,
        "TX NE US at -56.3° (deep night), RX Europe at -1.2° (near horizon). TX has no F-layer."))
    n += 1

    # FN20 night → IO91 day, 15m
    # 06z: NEUS=-56.3°, England=-8.0°
    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "FN20", "IO91", "15m",
        6, 2, 57, 120, 3, True, True,
        "TX NE US at -56.3° (deep night), RX England at -8.0°. TX has no F-layer for 21 MHz."))
    n += 1

    # EL96 (Florida) night → KG33 (S. Africa) day, 10m
    # 06z: Florida=-71.0°, S.Africa=+28.3°
    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "EL96", "KG33", "10m",
        6, 2, 57, 130, 2, True, True,
        "TX Florida at -71.0° (deep night), RX S.Africa at +28.3° (day). No F-layer above Florida."))
    n += 1

    # EL96 night → KG33 day, 15m
    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "EL96", "KG33", "15m",
        6, 2, 57, 130, 2, True, True,
        "TX Florida at -71.0°, RX S.Africa at +28.3°. 21 MHz dead above Florida."))
    n += 1

    # BP51 (Alaska) night → KP20 (Finland) day, 10m
    # 06z: Alaska=-22.8°, Finland=+3.5°
    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "BP51", "KP20", "10m",
        6, 2, 57, 120, 2, True, True,
        "TX Alaska at -22.8° (astronomical twilight), RX Finland at +3.5°. No F-layer above Alaska."))
    n += 1

    # PM95 (Japan) night → JN48 (Europe) day, 10m
    # 15z: Japan=-63.7°, Europe=+15.1°
    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "PM95", "JN48", "10m",
        15, 2, 57, 130, 2, True, True,
        "TX Japan at -63.7° (deep night), RX Europe at +15.1° (day). 28 MHz dead."))
    n += 1

    # PM95 night → JN48 day, 15m
    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "PM95", "JN48", "15m",
        15, 2, 57, 130, 2, True, True,
        "TX Japan at -63.7°, RX Europe at +15.1°. 21 MHz dead above Japan."))
    n += 1

    # QF56 (Australia) night → IO91 (England) day, 10m
    # 12z: Australia=-38.9°, England=+29.0°
    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "QF56", "IO91", "10m",
        12, 2, 57, 120, 2, True, True,
        "TX Australia at -38.9° (deep night), RX England at +29.0° (day). 28 MHz dead."))
    n += 1

    # RX-dark tests — TX in daylight, RX in deep night.
    # Rule B (TX-only) does NOT fire here. Multi-hop paths carry the signal
    # across sunlit ionosphere to the dark RX. Proven by KI7MT QSO data:
    # 301 QSOs 15m JA at 17z, 568 QSOs 15m EU at 18z Nov.
    # These are regression guards — override must NOT fire.
    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "JN48", "DN13", "10m",
        9, 2, 57, 120, 3, False, False,
        "TX Europe +23.9° (day), RX Idaho -52.6° (deep night). Multi-hop carries signal to dark RX."))
    n += 1

    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "KG33", "EL96", "15m",
        6, 2, 57, 130, 2, False, False,
        "TX S.Africa +28.3° (day), RX Florida -71.0° (deep night). Signal travels via sunlit hops."))
    n += 1

    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "KP20", "BP51", "10m",
        6, 2, 57, 120, 2, False, False,
        "TX Finland +3.5° (day), RX Alaska -22.8° (deep night). Multi-hop path, override does NOT fire."))
    n += 1

    # GH64 (Brazil) night → IO91 (England) day, 10m
    # 03z: Brazil=-64.9°, England=-34.8°
    # Both deep — this is really cat2. Let me use 00z: Brazil=-37.2°, England=-48.0°
    # Use a different pair. GH64→KG33 at 00z: Brazil=-37.2°, S.Africa=-45.5° — both dark.
    # Instead: GH64 night → JN48 day at 09z: Brazil=+0.6°, Europe=+23.9°
    # Brazil barely above horizon at 09z... not deep night.
    # Use 03z: Brazil=-64.9°, try KG33 at 03z: SA=-11.5° — not day enough
    # Use OK03 (Thailand) night → KG33 (S.Africa) day
    # 18z: Thailand=-78.5°, S.Africa=-19.1° — both dark
    # Use OK03→JN48 at 18z: Thailand=-78.5°, Europe=-13.1° — both negative
    # 21z: Thailand=-35.1°, Europe=-40.7° — both dark
    # Hard to find TX=deep_night + RX=day for these pairs in Feb.
    # Add more hour sweeps for Idaho→Europe
    # DN13→JN48 at 06z: Idaho=-48.7°, Europe=-1.2°
    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "DN13", "JN48", "10m",
        6, 2, 57, 120, 3, True, True,
        "TX Idaho at -48.7° (deep night), RX Europe at -1.2° (near horizon). 28 MHz dead above Idaho."))
    n += 1

    # DN13→IO91 at 06z: Idaho=-48.7°, England=-8.0°
    # England at -8° is deeper than -6° twilight but not deep dark.
    # The NEW rule: tx_solar < -18° fires regardless of RX. So override fires.
    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "DN13", "IO91", "10m",
        6, 2, 57, 120, 3, True, True,
        "TX Idaho at -48.7° (deep night), RX England at -8.0°. TX has no F-layer, 28 MHz dead."))
    n += 1

    # PM95 (Japan) night → KG33 (S.Africa) day, 10m
    # 18z: Japan=-38.5°, S.Africa=-19.1° — both negative, this is cat2
    # 15z: Japan=-63.7°, S.Africa=+20.3°
    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "PM95", "KG33", "10m",
        15, 2, 57, 130, 2, True, True,
        "TX Japan at -63.7° (deep night), RX S.Africa at +20.3° (day). 28 MHz dead above Japan."))
    n += 1

    # OK03 (Thailand) night → IO91 (England) day, 10m
    # 21z: Thailand=-35.1°, England=-33.8° — both dark
    # 18z: Thailand=-78.5°, England=-6.8° — TX deep dark, England barely past -6°
    # New rule fires on TX < -18° regardless, so this fires
    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "OK03", "IO91", "10m",
        18, 2, 57, 120, 2, True, True,
        "TX Thailand at -78.5° (deep night), RX England at -6.8°. TX has no F-layer."))
    n += 1

    # October tests (different season)
    # DN13→JN48 at 09z Oct (day 288): compute fresh
    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "DN13", "JN48", "10m",
        9, 10, 288, 120, 3, True, True,
        "TX Idaho deep night, RX Europe daytime. October. 28 MHz dead above Idaho."))
    n += 1

    # DN13→JN48 at 09z June (day 172) — summer, shorter nights but still deep
    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "DN13", "JN48", "10m",
        9, 6, 172, 100, 2, True, True,
        "TX Idaho deep night, RX Europe daytime. June. F-layer gone above Idaho."))
    n += 1

    # QF56 (Australia) night → GH64 (Brazil) day, 10m
    # 15z: Australia=-44.4°, Brazil undetermined...
    # 12z: Australia=-38.9°, Brazil=+43.8°
    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "QF56", "GH64", "10m",
        12, 2, 57, 120, 2, True, True,
        "TX Australia deep night, RX Brazil daytime. 28 MHz dead."))
    n += 1

    # BP51 (Alaska) deep night → EL96 (Florida) day, 15m
    # 06z: Alaska=-22.8°, Florida=-71.0° — both dark
    # 09z: Alaska=-37.0°, Florida=-36.3° — both dark
    # 12z: Alaska=-33.3°, Florida=+3.7°
    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "BP51", "EL96", "15m",
        12, 2, 57, 120, 2, True, True,
        "TX Alaska at -33.3° (deep night), RX Florida at +3.7°. 21 MHz dead above Alaska."))
    n += 1

    # DN13 night → PM95 (Japan) day, 10m
    # 03z: Idaho=-19.6°, Japan=+44.8°
    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "DN13", "PM95", "10m",
        3, 2, 57, 130, 2, True, True,
        "TX Idaho at -19.6° (past astronomical twilight), RX Japan at +44.8° (bright day). 28 MHz dead."))
    n += 1

    # DN13 night → QF56 (Australia) day, 15m
    # 03z: Idaho=-19.6°, Australia=+61.8°
    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "DN13", "QF56", "15m",
        3, 2, 57, 120, 2, True, True,
        "TX Idaho at -19.6°, RX Australia at +61.8° (bright day). 21 MHz dead above Idaho."))
    n += 1

    # KP20 (Finland) deep night → OK03 (Thailand) day, 10m
    # 00z: Finland=-35.7°, Thailand=+8.3°
    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "KP20", "OK03", "10m",
        0, 2, 57, 120, 2, True, True,
        "TX Finland at -35.7° (deep night), RX Thailand at +8.3° (day). 28 MHz dead above Finland."))
    n += 1

    # FN20 (NE US) deep night → GH64 (Brazil) day, 10m
    # 09z: NEUS=-28.8°, Brazil=+0.6°
    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "FN20", "GH64", "10m",
        9, 2, 57, 120, 2, True, True,
        "TX NE US at -28.8° (deep night), RX Brazil at +0.6° (horizon). 28 MHz dead above NE US."))
    n += 1

    # December tests
    # DN13→JN48 at 09z Dec (day 355)
    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "DN13", "JN48", "10m",
        9, 12, 355, 150, 2, True, True,
        "TX Idaho deep night, RX Europe daytime. December. 28 MHz dead above Idaho."))
    n += 1

    # FN20→IO91 at 06z Dec, 15m
    tests.append(make_test(
        f"OVR-{n:03d}", "tx_dark_closure", "FN20", "IO91", "15m",
        6, 12, 355, 150, 2, True, True,
        "TX NE US deep night, RX England near horizon. December. 21 MHz dead above NE US."))
    n += 1

    return tests


# ── Category 2: Both-Dark High-Band Closure ──────────────────────────────────
# TX solar < -6° AND RX solar < -6° — existing rule, verify clamp value

def build_cat2_tests():
    tests = []
    n = 1

    # DN13→JN48 at 03z Feb: Idaho=-19.6°, Europe=-30.5°
    tests.append(make_test(
        f"OVR-B{n:03d}", "both_dark_closure", "DN13", "JN48", "10m",
        3, 2, 57, 120, 3, True, True,
        "Both endpoints in darkness. Idaho -19.6°, Europe -30.5°. F-layer gone at both ends."))
    n += 1

    tests.append(make_test(
        f"OVR-B{n:03d}", "both_dark_closure", "DN13", "JN48", "15m",
        3, 2, 57, 120, 3, True, True,
        "Both dark, 15m. Same conditions as above."))
    n += 1

    tests.append(make_test(
        f"OVR-B{n:03d}", "both_dark_closure", "DN13", "JN48", "12m",
        3, 2, 57, 120, 3, True, True,
        "Both dark, 12m."))
    n += 1

    # FN20→IO91 at 00z: NEUS=-17.5°, England=-48.0°
    tests.append(make_test(
        f"OVR-B{n:03d}", "both_dark_closure", "FN20", "IO91", "10m",
        0, 2, 57, 120, 2, True, True,
        "NE US -17.5°, England -48.0°. Both in darkness."))
    n += 1

    tests.append(make_test(
        f"OVR-B{n:03d}", "both_dark_closure", "FN20", "IO91", "15m",
        0, 2, 57, 120, 2, True, True,
        "NE US -17.5°, England -48.0°. Both dark, 15m."))
    n += 1

    # BP51→KP20 at 03z: Alaska=-1.8°, Finland=-18.047°
    # Alaska at -1.8° is NOT below -6° → Rule A won't fire.
    # Rule B is TX-only: Alaska (TX) at -1.8° is NOT below -18° → Rule B won't fire.
    # Override does NOT fire. This is a greyline edge case, moved here for coverage.
    tests.append(make_test(
        f"OVR-B{n:03d}", "both_dark_closure", "BP51", "KP20", "10m",
        3, 2, 57, 120, 2, False, False,
        "Alaska -1.8° (TX, above all thresholds), Finland -18.0° (RX). TX-only Rule B: no fire."))
    n += 1

    # PM95→QF56 at 18z: Japan=-38.5°, Australia=-17.9°
    tests.append(make_test(
        f"OVR-B{n:03d}", "both_dark_closure", "PM95", "QF56", "10m",
        18, 2, 57, 120, 2, True, True,
        "Japan -38.5°, Australia -17.9°. Both below -6°, Rule A fires."))
    n += 1

    tests.append(make_test(
        f"OVR-B{n:03d}", "both_dark_closure", "PM95", "QF56", "15m",
        18, 2, 57, 120, 2, True, True,
        "Japan -38.5°, Australia -17.9°. Both dark, 15m."))
    n += 1

    # OK03→KP20 at 21z: Thailand=-35.1°, Finland=-36.9°
    tests.append(make_test(
        f"OVR-B{n:03d}", "both_dark_closure", "OK03", "KP20", "10m",
        21, 2, 57, 120, 2, True, True,
        "Thailand -35.1°, Finland -36.9°. Both deep dark."))
    n += 1

    # EL96→GH64 at 03z: Florida=-52.0°, Brazil=-64.9°
    tests.append(make_test(
        f"OVR-B{n:03d}", "both_dark_closure", "EL96", "GH64", "10m",
        3, 2, 57, 120, 2, True, True,
        "Florida -52.0°, Brazil -64.9°. Both deep dark."))
    n += 1

    # KI7MT acid test conditions: DN46→JN48 at 02z Feb
    # Use DN13 (close to DN46): Idaho at 02z ≈ similar
    tests.append(make_test(
        f"OVR-B{n:03d}", "both_dark_closure", "DN13", "JN48", "10m",
        2, 2, 45, 150, 2, True, True,
        "Acid test analog: Idaho deep night, Europe deep night. 02z Feb."))
    n += 1

    # December both-dark
    tests.append(make_test(
        f"OVR-B{n:03d}", "both_dark_closure", "FN20", "JN48", "10m",
        3, 12, 355, 150, 2, True, True,
        "NE US and Europe both in darkness, December, 10m."))
    n += 1

    # KG33→GH64 at 00z: S.Africa=-45.5°, Brazil=-37.2°
    tests.append(make_test(
        f"OVR-B{n:03d}", "both_dark_closure", "KG33", "GH64", "10m",
        0, 2, 57, 120, 2, True, True,
        "S.Africa -45.5°, Brazil -37.2°. Both deep dark."))
    n += 1

    # High SFI — override must fire regardless of solar activity
    tests.append(make_test(
        f"OVR-B{n:03d}", "both_dark_closure", "DN13", "JN48", "10m",
        3, 2, 57, 250, 1, True, True,
        "Both dark, very high SFI=250. Override fires — SFI irrelevant when F-layer is gone."))
    n += 1

    # Low Kp storm — override must fire regardless
    tests.append(make_test(
        f"OVR-B{n:03d}", "both_dark_closure", "DN13", "JN48", "15m",
        3, 2, 57, 120, 8, True, True,
        "Both dark, geomagnetic storm Kp=8. Override fires — no F-layer regardless."))
    n += 1

    return tests


# ── Category 3: Should-Be-Open (regression guard) ────────────────────────────
# Both endpoints well in daylight — override must NOT fire

def build_cat3_tests():
    tests = []
    n = 1

    # DN13→JN48 at 18z: Idaho=+31.6°, Europe=-13.1° — Europe is dark!
    # Use 15z: Idaho=+6.2°, Europe=+15.1° — both positive
    tests.append(make_test(
        f"OVR-O{n:03d}", "should_be_open", "DN13", "JN48", "10m",
        15, 2, 57, 150, 2, False, False,
        "Both in daylight. Idaho +6.2°, Europe +15.1°. 10m should be open."))
    n += 1

    tests.append(make_test(
        f"OVR-O{n:03d}", "should_be_open", "DN13", "JN48", "15m",
        15, 2, 57, 150, 2, False, False,
        "Both daylight, 15m open."))
    n += 1

    tests.append(make_test(
        f"OVR-O{n:03d}", "should_be_open", "DN13", "JN48", "20m",
        15, 2, 57, 150, 2, False, False,
        "Both daylight, 20m. Override should never touch 20m (below 21 MHz)."))
    n += 1

    # FN20→IO91 at 12z: NEUS=+5.0°, England=+29.0°
    tests.append(make_test(
        f"OVR-O{n:03d}", "should_be_open", "FN20", "IO91", "10m",
        12, 2, 57, 150, 2, False, False,
        "NE US +5.0°, England +29.0°. Both daylight."))
    n += 1

    tests.append(make_test(
        f"OVR-O{n:03d}", "should_be_open", "FN20", "IO91", "15m",
        12, 2, 57, 150, 2, False, False,
        "NE US +5.0°, England +29.0°. 15m open."))
    n += 1

    # EL96→GH64 at 15z: Florida=+39.8°, Brazil=+83.7°
    tests.append(make_test(
        f"OVR-O{n:03d}", "should_be_open", "EL96", "GH64", "10m",
        15, 2, 57, 150, 2, False, False,
        "Florida +39.8°, Brazil +83.7°. Both bright daylight."))
    n += 1

    # PM95→OK03 at 03z: Japan=+44.8°, Thailand=+49.2°
    tests.append(make_test(
        f"OVR-O{n:03d}", "should_be_open", "PM95", "OK03", "10m",
        3, 2, 57, 150, 2, False, False,
        "Japan +44.8°, Thailand +49.2°. Both bright."))
    n += 1

    tests.append(make_test(
        f"OVR-O{n:03d}", "should_be_open", "PM95", "OK03", "15m",
        3, 2, 57, 150, 2, False, False,
        "Japan +44.8°, Thailand +49.2°. 15m open."))
    n += 1

    # QF56→KG33 at 06z: Australia=+29.3°, S.Africa=+28.3°
    tests.append(make_test(
        f"OVR-O{n:03d}", "should_be_open", "QF56", "KG33", "10m",
        6, 2, 57, 130, 2, False, False,
        "Australia +29.3°, S.Africa +28.3°. Both in full daylight."))
    n += 1

    # BP51→QF56 at 00z: Alaska=+15.0°, Australia=+54.1°
    tests.append(make_test(
        f"OVR-O{n:03d}", "should_be_open", "BP51", "QF56", "10m",
        0, 2, 57, 130, 2, False, False,
        "Alaska +15.0°, Australia +54.1°. Both daylight."))
    n += 1

    tests.append(make_test(
        f"OVR-O{n:03d}", "should_be_open", "BP51", "QF56", "15m",
        0, 2, 57, 130, 2, False, False,
        "Alaska +15.0°, Australia +54.1°. 15m open."))
    n += 1

    # October daytime tests
    # DN13→FN20 at 18z Oct: both in daylight in NA
    tests.append(make_test(
        f"OVR-O{n:03d}", "should_be_open", "DN13", "FN20", "10m",
        18, 10, 288, 150, 2, False, False,
        "Idaho→NE US, both afternoon daylight. October. 10m open."))
    n += 1

    # KG33→GH64 at 12z: S.Africa daytime, Brazil daytime
    tests.append(make_test(
        f"OVR-O{n:03d}", "should_be_open", "KG33", "GH64", "10m",
        12, 2, 57, 130, 2, False, False,
        "S.Africa→Brazil, both in daylight at 12z."))
    n += 1

    # DN13→IO91 at 15z: Idaho +6.2°, England +18.2°
    tests.append(make_test(
        f"OVR-O{n:03d}", "should_be_open", "DN13", "IO91", "10m",
        15, 2, 57, 150, 2, False, False,
        "Idaho +6.2°, England +18.2°. Both daylight. 10m should be open."))
    n += 1

    # December daytime
    # DN13→FN20 at 18z Dec (day 355)
    tests.append(make_test(
        f"OVR-O{n:03d}", "should_be_open", "DN13", "FN20", "10m",
        18, 12, 355, 150, 2, False, False,
        "Idaho→NE US, afternoon December. Both in daylight."))
    n += 1

    # High SFI, daylight — must not fire
    tests.append(make_test(
        f"OVR-O{n:03d}", "should_be_open", "EL96", "KG33", "10m",
        12, 2, 57, 250, 1, False, False,
        "Florida→S.Africa, high SFI, both in daylight. Override must not fire."))
    n += 1

    # Storm conditions but daylight — must not fire
    tests.append(make_test(
        f"OVR-O{n:03d}", "should_be_open", "EL96", "KG33", "15m",
        12, 2, 57, 120, 8, False, False,
        "Florida→S.Africa, Kp=8 storm, both daylight. Override must not fire (storm != darkness)."))
    n += 1

    # PM95→KG33 at 06z: Japan=+25.5°, S.Africa=+28.3°
    tests.append(make_test(
        f"OVR-O{n:03d}", "should_be_open", "PM95", "KG33", "10m",
        6, 2, 57, 130, 2, False, False,
        "Japan +25.5°, S.Africa +28.3°. Both daylight."))
    n += 1

    # OK03→KG33 at 06z: Thailand=+64.5°, S.Africa=+28.3°
    tests.append(make_test(
        f"OVR-O{n:03d}", "should_be_open", "OK03", "KG33", "15m",
        6, 2, 57, 130, 2, False, False,
        "Thailand +64.5°, S.Africa +28.3°. Both bright daylight."))
    n += 1

    # Equatorial path, both bright
    tests.append(make_test(
        f"OVR-O{n:03d}", "should_be_open", "OK03", "GH64", "10m",
        12, 2, 57, 130, 2, False, False,
        "Thailand→Brazil equatorial path. Both in daylight at 12z."))
    n += 1

    return tests


# ── Category 4: Greyline / Twilight Edge Cases ──────────────────────────────
# One endpoint between -6° and 0° — override should NOT fire (greyline possible)
# Neither endpoint below -18° in these tests

def build_cat4_tests():
    tests = []
    n = 1

    # BP51 at 03z: Alaska=-1.8° (just below horizon, greyline)
    # KP20 at 03z: Finland=-18.047° — just past -18°, Rule B fires correctly.
    # Moved to cat2 (both_dark_closure). Replaced with a true boundary case.

    # BP51 at 15z: Alaska=-15.0° (between -6° and -18°)
    # FN20 at 15z: NE US=+32.8° (daylight)
    # Alaska at -15° is not below -18° → Rule B doesn't fire. Not both < -6° → Rule A doesn't fire.
    tests.append(make_test(
        f"OVR-G{n:03d}", "greyline_edge", "BP51", "FN20", "10m",
        15, 2, 57, 120, 2, False, False,
        "Alaska -15.0° (nautical twilight, not deep dark), NE US +32.8° (day). F2 residual present above Alaska."))
    n += 1

    # DN13 at 15z: Idaho=+6.2° (low but daylight)
    # JN48 at 15z: Europe=+15.1°
    # Both positive — covered in cat3. Let me find greyline cases.

    # IO91 at 18z: England=-6.8° (just past -6°)
    # DN13 at 18z: Idaho=+31.6°
    # England at -6.8° — not below -18°, so Rule B doesn't fire.
    # Rule A needs both < -6°, Idaho is +31.6° so Rule A doesn't fire.
    # Override should NOT fire.
    tests.append(make_test(
        f"OVR-G{n:03d}", "greyline_edge", "DN13", "IO91", "10m",
        18, 2, 57, 120, 2, False, False,
        "Idaho +31.6° (day), England -6.8° (civil twilight). Neither triggers override."))
    n += 1

    # KP20 at 06z: Finland=+3.5° (just above horizon)
    # DN13 at 06z: Idaho=-48.7° — TX is deep dark! Rule B fires.
    # This is actually cat1. Need a case where the "dark" side is -6° to -18°.

    # QF56 at 21z: Australia=+18.5°
    # PM95 at 21z: Japan=-2.3° (just below horizon, greyline)
    # Japan at -2.3° — not below -6° for Rule A, not below -18° for Rule B.
    tests.append(make_test(
        f"OVR-G{n:03d}", "greyline_edge", "QF56", "PM95", "10m",
        21, 2, 57, 130, 2, False, False,
        "Australia +18.5° (day), Japan -2.3° (sunset/greyline). Override must NOT fire."))
    n += 1

    # KG33 at 18z: S.Africa=-19.1° — this IS < -18°! Rule B fires. Not greyline.
    # KG33 at 15z: S.Africa=+20.3° — daylight
    # Need S.Africa in twilight...
    # KG33 at 17z (interpolating): between +20.3° and -19.1°
    # Let me compute for 17z
    tests.append(make_test(
        f"OVR-G{n:03d}", "greyline_edge", "KG33", "GH64", "15m",
        17, 2, 57, 120, 2, False, False,
        "S.Africa near sunset, Brazil in daylight. Greyline conditions."))
    n += 1

    # GH64 at 21z: Brazil=+4.4° (low sun)
    # IO91 at 21z: England=-33.8° — deep dark. Rule B fires.
    # Not a greyline test. Need RX not below -18°.

    # GH64 at 09z: Brazil=+0.6° (just above horizon)
    # KG33 at 09z: S.Africa=+65.9° (bright day)
    # Both positive — not twilight.

    # EL96 at 00z: Florida=-12.2° (nautical twilight)
    # GH64 at 00z: Brazil=-37.2° — deep dark. Rule B fires on Brazil.
    # Not a pure greyline test.

    # EL96 at 12z: Florida=+3.7° (just after sunrise)
    # IO91 at 12z: England=+29.0°
    # Both positive, cat3.

    # DN13 at 00z: Idaho=+12.2° (still positive!) — not twilight at midnight?
    # Wait, Idaho at 00z UTC = 5 PM local. Makes sense.
    # DN13 at 01z: need to compute

    # PM95 at 09z: Japan=-8.7° (between -6° and -18°, nautical twilight)
    # DN13 at 09z: Idaho=-52.6° — deep dark. Rule B fires on Idaho.
    # We need a case where the "deepest" endpoint is between -6° and -17.9°

    # OK03 at 12z: Thailand=-12.8° (nautical twilight, not deep dark)
    # JN48 at 12z: Europe=+31.5° (bright day)
    # Thailand at -12.8° — not below -18° for Rule B. Europe not below -6° for Rule A.
    # Override should NOT fire.
    tests.append(make_test(
        f"OVR-G{n:03d}", "greyline_edge", "OK03", "JN48", "10m",
        12, 2, 57, 120, 2, False, False,
        "Thailand -12.8° (nautical twilight), Europe +31.5° (day). Neither triggers override — greyline possible."))
    n += 1

    # PM95 at 09z: Japan=-8.7° (nautical twilight)
    # OK03 at 09z: Thailand=+29.9° (day)
    tests.append(make_test(
        f"OVR-G{n:03d}", "greyline_edge", "PM95", "OK03", "10m",
        9, 2, 57, 130, 2, False, False,
        "Japan -8.7° (nautical twilight), Thailand +29.9° (day). F2 residual possible."))
    n += 1

    # GH64 at 09z: Brazil=+0.6° (sunrise)
    # EL96 at 09z: Florida=-36.3° — deep dark. Rule B fires.
    # Not greyline.

    # BP51 at 15z: Alaska=-15.0° (between -6° and -18°)
    # DN13 at 15z: Idaho=+6.2°
    tests.append(make_test(
        f"OVR-G{n:03d}", "greyline_edge", "BP51", "DN13", "10m",
        15, 2, 57, 120, 2, False, False,
        "Alaska -15.0° (between -6° and -18°), Idaho +6.2°. Not deep enough for Rule B."))
    n += 1

    # FN20 at 00z: NEUS=-17.5° (just above -18° threshold!)
    # IO91 at 00z: England=-48.0° — deep dark. Rule B fires on England.
    # Not a clean greyline test.

    # QF56 at 18z: Australia=-17.9° (just above -18°!)
    # PM95 at 18z: Japan=-38.5° — deep dark. Rule B fires on Japan.
    # Actually this is an interesting boundary test:
    # Australia at -17.9° is NOT < -18°, but Japan at -38.5° IS < -18°.
    # Rule B fires because Japan < -18°. So override fires. Not greyline.

    # Need: both endpoints have solar > -18° AND at least one < 0°

    # KP20 at 15z: Finland=+1.3° (just above horizon)
    # DN13 at 15z: Idaho=+6.2°
    tests.append(make_test(
        f"OVR-G{n:03d}", "greyline_edge", "KP20", "DN13", "10m",
        15, 2, 57, 120, 2, False, False,
        "Finland +1.3° (low sun), Idaho +6.2°. Both just in daylight, no override."))
    n += 1

    # IO91 at 15z: England=+18.2°
    # PM95 at 15z: Japan=-63.7° — deep dark. Rule B fires.
    # Not suitable.

    # EL96 at 00z: Florida=-12.2° (nautical twilight, not deep dark)
    # DN13 at 00z: Idaho=+12.2°
    tests.append(make_test(
        f"OVR-G{n:03d}", "greyline_edge", "EL96", "DN13", "15m",
        0, 2, 57, 120, 2, False, False,
        "Florida -12.2° (nautical twilight), Idaho +12.2° (day). F-layer residual present."))
    n += 1

    return tests


# ── Category 5: Low-Band Guard (override must NOT fire) ──────────────────────
# Cases where low bands should NOT be overridden:
# - Nighttime (low bands propagate best at night — no D-layer)
# - TX dark, RX daylight (greyline/one-sided darkness)
# - Short range daytime (NVIS, < 1500 km — D-layer rule doesn't apply)
# - 20m/17m in any conditions (between high-band and low-band thresholds)

def build_cat5_tests():
    tests = []
    n = 1

    bands_low = ["40m", "80m", "160m"]

    # Both dark (deep night) — low bands propagate best at night
    for band in bands_low:
        tests.append(make_test(
            f"OVR-L{n:03d}", "low_band_guard", "DN13", "JN48", band,
            3, 2, 57, 120, 3, False, False,
            f"Both endpoints dark, {band}. D-layer gone at night — low bands propagate."))
        n += 1

    # TX deep dark, RX daylight — greyline paths, no override
    tests.append(make_test(
        f"OVR-L{n:03d}", "low_band_guard", "DN13", "JN48", "40m",
        9, 2, 57, 120, 3, False, False,
        "TX deep dark, RX daylight, 40m. Only one endpoint in D-layer — greyline preserved."))
    n += 1

    tests.append(make_test(
        f"OVR-L{n:03d}", "low_band_guard", "DN13", "JN48", "80m",
        9, 2, 57, 120, 3, False, False,
        "TX deep dark, RX daylight, 80m. Greyline/one-sided darkness preserved."))
    n += 1

    tests.append(make_test(
        f"OVR-L{n:03d}", "low_band_guard", "DN13", "JN48", "160m",
        9, 2, 57, 120, 3, False, False,
        "TX deep dark, RX daylight, 160m. Greyline preserved."))
    n += 1

    # Short range daytime — NVIS and ground-wave, below 1500 km
    # DN13→FN20 is ~3000 km, too far. Use closer grids.
    # DN13 (Idaho) → EL96 (Florida) at 18z: both daylight, ~3100 km — too far
    # Use grids within 1500 km: DN13→DM79 (Nevada) ~800 km conceptually
    # Actually we need real grids. DN13→DN65 (Montana) ~500 km
    # grid4_to_latlon may not know arbitrary grids. Use known ones.
    # FN20 (NE US) → EL96 (Florida) at 15z: both daylight, ~2000 km — still > 1500
    # The closest pair in our grid set: PM95 (Japan) → OK03 (Thailand) = ~4600 km
    # All our grid pairs are > 1500 km! So short-range tests need close grids.
    # Use same grid for TX and RX (0 km) or adjacent grids.
    # DN13→DN13 (0 km): trivially < 1500 km
    tests.append(make_test(
        f"OVR-L{n:03d}", "low_band_guard", "DN13", "DN13", "40m",
        18, 2, 57, 120, 2, False, False,
        "Same grid, 0 km. Both daylight, 40m. NVIS — override must NOT fire."))
    n += 1

    tests.append(make_test(
        f"OVR-L{n:03d}", "low_band_guard", "DN13", "DN13", "80m",
        18, 2, 57, 120, 2, False, False,
        "Same grid, 0 km. Both daylight, 80m. Local ground-wave — no override."))
    n += 1

    tests.append(make_test(
        f"OVR-L{n:03d}", "low_band_guard", "DN13", "DN13", "160m",
        18, 2, 57, 120, 2, False, False,
        "Same grid, 0 km. Both daylight, 160m. Ground-wave — no override."))
    n += 1

    # IO91 (England) → JN48 (C. Europe) at 12z: both daylight, ~800 km
    tests.append(make_test(
        f"OVR-L{n:03d}", "low_band_guard", "IO91", "JN48", "40m",
        12, 2, 57, 120, 2, False, False,
        "England→C.Europe ~800 km. Both daylight, 40m. Within NVIS range — no override."))
    n += 1

    tests.append(make_test(
        f"OVR-L{n:03d}", "low_band_guard", "IO91", "JN48", "80m",
        12, 2, 57, 120, 2, False, False,
        "England→C.Europe ~800 km. Both daylight, 80m. Short path — no override."))
    n += 1

    tests.append(make_test(
        f"OVR-L{n:03d}", "low_band_guard", "IO91", "JN48", "160m",
        12, 2, 57, 120, 2, False, False,
        "England→C.Europe ~800 km. Both daylight, 160m. Short path — no override."))
    n += 1

    # 20m — between thresholds (14 MHz: > 7.5 MHz, < 21 MHz) — NEVER overridden
    tests.append(make_test(
        f"OVR-L{n:03d}", "low_band_guard", "DN13", "JN48", "20m",
        3, 2, 57, 120, 3, False, False,
        "Both dark, 20m (14 MHz). Between thresholds — no rule applies."))
    n += 1

    tests.append(make_test(
        f"OVR-L{n:03d}", "low_band_guard", "DN13", "JN48", "20m",
        15, 2, 57, 120, 2, False, False,
        "Both daylight, 20m (14 MHz). Between thresholds — no rule applies."))
    n += 1

    # 17m boundary — 18.1 MHz is < 21 MHz (Rule A/B) and > 7.5 MHz (Rule C)
    tests.append(make_test(
        f"OVR-L{n:03d}", "low_band_guard", "DN13", "JN48", "17m",
        3, 2, 57, 120, 3, False, False,
        "Both dark, 17m (18.1 MHz). Between thresholds — no rule applies."))
    n += 1

    # 30m (10.1 MHz) — above 7.5 MHz, should NOT fire Rule C
    tests.append(make_test(
        f"OVR-L{n:03d}", "low_band_guard", "DN13", "JN48", "30m",
        15, 2, 57, 120, 2, False, False,
        "Both daylight, 30m (10.1 MHz). Above 7.5 MHz — Rule C does not apply."))
    n += 1

    return tests


# ── Category 6: D-Layer Daytime Closure ──────────────────────────────────────
# Low bands (≤ 7.5 MHz), both endpoints in daylight, DX distance (> 1500 km)
# Override MUST fire — D-layer absorption makes propagation impossible

def build_cat6_tests():
    tests = []
    n = 1

    # DN13 (Idaho) → JN48 (Europe), both daylight at 15z, ~8500 km
    for band in ["160m", "80m", "40m"]:
        tests.append(make_test(
            f"OVR-D{n:03d}", "dlayer_daytime", "DN13", "JN48", band,
            15, 2, 57, 120, 2, True, True,
            f"Idaho→Europe ~8500 km, both daylight. {band} D-layer absorption kills DX."))
        n += 1

    # DN13 → PM95 (Japan) at 00z: Idaho=+12.2°, Japan=+82.3°, ~8300 km
    # Wait — at 00z UTC Idaho is 5 PM local (sun still up in Feb?), Japan is noon+
    # Let me use a time when both are in daylight
    # Actually 00z: Idaho at ~17:00 local, in Feb sun sets ~17:30. Marginal.
    # Use 21z: Idaho at ~14:00 local, Japan already past midnight (06:00 next day)
    # Japan at 21z = 06:00 JST, barely sunrise
    # Let me just use times where both are clearly in daylight.
    # FN20 (NE US) → IO91 (England) at 15z: NEUS=+32.8°, England=+18.2°, ~5500 km
    tests.append(make_test(
        f"OVR-D{n:03d}", "dlayer_daytime", "FN20", "IO91", "160m",
        15, 2, 57, 120, 2, True, True,
        "NE US→England ~5500 km, both daylight. 160m D-layer wall — impossible."))
    n += 1

    tests.append(make_test(
        f"OVR-D{n:03d}", "dlayer_daytime", "FN20", "IO91", "80m",
        15, 2, 57, 120, 2, True, True,
        "NE US→England ~5500 km, both daylight. 80m D-layer absorption."))
    n += 1

    tests.append(make_test(
        f"OVR-D{n:03d}", "dlayer_daytime", "FN20", "IO91", "40m",
        15, 2, 57, 120, 2, True, True,
        "NE US→England ~5500 km, both daylight. 40m D-layer kills DX."))
    n += 1

    # EL96 (Florida) → KG33 (S. Africa) at 12z: ~12000 km
    # 12z: Florida=+3.7°, S.Africa=+65.9°
    tests.append(make_test(
        f"OVR-D{n:03d}", "dlayer_daytime", "EL96", "KG33", "160m",
        12, 2, 57, 130, 2, True, True,
        "Florida→S.Africa ~12000 km, both daylight. 160m absolutely dead."))
    n += 1

    tests.append(make_test(
        f"OVR-D{n:03d}", "dlayer_daytime", "EL96", "KG33", "80m",
        12, 2, 57, 130, 2, True, True,
        "Florida→S.Africa ~12000 km, both daylight. 80m D-layer absorption."))
    n += 1

    tests.append(make_test(
        f"OVR-D{n:03d}", "dlayer_daytime", "EL96", "KG33", "40m",
        12, 2, 57, 130, 2, True, True,
        "Florida→S.Africa ~12000 km, both daylight. 40m D-layer kills at this distance."))
    n += 1

    # EL96 (Florida) → GH64 (Brazil) at 15z: ~7000 km
    # 15z: Florida=+39.8°, Brazil=+83.7°
    tests.append(make_test(
        f"OVR-D{n:03d}", "dlayer_daytime", "EL96", "GH64", "40m",
        15, 2, 57, 130, 2, True, True,
        "Florida→Brazil ~7000 km, both bright daylight. 40m D-layer DX dead."))
    n += 1

    # PM95 (Japan) → OK03 (Thailand) at 03z: ~4600 km
    # 03z: Japan=+44.8°, Thailand=+49.2°
    tests.append(make_test(
        f"OVR-D{n:03d}", "dlayer_daytime", "PM95", "OK03", "80m",
        3, 2, 57, 120, 2, True, True,
        "Japan→Thailand ~4600 km, both daylight. 80m D-layer absorption."))
    n += 1

    tests.append(make_test(
        f"OVR-D{n:03d}", "dlayer_daytime", "PM95", "OK03", "160m",
        3, 2, 57, 120, 2, True, True,
        "Japan→Thailand ~4600 km, both daylight. 160m — D-layer wall."))
    n += 1

    # QF56 (Australia) → KG33 (S. Africa) at 06z: ~10800 km
    # 06z: Australia=+29.3°, S.Africa=+28.3°
    tests.append(make_test(
        f"OVR-D{n:03d}", "dlayer_daytime", "QF56", "KG33", "40m",
        6, 2, 57, 130, 2, True, True,
        "Australia→S.Africa ~10800 km, both daylight. 40m D-layer kills."))
    n += 1

    tests.append(make_test(
        f"OVR-D{n:03d}", "dlayer_daytime", "QF56", "KG33", "160m",
        6, 2, 57, 130, 2, True, True,
        "Australia→S.Africa ~10800 km, both daylight. 160m impossible."))
    n += 1

    # KG33 (S. Africa) → GH64 (Brazil) at 12z: ~7800 km
    # 12z: S.Africa=+65.9°, Brazil=+43.8°
    tests.append(make_test(
        f"OVR-D{n:03d}", "dlayer_daytime", "KG33", "GH64", "80m",
        12, 2, 57, 120, 2, True, True,
        "S.Africa→Brazil ~7800 km, both bright. 80m D-layer absorption."))
    n += 1

    # October test — different season
    tests.append(make_test(
        f"OVR-D{n:03d}", "dlayer_daytime", "DN13", "JN48", "80m",
        15, 10, 288, 120, 2, True, True,
        "Idaho→Europe ~8500 km, both daylight. October. 80m D-layer kills."))
    n += 1

    # December test
    tests.append(make_test(
        f"OVR-D{n:03d}", "dlayer_daytime", "FN20", "IO91", "160m",
        15, 12, 355, 150, 2, True, True,
        "NE US→England ~5500 km, both daylight. December. 160m dead."))
    n += 1

    # High SFI — Rule C fires regardless (D-layer gets WORSE with high SFI)
    tests.append(make_test(
        f"OVR-D{n:03d}", "dlayer_daytime", "DN13", "JN48", "160m",
        15, 2, 57, 250, 1, True, True,
        "Idaho→Europe, SFI=250, both daylight. High SFI makes D-layer worse — must fire."))
    n += 1

    # Boundary: just above 1500 km — DN13→FN20 ~3000 km, both daylight at 18z
    tests.append(make_test(
        f"OVR-D{n:03d}", "dlayer_daytime", "DN13", "FN20", "80m",
        18, 2, 57, 120, 2, True, True,
        "Idaho→NE US ~3000 km, both afternoon daylight. 80m D-layer at DX range."))
    n += 1

    # BP51 (Alaska) → EL96 (Florida) at 21z: ~5800 km
    # 21z: Alaska needs to be in daylight... Alaska at 21z in Feb
    # 21z = 12:00 AKST, Alaska sun up in Feb? Barely. Let me check via test.
    # Actually, at 21z in Feb: Alaska at 61.5°N — sunrise ~17z, sunset ~03z (next day)
    # So at 21z Alaska is in daylight. Florida at 21z = 4pm, definitely daylight.
    tests.append(make_test(
        f"OVR-D{n:03d}", "dlayer_daytime", "BP51", "EL96", "40m",
        21, 2, 57, 120, 2, True, True,
        "Alaska→Florida ~5800 km, both daylight. 40m D-layer at DX distance."))
    n += 1

    return tests


# ── Test Runner ──────────────────────────────────────────────────────────────

def run_single_test(test, verbose=False):
    """Run a single override test case. Returns (passed, detail_dict)."""
    # Use a synthetic model prediction above the clamp to test if override fires
    # If we expect the override to fire, feed a prediction above clamp (+1.0σ)
    # If we expect no override, feed same prediction and verify it's untouched
    test_sigma = 1.0  # Positive prediction that should be clamped if override fires

    clamped, was_overridden = apply_override_to_prediction(
        test_sigma, test["freq_mhz"], test["tx_solar"], test["rx_solar"],
        distance_km=test.get("distance_km"))

    # Check override fired correctly
    override_correct = (was_overridden == test["expect_override"])

    # Check closed status (if override fires, output dB must be below -28)
    if test["expect_closed"]:
        output_db = sigma_to_db(clamped)
        closed_correct = output_db < WSPR_DECODE_FLOOR_DB
    else:
        closed_correct = True  # Not expecting closure

    passed = override_correct and closed_correct

    detail = {
        "test_sigma": test_sigma,
        "clamped_sigma": clamped,
        "clamped_db": sigma_to_db(clamped),
        "was_overridden": was_overridden,
        "override_correct": override_correct,
        "closed_correct": closed_correct,
        "tx_solar": test["tx_solar"],
        "rx_solar": test["rx_solar"],
        "distance_km": test.get("distance_km", 0),
    }

    return passed, detail


def run_all_tests(verbose=False):
    """Run the full global test battery."""
    categories = [
        ("TX-Dark Closure", "tx_dark_closure", build_cat1_tests()),
        ("Both-Dark Closure", "both_dark_closure", build_cat2_tests()),
        ("Should-Be-Open", "should_be_open", build_cat3_tests()),
        ("Greyline Edge", "greyline_edge", build_cat4_tests()),
        ("Low-Band Guard", "low_band_guard", build_cat5_tests()),
        ("D-Layer Daytime", "dlayer_daytime", build_cat6_tests()),
    ]

    print("OVERRIDE GLOBAL VALIDATION")
    print("=" * 72)
    print(f"Rules A/B: freq >= {FREQ_THRESHOLD_MHZ} MHz (F-layer collapse)")
    print(f"Rule C:    freq <= {LOW_FREQ_THRESHOLD_MHZ} MHz, dist > {DLAYER_DISTANCE_KM:.0f} km "
          f"(D-layer absorption)")
    print(f"Clamp: {CLAMP_SIGMA}σ ({sigma_to_db(CLAMP_SIGMA):.1f} dB) | "
          f"WSPR floor: {WSPR_DECODE_FLOOR_DB} dB ({db_to_sigma(WSPR_DECODE_FLOOR_DB):+.2f}σ)")
    print()

    total_pass = 0
    total_fail = 0
    total_tests = 0
    all_failures = []

    for cat_name, cat_key, cat_tests in categories:
        cat_pass = 0
        cat_fail = 0
        cat_failures = []

        for test in cat_tests:
            passed, detail = run_single_test(test, verbose)
            if passed:
                cat_pass += 1
            else:
                cat_fail += 1
                cat_failures.append((test, detail))

            if verbose:
                status = "PASS" if passed else "FAIL"
                ovr = "OVR" if detail["was_overridden"] else "   "
                dist = test.get('distance_km', 0)
                print(f"  [{status}] {test['id']:<10s} {test['band']:>4s} "
                      f"{test['tx_grid']}→{test['rx_grid']} "
                      f"TX:{test['tx_solar']:+6.1f}° RX:{test['rx_solar']:+6.1f}° "
                      f"{dist:>6.0f}km "
                      f"{ovr} {detail['clamped_sigma']:+.2f}σ "
                      f"({detail['clamped_db']:+.1f} dB)")

        cat_total = cat_pass + cat_fail
        total_pass += cat_pass
        total_fail += cat_fail
        total_tests += cat_total
        all_failures.extend(cat_failures)

        dots = "." * max(1, 40 - len(cat_name))
        status = "PASS" if cat_fail == 0 else "FAIL"
        print(f"Category: {cat_name} {dots} {cat_pass}/{cat_total} {status}")

    print()
    print("=" * 72)
    print(f"TOTAL: {total_pass}/{total_tests} PASS")

    if all_failures:
        print()
        print("FAILURES:")
        for test, detail in all_failures:
            dist = test.get('distance_km', 0)
            print(f"  {test['id']} ({test['category']}): {test['band']} "
                  f"{test['tx_grid']}→{test['rx_grid']} ({dist:.0f} km)")
            print(f"    TX solar: {detail['tx_solar']:+.1f}°, "
                  f"RX solar: {detail['rx_solar']:+.1f}°")
            print(f"    Override fired: {detail['was_overridden']} "
                  f"(expected: {test['expect_override']})")
            if test['expect_closed']:
                print(f"    Output: {detail['clamped_db']:+.1f} dB "
                      f"(need < {WSPR_DECODE_FLOOR_DB} dB for closed)")
            print(f"    Physics: {test['physics']}")

    return total_fail == 0


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    passed = run_all_tests(verbose=verbose)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
