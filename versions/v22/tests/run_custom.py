#!/usr/bin/env python3
"""
run_custom.py — IONIS V22-gamma Batch Custom Path Tests

Run predictions for a set of user-defined paths from a JSON file.
Applies PhysicsOverrideLayer to all predictions.

Usage:
  python run_custom.py my_paths.json
  ionis-validate custom my_paths.json

JSON format:
  {
    "description": "My 20m paths from KI7MT",
    "conditions": { "sfi": 140, "kp": 1.5 },
    "paths": [
      {
        "tx_grid": "DN26", "rx_grid": "IO91", "band": "20m",
        "hour": 14, "month": 6, "label": "KI7MT to G"
      },
      {
        "tx_grid": "DN26", "rx_grid": "PM95", "band": "20m",
        "hour": 6, "month": 12, "day_of_year": 355,
        "label": "KI7MT to JA", "expect_open": true
      }
    ]
  }

Per-path overrides for sfi/kp/hour/month/day_of_year take precedence.
Optional "expect_open" (bool) triggers pass/fail.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# -- Path Setup ----------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V22_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, V22_DIR)

from model import (
    get_device, load_model,
    grid4_to_latlon, build_features, haversine_km, BAND_FREQ_HZ,
    solar_elevation_deg,
)
from physics_override import apply_override_to_prediction

# -- Constants -----------------------------------------------------------------

WSPR_MEAN_DB = -17.53
WSPR_STD_DB = 6.7
PATH_OPEN_THRESHOLD_SIGMA = -2.5

MODE_THRESHOLDS_DB = {
    "WSPR": -28.0,
    "FT8":  -21.0,
    "CW":   -15.0,
    "RTTY":  -5.0,
    "SSB":    3.0,
}


def sigma_to_approx_db(sigma):
    return sigma * WSPR_STD_DB + WSPR_MEAN_DB


def main():
    parser = argparse.ArgumentParser(
        description="IONIS V22-gamma — Batch custom path tests",
    )
    parser.add_argument("json_file", help="Path to JSON file defining custom paths")
    parser.add_argument("--config", default=None, help="Path to config JSON")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint")

    args = parser.parse_args()

    with open(args.json_file) as f:
        spec = json.load(f)

    description = spec.get("description", "Custom paths")
    defaults = spec.get("conditions", {})
    paths = spec.get("paths", [])

    if not paths:
        print("  ERROR: No paths defined in JSON file", file=sys.stderr)
        return 1

    device = get_device()
    config_path = args.config
    if config_path is None:
        config_path = os.path.join(V22_DIR, "config_v22.json")

    model, config, metadata = load_model(config_path, args.checkpoint, device)

    print()
    print("=" * 70)
    print("  IONIS V22-gamma + PhysicsOverrideLayer — Custom Path Tests")
    print("=" * 70)
    print(f"\n  {description}")
    if defaults:
        parts = [f"{k}={v}" for k, v in defaults.items()]
        print(f"  Default conditions: {', '.join(parts)}")
    print(f"  Paths: {len(paths)}")
    print(f"  Device: {device}")
    print()

    results = []
    for i, p in enumerate(paths):
        tx_grid = p["tx_grid"]
        rx_grid = p["rx_grid"]
        band = p["band"]
        label = p.get("label", f"{tx_grid}->{rx_grid} {band}")

        sfi = p.get("sfi", defaults.get("sfi", 150))
        kp = p.get("kp", defaults.get("kp", 2))
        hour = p.get("hour", defaults.get("hour", 12))
        month = p.get("month", defaults.get("month", 6))
        expect_open = p.get("expect_open", None)

        # day_of_year: per-path > default > estimate from month
        day_of_year = p.get("day_of_year", defaults.get("day_of_year", None))
        if day_of_year is None:
            month_starts = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
            day_of_year = month_starts[month] + 15

        if band not in BAND_FREQ_HZ:
            print(f"  SKIP: Unknown band '{band}' for path '{label}'")
            results.append((label, None, None, None, "SKIP", False))
            continue

        tx_lat, tx_lon = grid4_to_latlon(tx_grid)
        rx_lat, rx_lon = grid4_to_latlon(rx_grid)
        freq_hz = BAND_FREQ_HZ[band]
        freq_mhz = freq_hz / 1e6
        distance_km = haversine_km(tx_lat, tx_lon, rx_lat, rx_lon)

        features = build_features(
            tx_lat, tx_lon, rx_lat, rx_lon,
            freq_hz, sfi, kp, hour, month,
            day_of_year=day_of_year,
            include_solar_depression=True,
        )
        x = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            snr_sigma = model(x).item()

        # Apply PhysicsOverrideLayer
        tx_solar = solar_elevation_deg(tx_lat, tx_lon, hour, day_of_year)
        rx_solar = solar_elevation_deg(rx_lat, rx_lon, hour, day_of_year)
        clamped_sigma, was_overridden = apply_override_to_prediction(
            snr_sigma, freq_mhz, tx_solar, rx_solar, distance_km=distance_km)

        snr_db = sigma_to_approx_db(clamped_sigma)
        is_open = clamped_sigma > PATH_OPEN_THRESHOLD_SIGMA

        if expect_open is not None:
            if expect_open and is_open:
                status = "PASS"
            elif not expect_open and not is_open:
                status = "PASS"
            else:
                status = "FAIL"
        else:
            status = "OPEN" if is_open else "closed"

        results.append((label, clamped_sigma, snr_db, distance_km, status, was_overridden))

    print(f"  {'#':>3s}  {'Label':<30s}  {'SNR':>8s}  {'dB':>7s}  {'km':>8s}  {'Status':>8s}  {'OVR':>4s}")
    print(f"  {'─' * 76}")

    pass_count = 0
    fail_count = 0
    for i, (label, sigma, db, km, status, ovr) in enumerate(results):
        if sigma is None:
            print(f"  {i+1:>3d}  {label:<30s}  {'—':>8s}  {'—':>7s}  {'—':>8s}  {status:>8s}")
        else:
            ovr_str = "YES" if ovr else ""
            print(f"  {i+1:>3d}  {label:<30s}  {sigma:>+8.3f}  {db:>+7.1f}  {km:>8,.0f}  {status:>8s}  {ovr_str:>4s}")

        if status == "PASS":
            pass_count += 1
        elif status == "FAIL":
            fail_count += 1

    print(f"  {'─' * 76}")

    has_expectations = any(r[4] in ("PASS", "FAIL") for r in results)
    if has_expectations:
        total_tested = pass_count + fail_count
        print(f"\n  Expectations: {pass_count}/{total_tested} passed")
        if fail_count > 0:
            print(f"  {fail_count} path(s) did not match expected open/closed state")

    if results and results[0][1] is not None:
        print(f"\n  Mode verdicts for: {results[0][0]}")
        snr_db_first = results[0][2]
        for mode, threshold in MODE_THRESHOLDS_DB.items():
            v = "OPEN" if snr_db_first >= threshold else "closed"
            marker = ">>>" if v == "OPEN" else "   "
            print(f"    {marker} {mode:<5s}  {v:<6s}  (threshold: {threshold:+.0f} dB)")

    print()
    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
