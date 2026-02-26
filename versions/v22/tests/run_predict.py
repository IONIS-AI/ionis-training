#!/usr/bin/env python3
"""
run_predict.py — IONIS V22-gamma Single Path Prediction CLI

Predict the SNR for a single HF propagation path with PhysicsOverrideLayer.

Usage:
  python run_predict.py --tx-grid FN20 --rx-grid IO91 --band 20m \
      --sfi 150 --kp 2 --hour 14 --month 6 --day-of-year 172

  ionis-validate predict --tx-grid DN13 --rx-grid JN48 --band 10m \
      --sfi 150 --kp 2 --hour 2 --month 2 --day-of-year 45
"""

import argparse
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
from physics_override import apply_override_to_prediction, PhysicsOverrideLayer

# -- Constants -----------------------------------------------------------------

MODE_THRESHOLDS_DB = {
    "WSPR": -28.0,
    "FT8":  -21.0,
    "CW":   -15.0,
    "RTTY":  -5.0,
    "SSB":    3.0,
}

WSPR_MEAN_DB = -17.53
WSPR_STD_DB = 6.7


def sigma_to_approx_db(sigma):
    return sigma * WSPR_STD_DB + WSPR_MEAN_DB


def mode_verdicts(snr_db):
    verdicts = {}
    for mode, threshold in MODE_THRESHOLDS_DB.items():
        verdicts[mode] = "OPEN" if snr_db >= threshold else "closed"
    return verdicts


def main():
    parser = argparse.ArgumentParser(
        description="IONIS V22-gamma + PhysicsOverrideLayer — Single path prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s --tx-grid FN20 --rx-grid IO91 --band 20m --sfi 150 --kp 2 --hour 14 --month 6
  %(prog)s --tx-grid DN13 --rx-grid JN48 --band 10m --sfi 150 --kp 2 --hour 2 --month 2 --day-of-year 45
""",
    )
    parser.add_argument("--tx-grid", required=True, help="TX Maidenhead grid (4-char)")
    parser.add_argument("--rx-grid", required=True, help="RX Maidenhead grid (4-char)")
    parser.add_argument("--band", required=True, choices=list(BAND_FREQ_HZ.keys()),
                        help="HF band (e.g. 20m, 40m)")
    parser.add_argument("--sfi", type=float, required=True, help="Solar Flux Index (65-300)")
    parser.add_argument("--kp", type=float, required=True, help="Kp index (0-9)")
    parser.add_argument("--hour", type=int, required=True, help="Hour UTC (0-23)")
    parser.add_argument("--month", type=int, required=True, help="Month (1-12)")
    parser.add_argument("--day-of-year", type=int, default=None,
                        help="Day of year (1-366). Default: estimated from month.")
    parser.add_argument("--config", default=None, help="Path to config JSON")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint")

    args = parser.parse_args()

    if not (65 <= args.sfi <= 350):
        print(f"  WARNING: SFI {args.sfi} outside typical range (65-300)", file=sys.stderr)
    if not (0 <= args.kp <= 9):
        print(f"  ERROR: Kp must be 0-9, got {args.kp}", file=sys.stderr)
        return 1
    if not (0 <= args.hour <= 23):
        print(f"  ERROR: Hour must be 0-23, got {args.hour}", file=sys.stderr)
        return 1
    if not (1 <= args.month <= 12):
        print(f"  ERROR: Month must be 1-12, got {args.month}", file=sys.stderr)
        return 1

    # Estimate day_of_year from month if not provided
    day_of_year = args.day_of_year
    if day_of_year is None:
        month_starts = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
        day_of_year = month_starts[args.month] + 15  # Mid-month estimate

    # Load model
    device = get_device()
    config_path = args.config
    if config_path is None:
        config_path = os.path.join(V22_DIR, "config_v22.json")

    model, config, metadata = load_model(config_path, args.checkpoint, device)

    # Resolve grids
    tx_lat, tx_lon = grid4_to_latlon(args.tx_grid)
    rx_lat, rx_lon = grid4_to_latlon(args.rx_grid)

    freq_hz = BAND_FREQ_HZ[args.band]
    freq_mhz = freq_hz / 1e6
    distance_km = haversine_km(tx_lat, tx_lon, rx_lat, rx_lon)

    # Build V22 features (solar depression + cross-products)
    features = build_features(
        tx_lat, tx_lon, rx_lat, rx_lon,
        freq_hz, args.sfi, args.kp, args.hour, args.month,
        day_of_year=day_of_year,
        include_solar_depression=True,
    )
    x = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        snr_sigma = model(x).item()

    # Apply PhysicsOverrideLayer
    tx_solar = solar_elevation_deg(tx_lat, tx_lon, args.hour, day_of_year)
    rx_solar = solar_elevation_deg(rx_lat, rx_lon, args.hour, day_of_year)
    clamped_sigma, was_overridden = apply_override_to_prediction(
        snr_sigma, freq_mhz, tx_solar, rx_solar, distance_km=distance_km)

    snr_db = sigma_to_approx_db(clamped_sigma)
    verdicts = mode_verdicts(snr_db)

    # Output
    print()
    print("=" * 60)
    print("  IONIS V22-gamma + PhysicsOverrideLayer — Path Prediction")
    print("=" * 60)
    print()
    print(f"  TX Grid:    {args.tx_grid.upper()} ({tx_lat:.1f}N, {tx_lon:.1f}E)")
    print(f"  RX Grid:    {args.rx_grid.upper()} ({rx_lat:.1f}N, {rx_lon:.1f}E)")
    print(f"  Distance:   {distance_km:,.0f} km")
    print(f"  Band:       {args.band} ({freq_mhz:.3f} MHz)")
    print(f"  Conditions: SFI {args.sfi:.0f}, Kp {args.kp:.1f}")
    print(f"  Time:       {args.hour:02d}:00 UTC, month {args.month}, DOY {day_of_year}")
    print()
    print(f"  TX Solar:   {tx_solar:+.1f} deg")
    print(f"  RX Solar:   {rx_solar:+.1f} deg")
    print()

    if was_overridden:
        print(f"  Raw Model:  {snr_sigma:+.3f} sigma")
        print(f"  Override:   {clamped_sigma:+.3f} sigma (CLAMPED)")
        print(f"  Predicted:  {clamped_sigma:+.3f} sigma ({snr_db:+.1f} dB)")
    else:
        print(f"  Predicted:  {clamped_sigma:+.3f} sigma ({snr_db:+.1f} dB)")

    print()
    print("  Mode Verdicts:")
    for mode, verdict in verdicts.items():
        marker = ">>>" if verdict == "OPEN" else "   "
        threshold = MODE_THRESHOLDS_DB[mode]
        print(f"    {marker} {mode:<5s}  {verdict:<6s}  (threshold: {threshold:+.0f} dB)")
    print()
    print(f"  Device: {device}")
    if was_overridden:
        override = PhysicsOverrideLayer()
        print(f"  Override: {override.describe()}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
