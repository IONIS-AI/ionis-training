#!/usr/bin/env python3
"""
run_predict.py — IONIS V20 Single Path Prediction CLI

Predict the SNR for a single HF propagation path.

Usage:
  python run_predict.py --tx-grid FN20 --rx-grid IO91 --band 20m \
      --sfi 150 --kp 2 --hour 14 --month 6

  ionis-validate predict --tx-grid DN26 --rx-grid IO91 --band 20m \
      --sfi 140 --kp 1.5 --hour 14 --month 6

Output includes predicted SNR in sigma and dB, plus per-mode verdicts
(WSPR/FT8/CW/RTTY/SSB open/closed).
"""

import argparse
import os
import sys

import numpy as np
import torch

# ── Path Setup ───────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V20_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, V20_DIR)

from model import (
    IonisGate, get_device, load_model,
    grid4_to_latlon, build_features, haversine_km, BAND_FREQ_HZ,
)

# ── Constants ────────────────────────────────────────────────────────────────

SIGMA_TO_DB = 6.7  # Approximate sigma → dB conversion

# Mode decode thresholds (dB relative to noise floor)
# These represent typical minimum SNR for each mode to decode
MODE_THRESHOLDS_DB = {
    "WSPR": -28.0,
    "FT8":  -21.0,
    "CW":   -15.0,
    "RTTY":  -5.0,
    "SSB":    3.0,
}

# WSPR normalization: mean and std from V20 training data (20m reference)
# Model output is in Z-score units relative to WSPR distribution
WSPR_MEAN_DB = -17.53  # 20m WSPR mean SNR
WSPR_STD_DB = 6.7      # approximate std


def sigma_to_approx_db(sigma):
    """Convert model sigma output to approximate dB SNR."""
    return sigma * WSPR_STD_DB + WSPR_MEAN_DB


def mode_verdicts(snr_db):
    """Return per-mode open/closed verdicts based on approximate dB SNR."""
    verdicts = {}
    for mode, threshold in MODE_THRESHOLDS_DB.items():
        verdicts[mode] = "OPEN" if snr_db >= threshold else "closed"
    return verdicts


def main():
    parser = argparse.ArgumentParser(
        description="IONIS V20 — Single path prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s --tx-grid FN20 --rx-grid IO91 --band 20m --sfi 150 --kp 2 --hour 14 --month 6
  %(prog)s --tx-grid DN26 --rx-grid PM95 --band 20m --sfi 140 --kp 1.5 --hour 6 --month 12
""",
    )
    parser.add_argument("--tx-grid", required=True, help="TX Maidenhead grid (4-char, e.g. FN20)")
    parser.add_argument("--rx-grid", required=True, help="RX Maidenhead grid (4-char, e.g. IO91)")
    parser.add_argument("--band", required=True, choices=list(BAND_FREQ_HZ.keys()),
                        help="HF band (e.g. 20m, 40m)")
    parser.add_argument("--sfi", type=float, required=True, help="Solar Flux Index (65-300)")
    parser.add_argument("--kp", type=float, required=True, help="Kp index (0-9)")
    parser.add_argument("--hour", type=int, required=True, help="Hour UTC (0-23)")
    parser.add_argument("--month", type=int, required=True, help="Month (1-12)")
    parser.add_argument("--config", default=None, help="Path to config JSON (default: auto-discover V20)")
    parser.add_argument("--checkpoint", default=None, help="Path to .pth file (default: from config)")

    args = parser.parse_args()

    # Validate inputs
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

    # Load model
    device = get_device()
    config_path = args.config
    if config_path is None:
        config_path = os.path.join(V20_DIR, "config_v20.json")

    model, config, checkpoint = load_model(config_path, args.checkpoint, device)

    # Resolve grids
    tx_lat, tx_lon = grid4_to_latlon(args.tx_grid)
    rx_lat, rx_lon = grid4_to_latlon(args.rx_grid)

    freq_hz = BAND_FREQ_HZ[args.band]
    distance_km = haversine_km(tx_lat, tx_lon, rx_lat, rx_lon)

    # Build features and predict
    features = build_features(
        tx_lat, tx_lon, rx_lat, rx_lon,
        freq_hz, args.sfi, args.kp, args.hour, args.month,
    )
    x = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        snr_sigma = model(x).item()

    snr_db = sigma_to_approx_db(snr_sigma)
    verdicts = mode_verdicts(snr_db)

    # Output
    print()
    print("=" * 60)
    print("  IONIS V20 — Path Prediction")
    print("=" * 60)
    print()
    print(f"  TX Grid:    {args.tx_grid.upper()} ({tx_lat:.1f}N, {tx_lon:.1f}E)")
    print(f"  RX Grid:    {args.rx_grid.upper()} ({rx_lat:.1f}N, {rx_lon:.1f}E)")
    print(f"  Distance:   {distance_km:,.0f} km")
    print(f"  Band:       {args.band} ({freq_hz/1e6:.3f} MHz)")
    print(f"  Conditions: SFI {args.sfi:.0f}, Kp {args.kp:.1f}")
    print(f"  Time:       {args.hour:02d}:00 UTC, month {args.month}")
    print()
    print(f"  Predicted SNR: {snr_sigma:+.3f} sigma ({snr_db:+.1f} dB)")
    print()
    print("  Mode Verdicts:")
    for mode, verdict in verdicts.items():
        marker = ">>>" if verdict == "OPEN" else "   "
        threshold = MODE_THRESHOLDS_DB[mode]
        print(f"    {marker} {mode:<5s}  {verdict:<6s}  (threshold: {threshold:+.0f} dB)")
    print()
    print(f"  Device: {device}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
