#!/usr/bin/env python3
"""
validate_template.py — Step I Recall Validation Template

Queries contest paths from validation.step_i_paths, runs inference,
denormalizes to dB, applies mode thresholds, and reports recall.

Usage:
    python validate_template.py v18
    python validate_template.py v17
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import clickhouse_connect

# Add templates to path
sys.path.insert(0, str(Path(__file__).parent))
from common import (
    load_config, load_model, DEVICE, INPUT_DIM, FREQ_MHZ, MHZ_TO_BAND, BAND_TO_HZ
)


CH_HOST = "10.60.1.1"
CH_PORT = 8123


def engineer_features_batch(df) -> np.ndarray:
    """Build feature matrix for validation paths."""
    n = len(df)
    features = np.zeros((n, INPUT_DIM), dtype=np.float32)

    for i in range(n):
        row = df.iloc[i]

        # Grid to lat/lon
        def grid4_to_latlon(g):
            s = str(g).strip().upper()[:4]
            if len(s) < 4:
                s = "JJ00"
            lon = (ord(s[0]) - ord("A")) * 20.0 - 180.0 + int(s[2]) * 2.0 + 1.0
            lat = (ord(s[1]) - ord("A")) * 10.0 - 90.0 + int(s[3]) * 1.0 + 0.5
            return lat, lon

        tx_lat, tx_lon = grid4_to_latlon(row["tx_grid"])
        rx_lat, rx_lon = grid4_to_latlon(row["rx_grid"])

        midpoint_lat = (tx_lat + rx_lat) / 2.0
        midpoint_lon = (tx_lon + rx_lon) / 2.0

        band_id = MHZ_TO_BAND.get(row["band_mhz"], 107)
        freq_hz = BAND_TO_HZ.get(band_id, 14_097_100)
        kp_penalty = 1.0 - row["kp"] / 9.0

        features[i] = [
            row["distance"] / 20000.0,
            np.log10(freq_hz) / 8.0,
            np.sin(2.0 * np.pi * row["hour"] / 24.0),
            np.cos(2.0 * np.pi * row["hour"] / 24.0),
            np.sin(2.0 * np.pi * row["azimuth"] / 360.0),
            np.cos(2.0 * np.pi * row["azimuth"] / 360.0),
            abs(tx_lat - rx_lat) / 180.0,
            midpoint_lat / 90.0,
            np.sin(2.0 * np.pi * row["month"] / 12.0),
            np.cos(2.0 * np.pi * row["month"] / 12.0),
            np.cos(2.0 * np.pi * (row["hour"] + midpoint_lon / 15.0) / 24.0),
            row["sfi"] / 300.0,
            kp_penalty,
        ]

    return features


def main():
    parser = argparse.ArgumentParser(description="IONIS Step I Recall Validation")
    parser.add_argument("version", help="Version to test (e.g., v18)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  IONIS {args.version.upper()} — Step I Validation")
    print("=" * 70)
    print()

    # Load config and model
    try:
        config = load_config(args.version)
        print(f"Loaded config: {config.config_path}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    try:
        model = load_model(config)
        print(f"Loaded checkpoint: {config.checkpoint_path}")
        print(f"  norm_mean = {model.norm_mean:.2f}")
        print(f"  norm_std  = {model.norm_std:.2f}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"Model: {config.architecture} ({total_params:,} params)")
    print()

    # Load validation paths
    print(f"Connecting to ClickHouse at {CH_HOST}:{CH_PORT}...")
    client = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT)

    query = """
    SELECT
        tx_grid, rx_grid, band_mhz, mode, hour, month,
        sfi, kp, distance, azimuth
    FROM validation.step_i_paths
    """

    print("Loading Step I paths...")
    df = client.query_df(query)
    client.close()

    n_samples = len(df)
    print(f"  Loaded {n_samples:,} paths")
    print()

    # Map band_mhz to band_id
    df["band_id"] = df["band_mhz"].map(MHZ_TO_BAND)

    # Map mode to threshold using config
    mode_to_threshold = {
        "CW": config.get_threshold("cw"),
        "DG": config.get_threshold("dg"),
        "PH": config.get_threshold("ssb"),
        "RY": config.get_threshold("rtty"),
    }
    threshold = df["mode"].map(mode_to_threshold).values.astype(np.float32)
    mode = df["mode"].values
    band_ids = df["band_id"].values

    # Engineer features
    print("Engineering features...")
    features = engineer_features_batch(df)
    print(f"  Features shape: {features.shape}")
    print()

    # Run inference
    print("Running inference...")
    batch_size = 65536
    predictions_sigma = np.zeros(n_samples, dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            batch = torch.tensor(features[i:end], dtype=torch.float32, device=DEVICE)
            pred = model.model(batch).cpu().numpy().flatten()
            predictions_sigma[i:end] = pred
            if (i // batch_size) % 5 == 0:
                print(f"  Processed {end:,} / {n_samples:,} ({100*end/n_samples:.1f}%)")

    print(f"  Inference complete")
    print(f"  Predictions (σ): min={predictions_sigma.min():.3f}, max={predictions_sigma.max():.3f}")
    print()

    # Denormalize
    print("Denormalizing predictions to dB...")
    predictions_db = model.denormalize(predictions_sigma)

    print(f"  Predictions (dB): min={predictions_db.min():.1f}, max={predictions_db.max():.1f}, mean={predictions_db.mean():.1f}")
    print(f"  Thresholds: min={threshold.min():.1f}, max={threshold.max():.1f}")
    print()

    # Apply thresholds
    print("Applying mode thresholds...")
    band_open = predictions_db >= threshold

    # Calculate recall
    total = len(band_open)
    open_count = band_open.sum()
    recall = 100.0 * open_count / total

    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print()
    print(f"  Total paths:     {total:,}")
    print(f"  Band open:       {open_count:,}")
    print(f"  {args.version.upper()} Recall:  {recall:.2f}%")
    print()

    # Baselines from config
    voacap = config.baselines.get("voacap", 75.82)
    prev_version = config.baselines.get("previous_version", "v20")
    prev_recall = config.baselines.get("previous_recall", 96.38)

    print("  Baselines:")
    print(f"    VOACAP:        {voacap:.2f}%")
    print(f"    IONIS {prev_version}:     {prev_recall:.2f}%")
    print()

    delta_voacap = recall - voacap
    delta_prev = recall - prev_recall
    print(f"  {args.version.upper()} vs VOACAP:   {delta_voacap:+.2f} pp")
    print(f"  {args.version.upper()} vs {prev_version}:      {delta_prev:+.2f} pp")
    print()

    # Breakdown by mode
    print("  Recall by Mode:")
    for m in ["CW", "PH", "RY", "DG"]:
        mask = mode == m
        if mask.sum() > 0:
            mode_recall = 100.0 * band_open[mask].sum() / mask.sum()
            print(f"    {m}:  {mode_recall:.2f}% ({mask.sum():,} paths)")
    print()

    # Breakdown by band
    print("  Recall by Band:")
    band_names = {102: "160m", 103: "80m", 105: "40m", 107: "20m", 109: "15m", 111: "10m"}
    for band_id, band_name in sorted(band_names.items()):
        mask = band_ids == band_id
        if mask.sum() > 0:
            band_recall = 100.0 * band_open[mask].sum() / mask.sum()
            print(f"    {band_name}:  {band_recall:.2f}% ({mask.sum():,} paths)")
    print()

    print("=" * 70)

    # Check against validation expectations
    min_recall = config.validation.get("step_i_recall_min", 80.0)
    max_recall = config.validation.get("step_i_recall_max", 99.0)

    if recall > max_recall:
        print(f"  STATUS: WARNING — Recall {recall:.1f}% > {max_recall}% suggests normalization issue")
        return 1
    elif recall >= min_recall:
        print(f"  STATUS: PASS (recall {recall:.1f}% in range [{min_recall}, {max_recall}])")
        return 0
    else:
        print(f"  STATUS: FAIL (recall {recall:.1f}% < {min_recall}%)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
