#!/usr/bin/env python3
"""
run_adif.py — IONIS V20 ADIF Log Validation

Validates the V20 model against your own QSO log. Export your log from
eQSL, LoTW, or QRZ as an ADIF (.adi) file, then run this tool. All
processing happens locally — callsigns are stripped immediately and
never leave your machine.

Usage:
  ionis-validate adif my_log.adi --my-grid DN26
  ionis-validate adif my_log.adi --my-grid DN26 --export observations.json
  ionis-validate adif my_log.adi --my-grid DN26 --sfi 140 --kp 2
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import torch

# ── Path Setup ───────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V20_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, V20_DIR)

from model import IonisGate, get_device, load_model, build_features, grid4_to_latlon

# ── ADIF Parser (self-contained, no external dependencies) ───────────────────

_TAG_RE = re.compile(
    r"<(?P<tag>[A-Za-z0-9_]+)(?::(?P<len>\d+))?(?::[A-Za-z0-9]+)?>",
)

_GRID_RE = re.compile(r"^[A-Ra-r]{2}[0-9]{2}([A-Xa-x]{2})?$")

# Fields we extract — everything else is discarded
_OBSERVATION_FIELDS = {
    "band", "mode", "submode", "freq", "qso_date", "time_on",
    "gridsquare", "my_gridsquare", "rst_sent", "rst_rcvd",
}

# ADIF band labels → IONIS band IDs
_BAND_MAP = {
    "160m": 102, "80m": 103, "60m": 104, "40m": 105, "30m": 106,
    "20m": 107, "17m": 108, "15m": 109, "12m": 110, "10m": 111,
}

# IONIS band IDs → freq MHz (for feature engineering)
_BAND_MHZ = {
    102: 1.8, 103: 3.5, 104: 5.3, 105: 7.0, 106: 10.1,
    107: 14.0, 108: 18.1, 109: 21.0, 110: 24.9, 111: 28.0,
}

# Mode mapping: ADIF mode → IONIS mode family
_MODE_MAP = {
    "SSB": "PH", "USB": "PH", "LSB": "PH", "AM": "PH", "FM": "PH",
    "CW": "CW",
    "RTTY": "RY", "PSK31": "RY", "PSK63": "RY", "PSK": "RY",
    "FT8": "DG", "FT4": "DG", "JT65": "DG", "JT9": "DG",
    "WSPR": "DG", "JS8": "DG", "MSK144": "DG", "Q65": "DG",
    "MFSK": "DG", "OLIVIA": "DG", "CONTESTIA": "DG",
}

# Mode-specific thresholds (dB) — band is "open" if predicted SNR >= threshold
_MODE_THRESHOLDS = {
    "DG": -21.0,   # FT8/FT4 decode limit
    "CW": -15.0,   # Readable CW
    "RY": -5.0,    # Machine-copy RTTY
    "PH": 3.0,     # Voice-quality SSB
}


def parse_adif(filepath, encoding="utf-8"):
    """Parse ADIF file, returning only observation fields (no PII).

    Callsigns, names, comments, QSL messages, and all other personal
    information are discarded at parse time. They never enter memory
    beyond the raw line buffer.
    """
    with open(filepath, encoding=encoding, errors="replace") as f:
        text = f.read()

    records = []
    current = {}
    tokens = list(_TAG_RE.finditer(text))

    for i, match in enumerate(tokens):
        tag = match.group("tag").upper()
        length = match.group("len")
        end = match.end()

        if length is not None:
            val = text[end:end + int(length)].strip()
        else:
            next_start = tokens[i + 1].start() if i + 1 < len(tokens) else len(text)
            val = text[end:next_start].strip()

        if tag == "EOR":
            if current:
                records.append(current)
                current = {}
            continue

        if tag == "EOH":
            current = {}
            continue

        key = tag.lower()
        if key in _OBSERVATION_FIELDS and val:
            current[key] = val

    if current:
        records.append(current)

    return records


def extract_observations(records, my_grid_default):
    """Convert parsed ADIF records to anonymous observations.

    Returns list of dicts with: tx_grid, rx_grid, band_id, freq_mhz,
    mode, ionis_mode, hour_utc, month, year, snr_db (if available).
    """
    observations = []
    skipped = {"no_grid": 0, "no_band": 0, "bad_grid": 0, "non_hf": 0}

    for rec in records:
        # Grid resolution
        my_grid = rec.get("my_gridsquare", my_grid_default)
        their_grid = rec.get("gridsquare")

        if not my_grid or not their_grid:
            skipped["no_grid"] += 1
            continue

        # Normalize to 4-char uppercase
        my_grid = my_grid[:4].upper()
        their_grid = their_grid[:4].upper()

        if not _GRID_RE.match(my_grid) or not _GRID_RE.match(their_grid):
            skipped["bad_grid"] += 1
            continue

        # Band resolution
        band_str = rec.get("band", "").lower()
        band_id = _BAND_MAP.get(band_str)

        if not band_id:
            # Try deriving from freq
            freq_str = rec.get("freq")
            if freq_str:
                try:
                    freq_mhz = float(freq_str)
                    for b_label, b_id in _BAND_MAP.items():
                        if abs(_BAND_MHZ[b_id] - freq_mhz) / _BAND_MHZ[b_id] < 0.15:
                            band_id = b_id
                            break
                except ValueError:
                    pass

        if not band_id:
            skipped["no_band"] += 1
            continue

        if band_id not in _BAND_MHZ:
            skipped["non_hf"] += 1
            continue

        freq_mhz = _BAND_MHZ[band_id]

        # Mode
        mode_raw = rec.get("submode") or rec.get("mode") or "UNKNOWN"
        mode_raw = mode_raw.upper()
        ionis_mode = _MODE_MAP.get(mode_raw, "DG")

        # Time
        date_str = rec.get("qso_date", "")
        time_str = rec.get("time_on", "")

        month = 6
        hour_utc = 12
        year = 2024

        if len(date_str) >= 8:
            try:
                year = int(date_str[:4])
                month = int(date_str[4:6])
            except ValueError:
                pass

        if len(time_str) >= 4:
            try:
                hour_utc = int(time_str[:2])
            except ValueError:
                pass

        # SNR (if available — FT8 RST is often dB)
        snr_db = None
        rst = rec.get("rst_sent") or rec.get("rst_rcvd")
        if rst:
            rst = rst.strip()
            try:
                val = int(rst)
                if -50 <= val <= 50:
                    snr_db = float(val)
            except ValueError:
                try:
                    val = float(rst)
                    if -50 <= val <= 50:
                        snr_db = val
                except ValueError:
                    pass

        obs = {
            "tx_grid": my_grid,
            "rx_grid": their_grid,
            "band_id": band_id,
            "freq_mhz": freq_mhz,
            "mode": mode_raw,
            "ionis_mode": ionis_mode,
            "hour_utc": hour_utc,
            "month": month,
            "year": year,
        }
        if snr_db is not None:
            obs["snr_db"] = snr_db

        observations.append(obs)

    return observations, skipped


def run_predictions(observations, model, config, device, sfi_override=None,
                    kp_override=None):
    """Run V20 inference on observations and compute recall."""
    if not observations:
        return []

    # Build arrays
    n = len(observations)
    tx_lat = np.zeros(n, dtype=np.float32)
    tx_lon = np.zeros(n, dtype=np.float32)
    rx_lat = np.zeros(n, dtype=np.float32)
    rx_lon = np.zeros(n, dtype=np.float32)
    freq_mhz = np.zeros(n, dtype=np.float32)
    month = np.zeros(n, dtype=np.int32)
    hour_utc = np.zeros(n, dtype=np.int32)
    band_ids = np.zeros(n, dtype=np.int32)

    for i, obs in enumerate(observations):
        lat, lon = grid4_to_latlon(obs["tx_grid"])
        tx_lat[i], tx_lon[i] = lat, lon
        lat, lon = grid4_to_latlon(obs["rx_grid"])
        rx_lat[i], rx_lon[i] = lat, lon
        freq_mhz[i] = obs["freq_mhz"]
        month[i] = obs["month"]
        hour_utc[i] = obs["hour_utc"]
        band_ids[i] = obs["band_id"]

    sfi_val = sfi_override if sfi_override else 150.0
    ssn = np.full(n, (sfi_val - 63.0) / 0.7, dtype=np.float32)

    features = build_features(tx_lat, tx_lon, rx_lat, rx_lon, freq_mhz,
                              month, hour_utc, ssn)

    # Batch inference
    norm_constants = {}
    for band_str, sources in config["norm_constants_per_band"].items():
        bid = int(band_str)
        norm_constants[bid] = (sources["wspr"]["mean"], sources["wspr"]["std"])

    predictions_sigma = np.zeros(n, dtype=np.float32)
    batch_size = 50000

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = torch.tensor(features[start:end], dtype=torch.float32,
                                 device=device)
            pred = model(batch).cpu().numpy().flatten()
            predictions_sigma[start:end] = pred

    # Denormalize to dB
    predictions_db = np.zeros_like(predictions_sigma)
    for bid, (mean, std) in norm_constants.items():
        mask = band_ids == bid
        if mask.sum() > 0:
            predictions_db[mask] = predictions_sigma[mask] * std + mean

    # Store results back
    results = []
    for i, obs in enumerate(observations):
        threshold = _MODE_THRESHOLDS.get(obs["ionis_mode"], -21.0)
        predicted_db = float(predictions_db[i])
        band_open = predicted_db >= threshold
        r = {**obs, "predicted_db": predicted_db, "threshold_db": threshold,
             "band_open": band_open}
        results.append(r)

    return results


def print_report(results, skipped, filepath, my_grid, sfi, kp):
    """Print human-readable validation report."""
    total = len(results)
    if total == 0:
        print("  No valid observations to validate.")
        return

    open_count = sum(1 for r in results if r["band_open"])
    recall = 100.0 * open_count / total

    print()
    print("=" * 70)
    print("  IONIS V20 — ADIF Log Validation")
    print("=" * 70)
    print()
    print(f"  Log file:      {os.path.basename(filepath)}")
    print(f"  Your grid:     {my_grid}")
    print(f"  Conditions:    SFI={sfi}, Kp={kp}")
    print()

    # Skipped summary
    total_skipped = sum(skipped.values())
    if total_skipped:
        print(f"  Records skipped: {total_skipped:,}")
        for reason, count in skipped.items():
            if count > 0:
                print(f"    {reason}: {count:,}")
        print()

    print(f"  Observations validated: {total:,}")
    print(f"  Band open (predicted):  {open_count:,}")
    print(f"  Recall:                 {recall:.2f}%")
    print()

    # Year range
    years = sorted(set(r["year"] for r in results))
    if years:
        print(f"  Date range:    {years[0]} – {years[-1]}")
        print()

    # Breakdown by mode
    print("  Recall by Mode:")
    modes_seen = sorted(set(r["ionis_mode"] for r in results))
    mode_labels = {"DG": "Digital", "CW": "CW", "RY": "RTTY", "PH": "SSB"}
    for m in modes_seen:
        m_results = [r for r in results if r["ionis_mode"] == m]
        m_open = sum(1 for r in m_results if r["band_open"])
        m_recall = 100.0 * m_open / len(m_results) if m_results else 0
        label = mode_labels.get(m, m)
        print(f"    {label:<10s}  {m_recall:6.2f}%  ({len(m_results):,} QSOs)")
    print()

    # Breakdown by band
    print("  Recall by Band:")
    band_names = {102: "160m", 103: "80m", 104: "60m", 105: "40m", 106: "30m",
                  107: "20m", 108: "17m", 109: "15m", 110: "12m", 111: "10m"}
    for bid in sorted(band_names.keys()):
        b_results = [r for r in results if r["band_id"] == bid]
        if not b_results:
            continue
        b_open = sum(1 for r in b_results if r["band_open"])
        b_recall = 100.0 * b_open / len(b_results) if b_results else 0
        print(f"    {band_names[bid]:<6s}  {b_recall:6.2f}%  ({len(b_results):,} QSOs)")
    print()

    # SNR comparison (if FT8/digital QSOs have SNR data)
    snr_results = [r for r in results if "snr_db" in r]
    if len(snr_results) >= 10:
        actual_snr = np.array([r["snr_db"] for r in snr_results])
        predicted_snr = np.array([r["predicted_db"] for r in snr_results])
        correlation = np.corrcoef(actual_snr, predicted_snr)[0, 1]
        rmse = np.sqrt(np.mean((actual_snr - predicted_snr) ** 2))
        print(f"  SNR Comparison ({len(snr_results):,} QSOs with signal reports):")
        print(f"    Pearson correlation:  {correlation:+.4f}")
        print(f"    RMSE:                {rmse:.1f} dB")
        print()

    print("=" * 70)
    print(f"  Overall Recall: {recall:.2f}% on {total:,} confirmed QSOs")
    print("=" * 70)
    print()
    print("  Every QSO in your log is a confirmed contact — the band WAS open.")
    print("  Recall measures how often the model agrees.")
    print()


def export_observations(results, output_path):
    """Export anonymized observations to JSON.

    Output contains ONLY: grid-pair, band, mode, datetime, predicted SNR.
    No callsigns, no names, no personal information.
    """
    exported = []
    for r in results:
        obs = {
            "tx_grid": r["tx_grid"],
            "rx_grid": r["rx_grid"],
            "band_id": r["band_id"],
            "freq_mhz": r["freq_mhz"],
            "mode": r["mode"],
            "hour_utc": r["hour_utc"],
            "month": r["month"],
            "year": r["year"],
            "predicted_db": round(r["predicted_db"], 2),
            "band_open": r["band_open"],
        }
        if "snr_db" in r:
            obs["reported_snr_db"] = r["snr_db"]
        exported.append(obs)

    with open(output_path, "w") as f:
        json.dump({
            "format": "ionis-adif-observations-v1",
            "model": "V20",
            "generated": datetime.now(timezone.utc).isoformat(),
            "count": len(exported),
            "privacy": "Anonymous grid-pair observations. No callsigns or personal data.",
            "observations": exported,
        }, f, indent=2)

    print(f"  Exported {len(exported):,} anonymous observations to {output_path}")
    print(f"  Review the file before sharing — it should contain no personal data.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Validate V20 model against your ADIF QSO log",
    )
    parser.add_argument(
        "adif_file",
        help="Path to your ADIF (.adi) log file",
    )
    parser.add_argument(
        "--my-grid",
        required=True,
        help="Your 4-character Maidenhead grid (e.g., DN26)",
    )
    parser.add_argument(
        "--sfi", type=float, default=150.0,
        help="Solar Flux Index (default: 150)",
    )
    parser.add_argument(
        "--kp", type=float, default=2.0,
        help="Kp geomagnetic index (default: 2)",
    )
    parser.add_argument(
        "--export", metavar="FILE",
        help="Export anonymized observations to JSON file",
    )
    parser.add_argument(
        "--encoding", default="utf-8",
        help="File encoding (default: utf-8)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.adif_file):
        print(f"  ERROR: File not found: {args.adif_file}", file=sys.stderr)
        return 1

    my_grid = args.my_grid.upper()[:4]
    if not _GRID_RE.match(my_grid):
        print(f"  ERROR: Invalid grid: {args.my_grid}", file=sys.stderr)
        return 1

    # Parse ADIF (PII stripped at parse time)
    print(f"  Parsing {args.adif_file}...")
    records = parse_adif(args.adif_file, encoding=args.encoding)
    print(f"  Parsed {len(records):,} ADIF records")

    # Extract anonymous observations
    observations, skipped = extract_observations(records, my_grid)
    print(f"  Extracted {len(observations):,} HF observations with valid grids")

    if not observations:
        print("  No valid observations found. Check that your log contains")
        print("  GRIDSQUARE fields and HF band QSOs.")
        return 1

    # Load model
    print(f"  Loading V20 model...")
    model, config, checkpoint = load_model()
    device = get_device()
    model = model.to(device)

    # Run predictions
    print(f"  Running inference on {len(observations):,} observations...")
    results = run_predictions(observations, model, config, device,
                              sfi_override=args.sfi, kp_override=args.kp)

    # Print report
    print_report(results, skipped, args.adif_file, my_grid, args.sfi, args.kp)

    # Export if requested
    if args.export:
        export_observations(results, args.export)

    return 0


if __name__ == "__main__":
    sys.exit(main())
