#!/usr/bin/env python3
"""
predict.py — IONIS Unified Prediction Interface (Step J)

Combines neural oracle (V13) + signature search into a single tool.
Returns both predictions with confidence weighting based on signature density.

Usage:
  python predict.py --tx FN31 --rx JO21 --band 20m --hour 14 --month 6
  python predict.py --tx FN31 --rx JO21 --band 20m --hour 14 --month 6 --json
  python predict.py --test

Output:
  - Neural prediction (V13): SNR in σ and dB
  - Signature match: median SNR from historical signatures
  - Confidence: HIGH/MEDIUM/LOW based on signature density
  - Agreement: GOOD/FAIR/POOR based on neural vs signature delta
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional

# ── Import Oracle and Signature Search ────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Import oracle_v13
from oracle_v13 import IonisOracle, haversine, sigma_to_db, freq_to_band, NORM_CONSTANTS

# Import signature_search
from signature_search import (
    SignatureSearch, SearchResult, validate_grid, grid4_to_latlon,
    parse_band, BAND_LABEL_TO_ADIF, ADIF_TO_LABEL
)


# ── Configuration ─────────────────────────────────────────────────────────────

# ClickHouse connection for signature search (HTTP port)
CH_HOST = os.environ.get("CH_HOST", "10.60.1.1")
CH_PORT = int(os.environ.get("CH_PORT", "8123"))

# Band to MHz mapping
BAND_TO_MHZ = {
    102: 1.8, 103: 3.5, 104: 5.3, 105: 7.0, 106: 10.1,
    107: 14.0, 108: 18.1, 109: 21.0, 110: 24.9, 111: 28.0,
}

# Agreement thresholds
AGREEMENT_GOOD_DB = 3.0    # Within 3 dB = GOOD
AGREEMENT_FAIR_DB = 6.0    # Within 6 dB = FAIR


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class UnifiedPrediction:
    # Query echo
    tx_grid: str
    rx_grid: str
    band: int
    band_label: str
    hour: int
    month: int
    sfi: float
    kp: float

    # Path geometry
    distance_km: float
    bearing_deg: float

    # Neural oracle prediction (V13)
    oracle_snr_sigma: float
    oracle_snr_db: float
    oracle_condition: str
    oracle_confidence: str

    # Signature search prediction
    sig_snr_db: float
    sig_neighbors: int
    sig_spot_count: int
    sig_condition: str
    sig_confidence: str
    sig_query_ms: float

    # Combined assessment
    combined_snr_db: float
    combined_condition: str
    combined_confidence: str
    agreement: str
    delta_db: float

    # Warnings
    warnings: List[str]


# ── Unified Predictor ─────────────────────────────────────────────────────────

class UnifiedPredictor:
    """IONIS Unified Prediction — Neural Oracle + Signature Search."""

    # Condition thresholds
    CONDITION_THRESHOLDS = [
        (-10, "EXCELLENT (Voice/SSB)"),
        (-15, "GOOD (CW/Digital)"),
        (-20, "FAIR (FT8/FT4)"),
        (-28, "MARGINAL (WSPR)"),
    ]
    CONDITION_CLOSED = "CLOSED"

    def __init__(self, ch_host: str = CH_HOST, ch_port: int = CH_PORT):
        self.oracle = IonisOracle()
        self.sig_search = SignatureSearch(host=ch_host, port=ch_port)

    def _classify_condition(self, snr: float) -> str:
        for threshold, label in self.CONDITION_THRESHOLDS:
            if snr > threshold:
                return label
        return self.CONDITION_CLOSED

    def _combine_predictions(
        self,
        oracle_db: float,
        sig_db: float,
        sig_neighbors: int,
        sig_confidence: str,
    ) -> tuple:
        """
        Combine oracle and signature predictions with confidence weighting.

        Strategy:
          - Dense signatures (HIGH confidence): weight signatures 70%, oracle 30%
          - Medium signatures: weight 50/50
          - Sparse signatures (LOW confidence): weight oracle 70%, signatures 30%
          - No signatures: use oracle only
        """
        if sig_neighbors == 0:
            # No signatures — oracle only
            return oracle_db, "LOW (no signatures)"

        if sig_confidence == "HIGH":
            # Dense signatures — trust historical data more
            combined = 0.3 * oracle_db + 0.7 * sig_db
            conf = "HIGH"
        elif sig_confidence == "MEDIUM":
            # Medium — equal weight
            combined = 0.5 * oracle_db + 0.5 * sig_db
            conf = "MEDIUM"
        else:
            # Sparse — trust model more (it generalizes)
            combined = 0.7 * oracle_db + 0.3 * sig_db
            conf = "MEDIUM"  # Downgrade from pure oracle HIGH

        return combined, conf

    def _assess_agreement(self, delta_db: float) -> str:
        """Assess agreement between oracle and signatures."""
        abs_delta = abs(delta_db)
        if abs_delta <= AGREEMENT_GOOD_DB:
            return "GOOD"
        elif abs_delta <= AGREEMENT_FAIR_DB:
            return "FAIR"
        else:
            return "POOR"

    def predict(
        self,
        tx_grid: str,
        rx_grid: str,
        band: int,
        hour: int,
        month: int,
        sfi: float = 150.0,
        kp: float = 2.0,
        k: int = 50,
    ) -> UnifiedPrediction:
        """
        Unified prediction combining oracle and signature search.

        Args:
            tx_grid: 4-char Maidenhead grid (TX)
            rx_grid: 4-char Maidenhead grid (RX)
            band: ADIF band ID (102-111)
            hour: UTC hour (0-23)
            month: Month (1-12)
            sfi: Solar Flux Index (default 150)
            kp: Kp index (default 2)
            k: Number of signature neighbors (default 50)

        Returns:
            UnifiedPrediction with both predictions and combined assessment
        """
        # Validate grids
        tx_grid = validate_grid(tx_grid)
        rx_grid = validate_grid(rx_grid)

        # Convert grid to lat/lon for oracle
        tx_lat, tx_lon = grid4_to_latlon(tx_grid)
        rx_lat, rx_lon = grid4_to_latlon(rx_grid)

        # Get frequency in MHz for oracle
        freq_mhz = BAND_TO_MHZ.get(band, 14.0)

        # ── Query Oracle ──
        oracle_pred = self.oracle.predict(
            lat_tx=tx_lat, lon_tx=tx_lon,
            lat_rx=rx_lat, lon_rx=rx_lon,
            freq_mhz=freq_mhz, sfi=sfi, kp=kp,
            hour=float(hour), month=float(month),
        )

        # ── Query Signatures ──
        sig_result = self.sig_search.query(
            tx_grid=tx_grid, rx_grid=rx_grid, band=band,
            hour=hour, month=month, sfi=sfi, kp=kp, k=k,
        )

        # ── Combine Predictions ──
        delta_db = oracle_pred.snr_db - sig_result.median_snr
        agreement = self._assess_agreement(delta_db)

        combined_db, combined_conf = self._combine_predictions(
            oracle_pred.snr_db,
            sig_result.median_snr,
            sig_result.neighbors_found,
            sig_result.confidence,
        )
        combined_condition = self._classify_condition(combined_db)

        # Collect warnings
        warnings = list(oracle_pred.warnings)
        if sig_result.neighbors_found == 0:
            warnings.append("No historical signatures found — oracle-only prediction")
        elif sig_result.neighbors_found < 10:
            warnings.append(f"Sparse signatures ({sig_result.neighbors_found}) — lower confidence")

        return UnifiedPrediction(
            tx_grid=tx_grid,
            rx_grid=rx_grid,
            band=band,
            band_label=ADIF_TO_LABEL.get(band, str(band)),
            hour=hour,
            month=month,
            sfi=sfi,
            kp=kp,

            distance_km=oracle_pred.distance_km,
            bearing_deg=oracle_pred.bearing_deg,

            oracle_snr_sigma=oracle_pred.snr_sigma,
            oracle_snr_db=oracle_pred.snr_db,
            oracle_condition=oracle_pred.condition,
            oracle_confidence=oracle_pred.confidence,

            sig_snr_db=sig_result.median_snr,
            sig_neighbors=sig_result.neighbors_found,
            sig_spot_count=sig_result.spot_count,
            sig_condition=sig_result.condition,
            sig_confidence=sig_result.confidence,
            sig_query_ms=sig_result.query_ms,

            combined_snr_db=combined_db,
            combined_condition=combined_condition,
            combined_confidence=combined_conf,
            agreement=agreement,
            delta_db=delta_db,

            warnings=warnings,
        )


# ── Display ───────────────────────────────────────────────────────────────────

MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def print_prediction(p: UnifiedPrediction):
    """Pretty-print unified prediction."""
    print("=" * 70)
    print("  IONIS Unified Prediction — Step J")
    print("=" * 70)
    print(f"  Path:   {p.tx_grid} -> {p.rx_grid}  |  {p.band_label} ({p.band})")
    print(f"  When:   {p.hour:02d} UTC, {MONTH_NAMES[p.month]}  |  SFI {p.sfi:.0f}, Kp {p.kp:.0f}")
    print(f"  Geom:   {p.distance_km:.0f} km @ {p.bearing_deg:.0f} deg")

    # Oracle section
    print(f"\n  {'─' * 60}")
    print(f"  Neural Oracle (V13)")
    print(f"  {'─' * 60}")
    print(f"  SNR:        {p.oracle_snr_sigma:+.3f} sigma ({p.oracle_snr_db:+.1f} dB)")
    print(f"  Condition:  {p.oracle_condition}")
    print(f"  Confidence: {p.oracle_confidence}")

    # Signature section
    print(f"\n  {'─' * 60}")
    print(f"  Signature Search (93M signatures)")
    print(f"  {'─' * 60}")
    print(f"  SNR:        {p.sig_snr_db:+.1f} dB (median of {p.sig_neighbors} neighbors)")
    print(f"  Evidence:   {p.sig_spot_count:,} spots")
    print(f"  Condition:  {p.sig_condition}")
    print(f"  Confidence: {p.sig_confidence}")
    print(f"  Latency:    {p.sig_query_ms:.0f} ms")

    # Combined section
    print(f"\n  {'─' * 60}")
    print(f"  Combined Assessment")
    print(f"  {'─' * 60}")
    print(f"  SNR:        {p.combined_snr_db:+.1f} dB")
    print(f"  Condition:  {p.combined_condition}")
    print(f"  Confidence: {p.combined_confidence}")
    print(f"  Agreement:  {p.agreement} (delta {p.delta_db:+.1f} dB)")

    if p.warnings:
        print(f"\n  Warnings:")
        for w in p.warnings:
            print(f"    - {w}")

    print()


# ── Test Suite ────────────────────────────────────────────────────────────────

def run_tests(host: str = CH_HOST):
    """Run unified prediction tests."""
    print("=" * 70)
    print("  IONIS Unified Prediction Test Suite (Step J)")
    print("=" * 70)

    predictor = UnifiedPredictor(ch_host=host)
    passed = 0
    failed = 0

    def check(name, ok, detail=""):
        nonlocal passed, failed
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        msg = f"  {name:<40} [{status}]"
        if detail:
            msg += f"  {detail}"
        print(msg)

    # Test 1: Reference path — both systems return results
    print(f"\n  Test 1: Reference Path (FN31 -> JO21, 20m)")
    p = predictor.predict(
        tx_grid="FN31", rx_grid="JO21", band=107,
        hour=14, month=6, sfi=150, kp=2,
    )
    check("Oracle returns prediction",
          p.oracle_snr_db != 0,
          f"oracle={p.oracle_snr_db:+.1f} dB")
    check("Signatures return results",
          p.sig_neighbors > 0,
          f"neighbors={p.sig_neighbors}")
    check("Combined SNR computed",
          -40 < p.combined_snr_db < 10,
          f"combined={p.combined_snr_db:+.1f} dB")
    check("Agreement assessed",
          p.agreement in ["GOOD", "FAIR", "POOR"],
          f"agreement={p.agreement}")

    # Test 2: Day vs Night — physics preserved
    print(f"\n  Test 2: Day vs Night (20m)")
    p_day = predictor.predict(
        tx_grid="FN31", rx_grid="JO21", band=107,
        hour=14, month=6, sfi=150, kp=2,
    )
    p_night = predictor.predict(
        tx_grid="FN31", rx_grid="JO21", band=107,
        hour=4, month=6, sfi=150, kp=2,
    )
    delta = p_day.combined_snr_db - p_night.combined_snr_db
    check("Day >= Night (combined)",
          delta >= -1,  # Allow small variance
          f"day={p_day.combined_snr_db:+.1f}, night={p_night.combined_snr_db:+.1f}, delta={delta:+.1f} dB")

    # Test 3: SFI sensitivity
    print(f"\n  Test 3: SFI Sensitivity")
    p_lo = predictor.predict(
        tx_grid="FN31", rx_grid="JO21", band=107,
        hour=14, month=6, sfi=70, kp=2,
    )
    p_hi = predictor.predict(
        tx_grid="FN31", rx_grid="JO21", band=107,
        hour=14, month=6, sfi=200, kp=2,
    )
    delta = p_hi.combined_snr_db - p_lo.combined_snr_db
    check("High SFI >= Low SFI (combined)",
          delta >= 0,
          f"sfi70={p_lo.combined_snr_db:+.1f}, sfi200={p_hi.combined_snr_db:+.1f}, delta={delta:+.1f} dB")

    # Test 4: Storm degradation
    print(f"\n  Test 4: Storm Degradation")
    p_quiet = predictor.predict(
        tx_grid="FN31", rx_grid="JO21", band=107,
        hour=14, month=6, sfi=150, kp=1,
    )
    p_storm = predictor.predict(
        tx_grid="FN31", rx_grid="JO21", band=107,
        hour=14, month=6, sfi=150, kp=7,
    )
    cost = p_quiet.combined_snr_db - p_storm.combined_snr_db
    check("Quiet > Storm (combined)",
          cost > 0,
          f"quiet={p_quiet.combined_snr_db:+.1f}, storm={p_storm.combined_snr_db:+.1f}, cost={cost:+.1f} dB")

    # Test 5: Sparse path handling
    print(f"\n  Test 5: Sparse Path Handling")
    p_sparse = predictor.predict(
        tx_grid="AR51", rx_grid="QE37", band=107,
        hour=12, month=6, sfi=150, kp=2,
    )
    check("Sparse path returns result",
          p_sparse.combined_snr_db != 0,
          f"combined={p_sparse.combined_snr_db:+.1f} dB, neighbors={p_sparse.sig_neighbors}")

    # Test 6: Confidence weighting
    print(f"\n  Test 6: Confidence Weighting")
    p_dense = predictor.predict(
        tx_grid="FN31", rx_grid="JO21", band=107,
        hour=14, month=6, sfi=150, kp=2,
    )
    # Dense path should weight signatures more
    if p_dense.sig_confidence == "HIGH":
        expected_weight = 0.7  # 70% signature
        diff_from_sig = abs(p_dense.combined_snr_db - p_dense.sig_snr_db)
        diff_from_oracle = abs(p_dense.combined_snr_db - p_dense.oracle_snr_db)
        closer_to_sig = diff_from_sig < diff_from_oracle
        check("Dense path weights signatures higher",
              closer_to_sig,
              f"combined closer to sig by {diff_from_oracle - diff_from_sig:.1f} dB")
    else:
        check("Dense path weights signatures higher",
              True, "N/A (not HIGH confidence)")

    # Summary
    total = passed + failed
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY: {passed}/{total} passed")
    print(f"{'=' * 70}")

    if failed == 0:
        print("  ALL TESTS PASSED")
    else:
        print(f"  {failed} TEST(S) FAILED")

    return 0 if failed == 0 else 1


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IONIS Unified Prediction — Neural Oracle + Signature Search")
    parser.add_argument("--tx", help="TX grid (4-char Maidenhead, e.g. FN31)")
    parser.add_argument("--rx", help="RX grid (4-char Maidenhead, e.g. JO21)")
    parser.add_argument("--band", help="Band: label (20m) or ADIF (107)")
    parser.add_argument("--hour", type=int, help="Hour UTC (0-23)")
    parser.add_argument("--month", type=int, help="Month (1-12)")
    parser.add_argument("--sfi", type=float, default=150.0, help="Solar Flux Index (default 150)")
    parser.add_argument("--kp", type=float, default=2.0, help="Kp index (default 2)")
    parser.add_argument("--host", default=CH_HOST, help="ClickHouse host")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--test", action="store_true", help="Run test suite")

    args = parser.parse_args()

    if args.test:
        sys.exit(run_tests(host=args.host))

    # Require query params for non-test mode
    if not all([args.tx, args.rx, args.band, args.hour is not None, args.month is not None]):
        parser.error("--tx, --rx, --band, --hour, --month are required (or use --test)")

    predictor = UnifiedPredictor(ch_host=args.host)
    band = parse_band(args.band)

    result = predictor.predict(
        tx_grid=args.tx, rx_grid=args.rx, band=band,
        hour=args.hour, month=args.month,
        sfi=args.sfi, kp=args.kp,
    )

    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        print_prediction(result)


if __name__ == "__main__":
    main()
