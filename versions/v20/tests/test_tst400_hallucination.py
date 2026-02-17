#!/usr/bin/env python3
"""
test_tst400_hallucination.py — IONIS V20 Hallucination Trap Tests

TST-400 Group: Catch cases where the model might produce confident but
wrong answers for inputs outside its training domain.

Tests:
  TST-401: EME Path Detection (VHF frequency rejection)
  TST-402: Sporadic E Trap (6m out-of-domain warning)
  TST-403: Ground Wave Confusion (very short path warning)
  TST-404: Extreme Solar Event (out-of-distribution warning)

These tests verify the oracle flags low-confidence scenarios rather than
producing meaningless predictions.
"""

import json
import os
import sys
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors

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

SIGMA_TO_DB = 6.7


# ── Oracle with Hallucination Detection ──────────────────────────────────────

@dataclass
class PredictionResult:
    """Result of a prediction with confidence and warnings."""
    snr_sigma: float
    confidence: str  # "high", "medium", "low"
    warnings: List[str]
    is_valid: bool


class IonisOracleWithWarnings:
    """
    Production oracle with hallucination detection.

    Extends basic validation with warnings for edge cases that are
    technically valid but outside the model's reliable prediction domain.
    """

    # HF band boundaries (MHz)
    FREQ_MIN_MHZ = 1.8    # 160m lower edge
    FREQ_MAX_MHZ = 30.0   # 10m upper edge

    # 6m band (50-54 MHz) - sporadic E territory, not in training
    FREQ_6M_MIN = 50.0
    FREQ_6M_MAX = 54.0

    # VHF/UHF (outside ionospheric propagation domain)
    FREQ_VHF_MIN = 30.0

    # Ground wave threshold
    GROUND_WAVE_DISTANCE_KM = 100.0

    # Extreme solar conditions
    SFI_EXTREME_HIGH = 350.0
    KP_EXTREME_HIGH = 8.0

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def predict(self, freq_mhz, tx_lat, tx_lon, rx_lat, rx_lon,
                sfi, kp, hour_utc=12, month=6) -> PredictionResult:
        """
        Make a prediction with hallucination detection.

        Returns:
            PredictionResult with SNR, confidence level, and any warnings
        """
        warnings = []
        confidence = "high"
        is_valid = True

        # Check for VHF/UHF (hard rejection)
        if freq_mhz >= self.FREQ_VHF_MIN:
            if self.FREQ_6M_MIN <= freq_mhz <= self.FREQ_6M_MAX:
                # 6m is special - sporadic E exists but unpredictable
                warnings.append(
                    f"6m band ({freq_mhz} MHz): Sporadic E propagation is unpredictable. "
                    "Model not trained on 6m data."
                )
                confidence = "low"
                is_valid = False
            else:
                # True VHF/UHF - not ionospheric
                warnings.append(
                    f"Frequency {freq_mhz} MHz is VHF/UHF. "
                    "IONIS predicts HF ionospheric propagation only."
                )
                confidence = "low"
                is_valid = False

        # Check for sub-HF (hard rejection)
        if freq_mhz < self.FREQ_MIN_MHZ:
            warnings.append(
                f"Frequency {freq_mhz} MHz below HF range. "
                "IONIS trained on 1.8-30 MHz only."
            )
            confidence = "low"
            is_valid = False

        # Calculate distance
        distance_km = self._haversine(tx_lat, tx_lon, rx_lat, rx_lon)

        # Check for ground wave confusion
        if distance_km < self.GROUND_WAVE_DISTANCE_KM:
            warnings.append(
                f"Very short path ({distance_km:.0f} km): May be ground wave, "
                "not ionospheric skip. Prediction reliability reduced."
            )
            if confidence == "high":
                confidence = "medium"

        # Check for extreme solar conditions
        if sfi > self.SFI_EXTREME_HIGH:
            warnings.append(
                f"Extreme SFI ({sfi}): Outside normal training distribution. "
                "Prediction may be unreliable."
            )
            if confidence == "high":
                confidence = "medium"

        if kp >= self.KP_EXTREME_HIGH:
            warnings.append(
                f"Severe geomagnetic storm (Kp {kp}): Extreme space weather. "
                "Ionospheric conditions highly disturbed and unpredictable."
            )
            if confidence == "high":
                confidence = "medium"

        # Combined extreme conditions
        if sfi > self.SFI_EXTREME_HIGH and kp >= self.KP_EXTREME_HIGH:
            warnings.append(
                "Extreme solar event: Combined high SFI + severe storm is rare. "
                "Model has limited training data for this scenario."
            )
            confidence = "low"

        # If not valid, return without prediction
        if not is_valid:
            return PredictionResult(
                snr_sigma=float('nan'),
                confidence=confidence,
                warnings=warnings,
                is_valid=False
            )

        # Make prediction
        snr = self._predict_raw(
            freq_mhz, tx_lat, tx_lon, rx_lat, rx_lon,
            sfi, kp, hour_utc, month, distance_km
        )

        return PredictionResult(
            snr_sigma=snr,
            confidence=confidence,
            warnings=warnings,
            is_valid=True
        )

    def _haversine(self, lat1, lon1, lat2, lon2):
        """Calculate great-circle distance in km."""
        R = 6371.0
        lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    def _azimuth(self, lat1, lon1, lat2, lon2):
        """Calculate initial bearing in degrees."""
        lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
        dlon = np.radians(lon2 - lon1)
        x = np.sin(dlon) * np.cos(lat2_r)
        y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)
        return (np.degrees(np.arctan2(x, y)) + 360) % 360

    def _predict_raw(self, freq_mhz, tx_lat, tx_lon, rx_lat, rx_lon,
                     sfi, kp, hour_utc, month, distance_km):
        """Make raw prediction without validation."""
        azimuth = self._azimuth(tx_lat, tx_lon, rx_lat, rx_lon)

        # Build normalized features
        distance = distance_km / 20000.0
        freq_log = np.log10(freq_mhz * 1e6) / 8.0
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
            dtype=torch.float32, device=self.device,
        )

        with torch.no_grad():
            return self.model(features).item()


# ── Test Functions ───────────────────────────────────────────────────────────

def test_tst401_eme_detection(oracle):
    """TST-401: EME Path Detection — VHF should be rejected"""
    print("\n" + "=" * 60)
    print("TST-401: EME Path Detection (144 MHz)")
    print("=" * 60)

    # Classic EME scenario: 2m, ~500 km "path", -28 dB
    # This looks like it could be ionospheric but it's actually lunar reflection
    result = oracle.predict(
        freq_mhz=144.0,  # 2m band
        tx_lat=39.14, tx_lon=-77.01,
        rx_lat=40.0, rx_lon=-74.0,  # ~500 km
        sfi=150, kp=2
    )

    print(f"\n  Scenario: 2m, ~500 km, high power station")
    print(f"  (Classic EME signature that could be mistaken for ionospheric)")
    print()
    print(f"  Valid: {result.is_valid}")
    print(f"  Confidence: {result.confidence}")
    if result.warnings:
        print(f"  Warnings:")
        for w in result.warnings:
            print(f"    - {w}")

    if not result.is_valid and result.confidence == "low":
        print(f"\n  PASS: Oracle correctly rejected VHF as non-ionospheric")
        return True
    else:
        print(f"\n  FAIL: Oracle should reject 144 MHz")
        return False


def test_tst402_sporadic_e(oracle):
    """TST-402: Sporadic E Trap — 6m should warn about uncertainty"""
    print("\n" + "=" * 60)
    print("TST-402: Sporadic E Trap (50 MHz)")
    print("=" * 60)

    # 6m, summer afternoon, ~1500 km - classic sporadic E scenario
    result = oracle.predict(
        freq_mhz=50.3,  # 6m band
        tx_lat=40.0, tx_lon=-100.0,
        rx_lat=35.0, rx_lon=-85.0,  # ~1500 km
        sfi=150, kp=2,
        hour_utc=20, month=7  # Summer afternoon local
    )

    print(f"\n  Scenario: 6m, 1500 km, summer afternoon")
    print(f"  (Classic sporadic E conditions)")
    print()
    print(f"  Valid: {result.is_valid}")
    print(f"  Confidence: {result.confidence}")
    if result.warnings:
        print(f"  Warnings:")
        for w in result.warnings:
            print(f"    - {w}")

    # 6m should be flagged as low confidence / not valid
    if not result.is_valid and "Sporadic E" in str(result.warnings):
        print(f"\n  PASS: Oracle flags 6m sporadic E uncertainty")
        return True
    else:
        print(f"\n  FAIL: Oracle should warn about sporadic E unpredictability")
        return False


def test_tst403_ground_wave(oracle):
    """TST-403: Ground Wave Confusion — very short path should warn"""
    print("\n" + "=" * 60)
    print("TST-403: Ground Wave Confusion (50 km)")
    print("=" * 60)

    # 80m, 50 km path - likely ground wave, not skywave
    result = oracle.predict(
        freq_mhz=3.5,  # 80m
        tx_lat=40.0, tx_lon=-100.0,
        rx_lat=40.4, rx_lon=-99.5,  # ~50 km
        sfi=100, kp=2
    )

    print(f"\n  Scenario: 80m, ~50 km path")
    print(f"  (Likely ground wave, not ionospheric skip)")
    print()
    print(f"  Valid: {result.is_valid}")
    print(f"  Confidence: {result.confidence}")
    print(f"  Prediction: {result.snr_sigma:+.3f} sigma" if result.is_valid else "  Prediction: N/A")
    if result.warnings:
        print(f"  Warnings:")
        for w in result.warnings:
            print(f"    - {w}")

    # Should be valid but with warning and reduced confidence
    ground_wave_warned = any("ground wave" in w.lower() for w in result.warnings)

    if ground_wave_warned and result.confidence in ("medium", "low"):
        print(f"\n  PASS: Oracle warns about ground wave possibility")
        return True
    else:
        print(f"\n  FAIL: Oracle should flag very short paths as potential ground wave")
        return False


def test_tst404_extreme_solar(oracle):
    """TST-404: Extreme Solar Event — should warn about out-of-distribution"""
    print("\n" + "=" * 60)
    print("TST-404: Extreme Solar Event (SFI 400, Kp 9)")
    print("=" * 60)

    # X-class flare scenario: very high SFI + severe storm
    result = oracle.predict(
        freq_mhz=14.0,  # 20m
        tx_lat=39.14, tx_lon=-77.01,
        rx_lat=51.50, rx_lon=-0.12,
        sfi=400, kp=9  # Extreme conditions
    )

    print(f"\n  Scenario: 20m transatlantic during extreme space weather")
    print(f"  SFI 400 + Kp 9 (X-class flare + severe storm)")
    print()
    print(f"  Valid: {result.is_valid}")
    print(f"  Confidence: {result.confidence}")
    print(f"  Prediction: {result.snr_sigma:+.3f} sigma ({result.snr_sigma * SIGMA_TO_DB:+.1f} dB)" if result.is_valid else "  Prediction: N/A")
    if result.warnings:
        print(f"  Warnings:")
        for w in result.warnings:
            print(f"    - {w}")

    # Should produce prediction but with warnings and reduced confidence
    has_extreme_warning = any("extreme" in w.lower() for w in result.warnings)

    if has_extreme_warning and result.confidence == "low":
        print(f"\n  PASS: Oracle flags extreme conditions as low confidence")
        return True
    elif has_extreme_warning:
        print(f"\n  PASS: Oracle warns about extreme conditions (confidence: {result.confidence})")
        return True
    else:
        print(f"\n  FAIL: Oracle should warn about extreme space weather")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  IONIS V20 — TST-400 Hallucination Trap Tests")
    print("=" * 60)
    print("\n  These tests verify the oracle catches out-of-domain queries")
    print("  that would produce meaningless predictions.")

    # Load model (safetensors — no pickle)
    print(f"\nLoading {MODEL_PATH}...")
    state_dict = load_safetensors(MODEL_PATH, device=str(DEVICE))

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

    # Create oracle with warning support
    oracle = IonisOracleWithWarnings(model, DEVICE)

    # Run tests
    results = []
    results.append(("TST-401", "EME Detection", test_tst401_eme_detection(oracle)))
    results.append(("TST-402", "Sporadic E Trap", test_tst402_sporadic_e(oracle)))
    results.append(("TST-403", "Ground Wave", test_tst403_ground_wave(oracle)))
    results.append(("TST-404", "Extreme Solar", test_tst404_extreme_solar(oracle)))

    # Summary
    print("\n" + "=" * 60)
    print("  TST-400 SUMMARY: Hallucination Traps")
    print("=" * 60)

    passed = sum(1 for _, _, p in results if p)
    total = len(results)

    for test_id, name, p in results:
        status = "PASS" if p else "FAIL"
        print(f"  {test_id}: {name:<20s}  {status}")

    print()
    print(f"  Results: {passed}/{total} passed")

    if passed == total:
        print("\n  ALL TST-400 TESTS PASSED")
        print("  Oracle correctly identifies out-of-domain scenarios.")
        return 0
    else:
        print(f"\n  {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
