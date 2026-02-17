#!/usr/bin/env python3
"""
test_tst300_input_validation.py — IONIS V20 Input Validation Tests

TST-300 Group: Verify the oracle rejects invalid inputs gracefully.

Tests:
  TST-301: VHF Frequency Rejection (144 MHz - EME trap)
  TST-302: UHF Frequency Rejection (432 MHz)
  TST-303: Invalid Latitude Rejection (95°)
  TST-304: Invalid Kp Rejection (15)
  TST-305: Valid Long Distance Path Acceptance

These tests verify input boundary checking before predictions reach the model.
"""

import json
import os
import sys

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


# ── Oracle with Input Validation ─────────────────────────────────────────────

class IonisOracle:
    """
    Production oracle wrapper with input validation.

    Validates all inputs before prediction to catch out-of-domain queries
    that would produce meaningless results.
    """

    # HF band boundaries (MHz)
    FREQ_MIN_MHZ = 1.8    # 160m lower edge
    FREQ_MAX_MHZ = 30.0   # 10m upper edge

    # Coordinate bounds
    LAT_MIN = -90.0
    LAT_MAX = 90.0
    LON_MIN = -180.0
    LON_MAX = 180.0

    # Solar/geomagnetic bounds
    SFI_MIN = 50.0
    SFI_MAX = 400.0
    KP_MIN = 0.0
    KP_MAX = 9.0

    # Distance bounds (km)
    DISTANCE_MIN = 0.0
    DISTANCE_MAX = 20015.0  # Half Earth circumference

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def validate_inputs(self, freq_mhz, tx_lat, tx_lon, rx_lat, rx_lon,
                        sfi, kp, distance_km=None):
        """
        Validate all inputs are within acceptable bounds.

        Raises:
            ValueError: If any input is outside valid range
        """
        # Frequency validation
        if freq_mhz < self.FREQ_MIN_MHZ or freq_mhz > self.FREQ_MAX_MHZ:
            raise ValueError(
                f"Frequency {freq_mhz} MHz outside HF range "
                f"[{self.FREQ_MIN_MHZ}, {self.FREQ_MAX_MHZ}] MHz. "
                f"IONIS is trained on HF ionospheric propagation only."
            )

        # Latitude validation
        for name, lat in [("TX", tx_lat), ("RX", rx_lat)]:
            if lat < self.LAT_MIN or lat > self.LAT_MAX:
                raise ValueError(
                    f"{name} latitude {lat}° outside valid range "
                    f"[{self.LAT_MIN}, {self.LAT_MAX}]°"
                )

        # Longitude validation
        for name, lon in [("TX", tx_lon), ("RX", rx_lon)]:
            if lon < self.LON_MIN or lon > self.LON_MAX:
                raise ValueError(
                    f"{name} longitude {lon}° outside valid range "
                    f"[{self.LON_MIN}, {self.LON_MAX}]°"
                )

        # SFI validation
        if sfi < self.SFI_MIN or sfi > self.SFI_MAX:
            raise ValueError(
                f"SFI {sfi} outside valid range [{self.SFI_MIN}, {self.SFI_MAX}]"
            )

        # Kp validation
        if kp < self.KP_MIN or kp > self.KP_MAX:
            raise ValueError(
                f"Kp {kp} outside valid range [{self.KP_MIN}, {self.KP_MAX}]"
            )

        # Distance validation (if provided)
        if distance_km is not None:
            if distance_km < self.DISTANCE_MIN or distance_km > self.DISTANCE_MAX:
                raise ValueError(
                    f"Distance {distance_km} km outside valid range "
                    f"[{self.DISTANCE_MIN}, {self.DISTANCE_MAX}] km"
                )

    def predict(self, freq_mhz, tx_lat, tx_lon, rx_lat, rx_lon,
                sfi, kp, hour_utc=12, month=6, distance_km=None, azimuth=None):
        """
        Make a validated prediction.

        Args:
            freq_mhz: Frequency in MHz (must be 1.8-30.0 for HF)
            tx_lat, tx_lon: Transmitter coordinates
            rx_lat, rx_lon: Receiver coordinates
            sfi: Solar Flux Index (50-400)
            kp: Geomagnetic Kp index (0-9)
            hour_utc: Hour of day UTC (0-23)
            month: Month (1-12)
            distance_km: Path distance (optional, calculated if not provided)
            azimuth: Path azimuth (optional, calculated if not provided)

        Returns:
            Predicted SNR in sigma units

        Raises:
            ValueError: If inputs are outside valid range
        """
        # Validate inputs first
        self.validate_inputs(freq_mhz, tx_lat, tx_lon, rx_lat, rx_lon, sfi, kp, distance_km)

        # Calculate distance if not provided
        if distance_km is None:
            distance_km = self._haversine(tx_lat, tx_lon, rx_lat, rx_lon)

        # Calculate azimuth if not provided
        if azimuth is None:
            azimuth = self._azimuth(tx_lat, tx_lon, rx_lat, rx_lon)

        # Build feature vector
        features = self._build_features(
            distance_km, freq_mhz * 1e6, hour_utc, month, azimuth,
            tx_lat, tx_lon, rx_lat, rx_lon, sfi, kp
        )

        # Run prediction
        with torch.no_grad():
            snr = self.model(features).item()

        return snr

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

    def _build_features(self, distance_km, freq_hz, hour, month, azimuth,
                        tx_lat, tx_lon, rx_lat, rx_lon, sfi, kp):
        """Build normalized feature vector matching training format."""
        distance = distance_km / 20000.0
        freq_log = np.log10(freq_hz) / 8.0
        hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
        hour_cos = np.cos(2.0 * np.pi * hour / 24.0)
        az_sin = np.sin(2.0 * np.pi * azimuth / 360.0)
        az_cos = np.cos(2.0 * np.pi * azimuth / 360.0)
        lat_diff = abs(tx_lat - rx_lat) / 180.0
        midpoint_lat = (tx_lat + rx_lat) / 2.0 / 90.0
        season_sin = np.sin(2.0 * np.pi * month / 12.0)
        season_cos = np.cos(2.0 * np.pi * month / 12.0)
        midpoint_lon = (tx_lon + rx_lon) / 2.0
        local_solar_h = hour + midpoint_lon / 15.0
        day_night_est = np.cos(2.0 * np.pi * local_solar_h / 24.0)
        sfi_norm = sfi / 300.0
        kp_penalty = 1.0 - kp / 9.0

        return torch.tensor(
            [[distance, freq_log, hour_sin, hour_cos,
              az_sin, az_cos, lat_diff, midpoint_lat,
              season_sin, season_cos, day_night_est,
              sfi_norm, kp_penalty]],
            dtype=torch.float32, device=self.device,
        )


# ── Test Functions ───────────────────────────────────────────────────────────

def test_tst301_vhf_rejection(oracle):
    """TST-301: VHF Frequency Rejection (EME Trap)"""
    print("\n" + "=" * 60)
    print("TST-301: VHF Frequency Rejection (144 MHz)")
    print("=" * 60)

    try:
        # Attempt prediction at 144 MHz (2m band - EME territory)
        oracle.predict(
            freq_mhz=144.0,
            tx_lat=39.14, tx_lon=-77.01,
            rx_lat=51.50, rx_lon=-0.12,
            sfi=150, kp=2
        )
        print("  FAIL: Oracle accepted VHF frequency without error")
        return False
    except ValueError as e:
        print(f"  Caught: {e}")
        print("  PASS: Oracle correctly rejected VHF frequency")
        return True


def test_tst302_uhf_rejection(oracle):
    """TST-302: UHF Frequency Rejection"""
    print("\n" + "=" * 60)
    print("TST-302: UHF Frequency Rejection (432 MHz)")
    print("=" * 60)

    try:
        oracle.predict(
            freq_mhz=432.0,
            tx_lat=39.14, tx_lon=-77.01,
            rx_lat=51.50, rx_lon=-0.12,
            sfi=150, kp=2
        )
        print("  FAIL: Oracle accepted UHF frequency without error")
        return False
    except ValueError as e:
        print(f"  Caught: {e}")
        print("  PASS: Oracle correctly rejected UHF frequency")
        return True


def test_tst303_invalid_latitude(oracle):
    """TST-303: Invalid Latitude Rejection"""
    print("\n" + "=" * 60)
    print("TST-303: Invalid Latitude Rejection (95°)")
    print("=" * 60)

    try:
        oracle.predict(
            freq_mhz=14.0,
            tx_lat=95.0, tx_lon=-77.01,  # Impossible latitude
            rx_lat=51.50, rx_lon=-0.12,
            sfi=150, kp=2
        )
        print("  FAIL: Oracle accepted impossible latitude without error")
        return False
    except ValueError as e:
        print(f"  Caught: {e}")
        print("  PASS: Oracle correctly rejected invalid latitude")
        return True


def test_tst304_invalid_kp(oracle):
    """TST-304: Invalid Kp Rejection"""
    print("\n" + "=" * 60)
    print("TST-304: Invalid Kp Rejection (Kp=15)")
    print("=" * 60)

    try:
        oracle.predict(
            freq_mhz=14.0,
            tx_lat=39.14, tx_lon=-77.01,
            rx_lat=51.50, rx_lon=-0.12,
            sfi=150, kp=15  # Kp max is 9
        )
        print("  FAIL: Oracle accepted invalid Kp without error")
        return False
    except ValueError as e:
        print(f"  Caught: {e}")
        print("  PASS: Oracle correctly rejected invalid Kp")
        return True


def test_tst305_valid_long_path(oracle):
    """TST-305: Valid Long Distance Path"""
    print("\n" + "=" * 60)
    print("TST-305: Valid Long Distance Path (~12,000 km)")
    print("=" * 60)

    try:
        # W3 to Asia - valid long path
        snr = oracle.predict(
            freq_mhz=14.0,
            tx_lat=39.14, tx_lon=-77.01,   # W3 area
            rx_lat=35.68, rx_lon=139.69,   # Tokyo
            sfi=150, kp=2
        )
        print(f"  Prediction returned: {snr:+.3f} sigma")
        print("  PASS: Oracle accepted valid long-distance path")
        return True
    except ValueError as e:
        print(f"  FAIL: Oracle rejected valid path: {e}")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  IONIS V20 — TST-300 Input Validation Tests")
    print("=" * 60)

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

    # Create oracle wrapper
    oracle = IonisOracle(model, DEVICE)

    # Run tests
    results = []
    results.append(("TST-301", "VHF Frequency Rejection", test_tst301_vhf_rejection(oracle)))
    results.append(("TST-302", "UHF Frequency Rejection", test_tst302_uhf_rejection(oracle)))
    results.append(("TST-303", "Invalid Latitude Rejection", test_tst303_invalid_latitude(oracle)))
    results.append(("TST-304", "Invalid Kp Rejection", test_tst304_invalid_kp(oracle)))
    results.append(("TST-305", "Valid Long Distance Path", test_tst305_valid_long_path(oracle)))

    # Summary
    print("\n" + "=" * 60)
    print("  TST-300 SUMMARY: Input Validation")
    print("=" * 60)

    passed = sum(1 for _, _, p in results if p)
    total = len(results)

    for test_id, name, p in results:
        status = "PASS" if p else "FAIL"
        print(f"  {test_id}: {name:<30s}  {status}")

    print()
    print(f"  Results: {passed}/{total} passed")

    if passed == total:
        print("\n  ALL TST-300 TESTS PASSED")
        return 0
    else:
        print(f"\n  {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
