#!/usr/bin/env python3
"""
test_tst600_adversarial.py — IONIS V20 Adversarial & Security Tests

TST-600 Group: Tests for robustness against malicious or malformed inputs.

Tests:
  TST-601: Injection via String Coordinates
  TST-602: Extremely Large Values
  TST-603: Negative Physical Values
  TST-604: Type Coercion Attack

These tests verify the oracle rejects malicious inputs gracefully without
crashing or producing exploitable behavior.
"""

import json
import os
import sys

import numpy as np
import torch

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


# ── Secure Oracle ────────────────────────────────────────────────────────────

class SecureIonisOracle:
    """
    Production oracle with security hardening.

    Validates all inputs with strict type checking and bounds validation
    before any processing occurs.
    """

    # Valid ranges
    FREQ_MIN_MHZ = 1.8
    FREQ_MAX_MHZ = 30.0
    LAT_MIN, LAT_MAX = -90.0, 90.0
    LON_MIN, LON_MAX = -180.0, 180.0
    SFI_MIN, SFI_MAX = 50.0, 400.0
    KP_MIN, KP_MAX = 0.0, 9.0
    DISTANCE_MAX = 20015.0  # Half Earth circumference

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def _validate_numeric(self, value, name: str) -> float:
        """
        Validate that a value is a valid numeric type.

        Raises:
            TypeError: If value is not a valid numeric type
            ValueError: If value is NaN or Inf
        """
        # Reject non-numeric types
        if isinstance(value, (list, dict, tuple, set)):
            raise TypeError(
                f"{name} must be a number, got {type(value).__name__}"
            )

        if isinstance(value, str):
            raise TypeError(
                f"{name} must be a number, got string: '{value}'"
            )

        if isinstance(value, bool):
            raise TypeError(
                f"{name} must be a number, got bool"
            )

        # Try to convert to float
        try:
            numeric = float(value)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"{name} cannot be converted to float: {e}"
            )

        # Check for NaN/Inf
        if np.isnan(numeric):
            raise ValueError(f"{name} cannot be NaN")

        if np.isinf(numeric):
            raise ValueError(f"{name} cannot be Inf")

        return numeric

    def _validate_bounds(self, value: float, name: str,
                         min_val: float, max_val: float) -> float:
        """Validate that a value is within bounds."""
        if value < min_val or value > max_val:
            raise ValueError(
                f"{name} value {value} outside valid range [{min_val}, {max_val}]"
            )
        return value

    def predict(self, freq_mhz, tx_lat, tx_lon, rx_lat, rx_lon,
                sfi, kp, hour_utc=12, month=6) -> float:
        """
        Make a secure prediction with full input validation.

        All inputs are type-checked and bounds-validated before processing.

        Raises:
            TypeError: For non-numeric or wrong-type inputs
            ValueError: For out-of-bounds or invalid values
        """
        # Type validation first (catches injection attempts)
        freq_mhz = self._validate_numeric(freq_mhz, "freq_mhz")
        tx_lat = self._validate_numeric(tx_lat, "tx_lat")
        tx_lon = self._validate_numeric(tx_lon, "tx_lon")
        rx_lat = self._validate_numeric(rx_lat, "rx_lat")
        rx_lon = self._validate_numeric(rx_lon, "rx_lon")
        sfi = self._validate_numeric(sfi, "sfi")
        kp = self._validate_numeric(kp, "kp")
        hour_utc = self._validate_numeric(hour_utc, "hour_utc")
        month = self._validate_numeric(month, "month")

        # Bounds validation (catches extreme values)
        freq_mhz = self._validate_bounds(freq_mhz, "freq_mhz",
                                         self.FREQ_MIN_MHZ, self.FREQ_MAX_MHZ)
        tx_lat = self._validate_bounds(tx_lat, "tx_lat", self.LAT_MIN, self.LAT_MAX)
        tx_lon = self._validate_bounds(tx_lon, "tx_lon", self.LON_MIN, self.LON_MAX)
        rx_lat = self._validate_bounds(rx_lat, "rx_lat", self.LAT_MIN, self.LAT_MAX)
        rx_lon = self._validate_bounds(rx_lon, "rx_lon", self.LON_MIN, self.LON_MAX)
        sfi = self._validate_bounds(sfi, "sfi", self.SFI_MIN, self.SFI_MAX)
        kp = self._validate_bounds(kp, "kp", self.KP_MIN, self.KP_MAX)
        hour_utc = self._validate_bounds(hour_utc, "hour_utc", 0, 23)
        month = self._validate_bounds(month, "month", 1, 12)

        # Calculate distance and validate
        distance_km = self._haversine(tx_lat, tx_lon, rx_lat, rx_lon)
        if distance_km > self.DISTANCE_MAX:
            raise ValueError(
                f"Distance {distance_km:.0f} km exceeds maximum {self.DISTANCE_MAX:.0f} km"
            )

        # Build features and predict
        return self._predict_internal(
            freq_mhz, tx_lat, tx_lon, rx_lat, rx_lon,
            sfi, kp, hour_utc, month, distance_km
        )

    def _haversine(self, lat1, lon1, lat2, lon2):
        """Calculate great-circle distance in km."""
        R = 6371.0
        lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    def _predict_internal(self, freq_mhz, tx_lat, tx_lon, rx_lat, rx_lon,
                          sfi, kp, hour_utc, month, distance_km):
        """Internal prediction after validation."""
        # Calculate azimuth
        lat1_r, lat2_r = np.radians(tx_lat), np.radians(rx_lat)
        dlon = np.radians(rx_lon - tx_lon)
        x = np.sin(dlon) * np.cos(lat2_r)
        y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)
        azimuth = (np.degrees(np.arctan2(x, y)) + 360) % 360

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

def test_tst601_injection_attack(oracle):
    """TST-601: Injection via String Coordinates"""
    print("\n" + "=" * 60)
    print("TST-601: Injection via String Coordinates")
    print("=" * 60)

    injection_attempts = [
        ("SQL injection", "51.5; DROP TABLE users;--"),
        ("Command injection", "51.5 && rm -rf /"),
        ("Path traversal", "../../../etc/passwd"),
        ("Script injection", "<script>alert('xss')</script>"),
        ("Null byte", "51.5\x00malicious"),
    ]

    all_rejected = True

    print(f"\n  Testing malicious string inputs as latitude:")
    print(f"  {'-'*50}")

    for name, payload in injection_attempts:
        try:
            oracle.predict(
                freq_mhz=14.0,
                tx_lat=payload,  # Injection attempt
                tx_lon=-77.01,
                rx_lat=51.50,
                rx_lon=-0.12,
                sfi=150, kp=2
            )
            print(f"  {name:<20s}  ACCEPTED (FAIL)")
            all_rejected = False
        except TypeError as e:
            print(f"  {name:<20s}  Rejected (TypeError)")
        except ValueError as e:
            print(f"  {name:<20s}  Rejected (ValueError)")
        except Exception as e:
            print(f"  {name:<20s}  Rejected ({type(e).__name__})")

    if all_rejected:
        print(f"\n  PASS: All injection attempts rejected with clean errors")
        return True
    else:
        print(f"\n  FAIL: Some injection attempts were accepted")
        return False


def test_tst602_extreme_values(oracle):
    """TST-602: Extremely Large Values"""
    print("\n" + "=" * 60)
    print("TST-602: Extremely Large Values")
    print("=" * 60)

    extreme_cases = [
        ("SFI=1e30", dict(sfi=1e30)),
        ("SFI=1e308", dict(sfi=1e308)),
        ("Kp=1e10", dict(kp=1e10)),
        ("freq=1e15 MHz", dict(freq_mhz=1e15)),
        ("lat=1e6", dict(tx_lat=1e6)),
        ("distance via lon", dict(tx_lon=1e10)),
    ]

    all_rejected = True

    print(f"\n  Testing extremely large values:")
    print(f"  {'-'*50}")

    for name, overrides in extreme_cases:
        base_params = dict(
            freq_mhz=14.0,
            tx_lat=39.14, tx_lon=-77.01,
            rx_lat=51.50, rx_lon=-0.12,
            sfi=150, kp=2
        )
        base_params.update(overrides)

        try:
            result = oracle.predict(**base_params)
            print(f"  {name:<20s}  ACCEPTED: {result:.3f} (FAIL)")
            all_rejected = False
        except (ValueError, OverflowError) as e:
            print(f"  {name:<20s}  Rejected (bounds)")
        except Exception as e:
            print(f"  {name:<20s}  Rejected ({type(e).__name__})")

    if all_rejected:
        print(f"\n  PASS: All extreme values rejected before reaching model")
        return True
    else:
        print(f"\n  FAIL: Some extreme values were accepted")
        return False


def test_tst603_negative_values(oracle):
    """TST-603: Negative Physical Values"""
    print("\n" + "=" * 60)
    print("TST-603: Negative Physical Values")
    print("=" * 60)

    negative_cases = [
        ("SFI=-100", dict(sfi=-100)),
        ("Kp=-5", dict(kp=-5)),
        ("freq=-14 MHz", dict(freq_mhz=-14.0)),
        ("hour=-1", dict(hour_utc=-1)),
        ("month=-6", dict(month=-6)),
    ]

    all_rejected = True

    print(f"\n  Testing physically impossible negative values:")
    print(f"  {'-'*50}")

    for name, overrides in negative_cases:
        base_params = dict(
            freq_mhz=14.0,
            tx_lat=39.14, tx_lon=-77.01,
            rx_lat=51.50, rx_lon=-0.12,
            sfi=150, kp=2, hour_utc=12, month=6
        )
        base_params.update(overrides)

        try:
            result = oracle.predict(**base_params)
            print(f"  {name:<20s}  ACCEPTED: {result:.3f} (FAIL)")
            all_rejected = False
        except ValueError as e:
            print(f"  {name:<20s}  Rejected (ValueError)")
        except Exception as e:
            print(f"  {name:<20s}  Rejected ({type(e).__name__})")

    if all_rejected:
        print(f"\n  PASS: All negative physical values rejected")
        return True
    else:
        print(f"\n  FAIL: Some negative values were accepted")
        return False


def test_tst604_type_coercion(oracle):
    """TST-604: Type Coercion Attack"""
    print("\n" + "=" * 60)
    print("TST-604: Type Coercion Attack")
    print("=" * 60)

    type_attacks = [
        ("list", [51.5, 0.0]),
        ("dict", {"value": 51.5}),
        ("tuple", (51.5,)),
        ("set", {51.5}),
        ("bool True", True),
        ("bool False", False),
        ("None", None),
        ("object", object()),
        ("lambda", lambda: 51.5),
    ]

    all_rejected = True

    print(f"\n  Testing wrong-type inputs as latitude:")
    print(f"  {'-'*50}")

    for name, payload in type_attacks:
        try:
            oracle.predict(
                freq_mhz=14.0,
                tx_lat=payload,  # Type attack
                tx_lon=-77.01,
                rx_lat=51.50,
                rx_lon=-0.12,
                sfi=150, kp=2
            )
            print(f"  {name:<15s}  ACCEPTED (FAIL)")
            all_rejected = False
        except TypeError as e:
            print(f"  {name:<15s}  Rejected (TypeError)")
        except ValueError as e:
            print(f"  {name:<15s}  Rejected (ValueError)")
        except Exception as e:
            print(f"  {name:<15s}  Rejected ({type(e).__name__})")

    if all_rejected:
        print(f"\n  PASS: All type coercion attempts rejected")
        return True
    else:
        print(f"\n  FAIL: Some wrong types were silently coerced")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  IONIS V20 — TST-600 Adversarial & Security Tests")
    print("=" * 60)
    print("\n  These tests verify robustness against malicious inputs.")

    # Load model
    print(f"\nLoading {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)

    model = IonisGate(
        dnn_dim=DNN_DIM,
        sidecar_hidden=SIDECAR_HIDDEN,
        sfi_idx=SFI_IDX,
        kp_penalty_idx=KP_PENALTY_IDX,
        gate_init_bias=CONFIG["model"]["gate_init_bias"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    print(f"  Device: {DEVICE}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create secure oracle
    oracle = SecureIonisOracle(model, DEVICE)

    # Run tests
    results = []
    results.append(("TST-601", "Injection Attack", test_tst601_injection_attack(oracle)))
    results.append(("TST-602", "Extreme Values", test_tst602_extreme_values(oracle)))
    results.append(("TST-603", "Negative Values", test_tst603_negative_values(oracle)))
    results.append(("TST-604", "Type Coercion", test_tst604_type_coercion(oracle)))

    # Summary
    print("\n" + "=" * 60)
    print("  TST-600 SUMMARY: Adversarial & Security")
    print("=" * 60)

    passed = sum(1 for _, _, p in results if p)
    total = len(results)

    for test_id, name, p in results:
        status = "PASS" if p else "FAIL"
        print(f"  {test_id}: {name:<20s}  {status}")

    print()
    print(f"  Results: {passed}/{total} passed")

    if passed == total:
        print("\n  ALL TST-600 TESTS PASSED")
        print("  Oracle is hardened against adversarial inputs.")
        return 0
    else:
        print(f"\n  {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
