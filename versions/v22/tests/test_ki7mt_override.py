#!/usr/bin/env python3
"""
test_ki7mt_override.py — IONIS V22-gamma + PhysicsOverrideLayer Validation

Validates that V22-gamma + PhysicsOverrideLayer = 17/17 KI7MT hard pass.

Steps:
    1. Load V22-gamma checkpoint
    2. Run all 18 KI7MT tests RAW (confirm 16/17 baseline)
    3. Run all 18 KI7MT tests WITH override (confirm 17/17)
    4. Confirm zero regressions (no raw PASS -> override FAIL)
    5. Print audit report

Usage:
    python test_ki7mt_override.py
    python test_ki7mt_override.py --quiet

Exit code:
    0 = all gates passed
    1 = validation failed
"""

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
    get_device, build_features, BAND_FREQ_HZ,
    IonisGate, grid4_to_latlon, solar_elevation_deg,
)
from physics_override import apply_override_to_prediction, PhysicsOverrideLayer
from safetensors.torch import load_file as load_safetensors

# -- Constants -----------------------------------------------------------------

V22_CONFIG = os.path.join(V22_DIR, "config_v22.json")
V22_CHECKPOINT = os.path.join(V22_DIR, "ionis_v22_gamma.safetensors")
TEST_PATHS_FILE = os.path.join(SCRIPT_DIR, "ki7mt_test_paths.json")

BAND_ORDER = ["160m", "80m", "40m", "20m", "17m", "15m", "12m", "10m"]


# -- Model Loading -------------------------------------------------------------

def load_v22_gamma(device):
    """Load V22-gamma model."""
    with open(V22_CONFIG) as f:
        config = json.load(f)

    state_dict = load_safetensors(V22_CHECKPOINT, device=str(device))
    model_cfg = config["model"]

    model = IonisGate(
        dnn_dim=model_cfg["dnn_dim"],
        sidecar_hidden=model_cfg["sidecar_hidden"],
        sfi_idx=model_cfg["sfi_idx"],
        kp_penalty_idx=model_cfg["kp_penalty_idx"],
        gate_init_bias=model_cfg.get("gate_init_bias"),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


# -- Prediction ----------------------------------------------------------------

def predict_path(model, config, device, tx_grid, rx_grid, band, hour_utc,
                 month, day_of_year, sfi, kp):
    """Make a raw prediction (no override)."""
    tx_lat, tx_lon = grid4_to_latlon(tx_grid)
    rx_lat, rx_lon = grid4_to_latlon(rx_grid)
    freq_hz = BAND_FREQ_HZ[band]

    features = build_features(
        tx_lat, tx_lon, rx_lat, rx_lon, freq_hz,
        sfi, kp, hour_utc, month,
        day_of_year=day_of_year,
        include_solar_depression=True,
    )

    tensor = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        return model(tensor).item()


def predict_with_override(model, config, device, tx_grid, rx_grid, band,
                          hour_utc, month, day_of_year, sfi, kp):
    """Make a prediction with physics override applied."""
    sigma = predict_path(model, config, device, tx_grid, rx_grid, band,
                         hour_utc, month, day_of_year, sfi, kp)

    tx_lat, tx_lon = grid4_to_latlon(tx_grid)
    rx_lat, rx_lon = grid4_to_latlon(rx_grid)
    freq_mhz = BAND_FREQ_HZ[band] / 1e6
    tx_solar = solar_elevation_deg(tx_lat, tx_lon, hour_utc, day_of_year)
    rx_solar = solar_elevation_deg(rx_lat, rx_lon, hour_utc, day_of_year)

    clamped, was_overridden = apply_override_to_prediction(
        sigma, freq_mhz, tx_solar, rx_solar)

    return clamped, was_overridden, sigma, tx_solar, rx_solar


# -- Test Evaluation -----------------------------------------------------------

def evaluate_test(sigma, test):
    """Evaluate whether a prediction passes a test."""
    expected = test["expected"]
    threshold = test.get("threshold_sigma", 0.0) or 0.0

    if expected == "positive":
        return sigma > threshold
    elif expected == "negative":
        return sigma <= threshold
    elif expected in ("order_day", "order_night"):
        return None
    return False


def run_validation(verbose=True):
    """Run the full Phase 4.0 validation."""

    device = get_device()
    if verbose:
        print(f"Device:     {device}")
        print(f"Config:     {V22_CONFIG}")
        print(f"Checkpoint: {V22_CHECKPOINT}")

    if not os.path.exists(V22_CHECKPOINT):
        print(f"ERROR: Checkpoint not found: {V22_CHECKPOINT}")
        return False

    model, config = load_v22_gamma(device)
    override = PhysicsOverrideLayer()

    if verbose:
        print(f"Override:   {override.describe()}")

    with open(TEST_PATHS_FILE) as f:
        test_data = json.load(f)
    tests = test_data["tests"]

    if verbose:
        print(f"Tests:      {len(tests)} paths")
        print()

    raw_results = {}
    override_results = {}
    override_audit = {}

    for test in tests:
        tid = test["id"]
        expected = test["expected"]

        if expected in ("order_day", "order_night"):
            raw_preds = {}
            ovr_preds = {}
            for b in BAND_ORDER:
                if b == "30m":
                    continue
                raw_sigma = predict_path(
                    model, config, device,
                    test["tx_grid"], test["rx_grid"], b,
                    test["hour_utc"], test["month"], test["day_of_year"],
                    test["sfi"], test["kp"])
                ovr_sigma, was_ovr, _, tx_s, rx_s = predict_with_override(
                    model, config, device,
                    test["tx_grid"], test["rx_grid"], b,
                    test["hour_utc"], test["month"], test["day_of_year"],
                    test["sfi"], test["kp"])
                raw_preds[b] = raw_sigma
                ovr_preds[b] = ovr_sigma

            if expected == "order_day":
                high_raw = np.mean([raw_preds[b] for b in ["10m", "15m", "20m"] if b in raw_preds])
                low_raw = np.mean([raw_preds[b] for b in ["80m", "160m"] if b in raw_preds])
                high_ovr = np.mean([ovr_preds[b] for b in ["10m", "15m", "20m"] if b in ovr_preds])
                low_ovr = np.mean([ovr_preds[b] for b in ["80m", "160m"] if b in ovr_preds])
                raw_pass = high_raw > low_raw
                ovr_pass = high_ovr > low_ovr
            else:
                high_raw = np.mean([raw_preds[b] for b in ["10m", "15m"] if b in raw_preds])
                low_raw = np.mean([raw_preds[b] for b in ["40m", "80m", "160m"] if b in raw_preds])
                high_ovr = np.mean([ovr_preds[b] for b in ["10m", "15m"] if b in ovr_preds])
                low_ovr = np.mean([ovr_preds[b] for b in ["40m", "80m", "160m"] if b in ovr_preds])
                raw_pass = low_raw > high_raw
                ovr_pass = low_ovr > high_ovr

            raw_results[tid] = {"passed": raw_pass, "confidence": test["confidence"]}
            override_results[tid] = {"passed": ovr_pass, "confidence": test["confidence"]}
            override_audit[tid] = {"type": "ordering", "raw_preds": raw_preds, "ovr_preds": ovr_preds}
            continue

        raw_sigma = predict_path(
            model, config, device,
            test["tx_grid"], test["rx_grid"], test["band"],
            test["hour_utc"], test["month"], test["day_of_year"],
            test["sfi"], test["kp"])

        ovr_sigma, was_ovr, orig_sigma, tx_solar, rx_solar = predict_with_override(
            model, config, device,
            test["tx_grid"], test["rx_grid"], test["band"],
            test["hour_utc"], test["month"], test["day_of_year"],
            test["sfi"], test["kp"])

        raw_pass = evaluate_test(raw_sigma, test)
        ovr_pass = evaluate_test(ovr_sigma, test)

        raw_results[tid] = {
            "passed": raw_pass,
            "confidence": test["confidence"],
            "sigma": raw_sigma,
        }
        override_results[tid] = {
            "passed": ovr_pass,
            "confidence": test["confidence"],
            "sigma": ovr_sigma,
        }
        override_audit[tid] = {
            "type": "single",
            "raw_sigma": raw_sigma,
            "ovr_sigma": ovr_sigma,
            "was_overridden": was_ovr,
            "tx_solar": tx_solar,
            "rx_solar": rx_solar,
            "freq_mhz": BAND_FREQ_HZ.get(test["band"], 0) / 1e6,
        }

    if verbose:
        print("=" * 72)
        print("  KI7MT OVERRIDE VALIDATION: V22-gamma + PhysicsOverrideLayer")
        print("=" * 72)
        print()
        print(f"  {'Test':<14s}  {'Raw':>8s}  {'Override':>8s}  {'OVR?':>5s}  Name")
        print(f"  {'─' * 14}  {'─' * 8}  {'─' * 8}  {'─' * 5}  {'─' * 30}")

    raw_hard_pass = 0
    raw_hard_total = 0
    ovr_hard_pass = 0
    ovr_hard_total = 0
    regressions = 0
    overrides_fired = 0

    for test in tests:
        tid = test["id"]
        raw = raw_results[tid]
        ovr = override_results[tid]
        audit = override_audit[tid]
        conf = test["confidence"]

        if conf == "hard":
            raw_hard_total += 1
            ovr_hard_total += 1
            if raw["passed"]:
                raw_hard_pass += 1
            if ovr["passed"]:
                ovr_hard_pass += 1

        if raw["passed"] and not ovr["passed"]:
            regressions += 1

        was_ovr = audit.get("was_overridden", False)
        if was_ovr:
            overrides_fired += 1

        if verbose:
            if audit["type"] == "single":
                raw_str = f"{raw['sigma']:+.3f}s"
                ovr_str = f"{ovr['sigma']:+.3f}s"
                ovr_mark = "YES" if was_ovr else ""
            else:
                raw_str = "order"
                ovr_str = "order"
                ovr_mark = ""

            raw_tag = "PASS" if raw["passed"] else "FAIL"
            ovr_tag = "PASS" if ovr["passed"] else "FAIL"
            soft = " [soft]" if conf != "hard" else ""

            flip = ""
            if not raw["passed"] and ovr["passed"]:
                flip = " << FIXED"
            elif raw["passed"] and not ovr["passed"]:
                flip = " << REGRESSION"

            print(f"  {tid:<14s}  {raw_str:>8s} {raw_tag:4s}  "
                  f"{ovr_str:>8s} {ovr_tag:4s}  {ovr_mark:>5s}  "
                  f"{test['name']}{soft}{flip}")

    if verbose:
        print()
        print("=" * 72)
        print("  SUMMARY")
        print("=" * 72)
        print(f"  Raw baseline:      {raw_hard_pass}/{raw_hard_total} hard pass")
        print(f"  With override:     {ovr_hard_pass}/{ovr_hard_total} hard pass")
        print(f"  Regressions:       {regressions}")
        print(f"  Overrides fired:   {overrides_fired}")

        acid_audit = override_audit.get("KI7MT-005", {})
        if acid_audit.get("type") == "single":
            print()
            print(f"  ACID TEST (KI7MT-005):")
            print(f"    Raw prediction:  {acid_audit['raw_sigma']:+.3f}s")
            print(f"    Override result:  {acid_audit['ovr_sigma']:+.3f}s")
            print(f"    TX solar:        {acid_audit['tx_solar']:+.1f} deg")
            print(f"    RX solar:        {acid_audit['rx_solar']:+.1f} deg")
            print(f"    Freq:            {acid_audit['freq_mhz']:.1f} MHz")
            print(f"    Override fired:  {acid_audit['was_overridden']}")

    gates = []
    gate1 = raw_hard_pass == 16
    gates.append(("Raw baseline = 16/17", gate1))
    gate2 = ovr_hard_pass == ovr_hard_total
    gates.append((f"Override = {ovr_hard_total}/{ovr_hard_total}", gate2))
    gate3 = regressions == 0
    gates.append(("Zero regressions", gate3))
    acid_raw = raw_results.get("KI7MT-005", {})
    acid_ovr = override_results.get("KI7MT-005", {})
    gate4 = (not acid_raw.get("passed", True)) and acid_ovr.get("passed", False)
    gates.append(("Acid test FAIL->PASS", gate4))

    if verbose:
        print()
        print("  GATES:")
        for name, passed in gates:
            status = "PASS" if passed else "FAIL"
            print(f"    [{status}] {name}")

        all_pass = all(g[1] for g in gates)
        print()
        if all_pass:
            print("  VERDICT: PHASE 4.0 VALIDATED")
            print("  V22-gamma + PhysicsOverrideLayer is production-ready.")
        else:
            print("  VERDICT: PHASE 4.0 VALIDATION FAILED")
            print("  Do not proceed. Investigate failures above.")

    return all(g[1] for g in gates)


def main():
    quiet = "--quiet" in sys.argv or "-q" in sys.argv
    passed = run_validation(verbose=not quiet)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
