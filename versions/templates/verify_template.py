#!/usr/bin/env python3
"""
verify_template.py — Physics Verification Template (4 tests)

Verifies the dual monotonic sidecars work correctly:
1. Storm sidecar: Kp 0→9 should COST SNR (positive delta)
2. Sun sidecar: SFI 70→200 should BENEFIT SNR (positive delta)
3. Gate range: Both gates should be in [0.5, 2.0]
4. Decomposition: base + sun_gate*sun_boost + storm_gate*storm_boost = total

Usage:
    python verify_template.py v18
    python verify_template.py v17
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add templates to path
sys.path.insert(0, str(Path(__file__).parent))
from common import (
    load_config, load_model, DEVICE, INPUT_DIM, DNN_DIM, KP_PENALTY_IDX
)


def test_storm_sidecar(model, norm_std: float, config) -> bool:
    """Test 1: Kp 0→9 should COST SNR."""
    kp0_effect = model.model.get_storm_effect(1.0)   # Kp=0 (quiet)
    kp9_effect = model.model.get_storm_effect(0.0)   # Kp=9 (severe storm)

    storm_cost_sigma = kp0_effect - kp9_effect
    storm_cost_db = storm_cost_sigma * norm_std

    print(f"  Storm sidecar:")
    print(f"    Kp=0 effect: {kp0_effect:+.4f}σ")
    print(f"    Kp=9 effect: {kp9_effect:+.4f}σ")
    print(f"    Storm cost (Kp 0→9): {storm_cost_sigma:+.4f}σ ({storm_cost_db:+.1f} dB)")

    min_storm = config.validation.get("kp_storm_min", 2.0)
    if storm_cost_sigma > min_storm:
        print(f"    PASS: Storm cost {storm_cost_sigma:.2f}σ > {min_storm}σ")
        return True
    else:
        print(f"    FAIL: Storm cost {storm_cost_sigma:.2f}σ <= {min_storm}σ")
        return False


def test_sun_sidecar(model, norm_std: float, config) -> bool:
    """Test 2: SFI 70→200 should BENEFIT SNR."""
    sfi_70 = model.model.get_sun_effect(70.0 / 300.0)
    sfi_200 = model.model.get_sun_effect(200.0 / 300.0)

    sfi_benefit_sigma = sfi_200 - sfi_70
    sfi_benefit_db = sfi_benefit_sigma * norm_std

    print(f"  Sun sidecar:")
    print(f"    SFI=70 effect:  {sfi_70:+.4f}σ")
    print(f"    SFI=200 effect: {sfi_200:+.4f}σ")
    print(f"    SFI benefit (70→200): {sfi_benefit_sigma:+.4f}σ ({sfi_benefit_db:+.1f} dB)")

    min_sfi = config.validation.get("sfi_benefit_min", 0.3)
    if sfi_benefit_sigma > min_sfi:
        print(f"    PASS: SFI benefit {sfi_benefit_sigma:.2f}σ > {min_sfi}σ")
        return True
    else:
        print(f"    FAIL: SFI benefit {sfi_benefit_sigma:.2f}σ <= {min_sfi}σ")
        return False


def test_gate_range(model) -> bool:
    """Test 3: Gates should be in range [0.5, 2.0]."""
    test_inputs = []
    for distance in [1000, 5000, 10000, 15000]:
        for freq_log in [0.8, 0.85, 0.9]:
            for hour in [0, 6, 12, 18]:
                x = np.zeros(INPUT_DIM, dtype=np.float32)
                x[0] = distance / 20000.0
                x[1] = freq_log
                x[2] = np.sin(2 * np.pi * hour / 24)
                x[3] = np.cos(2 * np.pi * hour / 24)
                x[11] = 150.0 / 300.0
                x[12] = 1.0 - 3.0 / 9.0
                test_inputs.append(x)

    X = torch.tensor(np.array(test_inputs), dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        _, _, sun_gates, storm_gates, _, _ = model.model.forward_with_components(X)

    sun_min, sun_max = sun_gates.min().item(), sun_gates.max().item()
    storm_min, storm_max = storm_gates.min().item(), storm_gates.max().item()

    print(f"  Gate ranges:")
    print(f"    Sun gate:   [{sun_min:.4f}, {sun_max:.4f}]")
    print(f"    Storm gate: [{storm_min:.4f}, {storm_max:.4f}]")

    in_range = (0.5 <= sun_min <= sun_max <= 2.0) and (0.5 <= storm_min <= storm_max <= 2.0)
    if in_range:
        print(f"    PASS: All gates in expected range [0.5, 2.0]")
        return True
    else:
        print(f"    FAIL: Gates outside expected range")
        return False


def test_decomposition(model) -> bool:
    """Test 4: base + sun_gate*sun_boost + storm_gate*storm_boost = total."""
    x = np.zeros(INPUT_DIM, dtype=np.float32)
    x[0] = 5000 / 20000.0
    x[1] = 0.875
    x[2] = 0.0
    x[3] = 1.0
    x[11] = 150.0 / 300.0
    x[12] = 1.0 - 2.0 / 9.0

    X = torch.tensor([x], dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        total, base, sun_gate, storm_gate, sun_boost, storm_boost = model.model.forward_with_components(X)

    reconstructed = base + sun_gate * sun_boost + storm_gate * storm_boost
    diff = abs(total.item() - reconstructed.item())

    print(f"  Decomposition:")
    print(f"    Base SNR:    {base.item():+.4f}σ")
    print(f"    Sun gate:    {sun_gate.item():.4f}")
    print(f"    Sun boost:   {sun_boost.item():+.4f}σ")
    print(f"    Storm gate:  {storm_gate.item():.4f}")
    print(f"    Storm boost: {storm_boost.item():+.4f}σ")
    print(f"    Total:       {total.item():+.4f}σ")
    print(f"    Reconstructed: {reconstructed.item():+.4f}σ")
    print(f"    Difference:  {diff:.6f}")

    if diff < 1e-5:
        print(f"    PASS: Decomposition matches (diff < 1e-5)")
        return True
    else:
        print(f"    FAIL: Decomposition mismatch")
        return False


def main():
    parser = argparse.ArgumentParser(description="IONIS Physics Verification")
    parser.add_argument("version", help="Version to test (e.g., v18)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  IONIS {args.version.upper()} Physics Verification (4 tests)")
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

    # Run tests
    results = []

    print("Test 1: Storm Sidecar (Kp sensitivity)")
    results.append(test_storm_sidecar(model, model.norm_std, config))
    print()

    print("Test 2: Sun Sidecar (SFI sensitivity)")
    results.append(test_sun_sidecar(model, model.norm_std, config))
    print()

    print("Test 3: Gate Range")
    results.append(test_gate_range(model))
    print()

    print("Test 4: Decomposition")
    results.append(test_decomposition(model))
    print()

    # Summary
    passed = sum(results)
    total = len(results)

    print("=" * 70)
    print(f"  RESULTS: {passed}/{total} tests passed")
    print("=" * 70)

    if passed == total:
        print("  STATUS: PASS")
        return 0
    else:
        print("  STATUS: FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
