#!/usr/bin/env python3
"""
run_all.py — IONIS V20 Complete Test Suite Orchestrator

Runs all TST-XXX test groups and produces a consolidated summary.

Test Groups:
  TST-100: Canonical Paths (30 tests)
  TST-200: Physics Constraints (6 tests)
  TST-300: Input Validation (5 tests)
  TST-400: Hallucination Traps (4 tests)
  TST-500: Model Robustness (7 tests)
  TST-600: Adversarial & Security (4 tests)
  TST-700: Bias & Fairness (3 tests)
  TST-800: Regression Tests (3 tests)

Total: 62 tests

Usage:
  python run_all.py           # Run all tests
  python run_all.py --quick   # Skip slow tests (future)
"""

import subprocess
import sys
import os
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Test modules in execution order
TEST_MODULES = [
    ("TST-200", "Physics Constraints", "test_tst200_physics.py", 6),
    ("TST-300", "Input Validation", "test_tst300_input_validation.py", 5),
    ("TST-500", "Model Robustness", "test_tst500_robustness.py", 7),
    ("TST-800", "Regression Tests", "test_tst800_regression.py", 3),
    ("TST-100", "Canonical Paths", "test_tst100_canonical.py", 30),
    ("TST-400", "Hallucination Traps", "test_tst400_hallucination.py", 4),
    ("TST-600", "Adversarial & Security", "test_tst600_adversarial.py", 4),
    ("TST-700", "Bias & Fairness", "test_tst700_bias.py", 3),
]


def run_test_module(script_name: str) -> tuple[bool, str]:
    """Run a test module and capture output."""
    script_path = os.path.join(SCRIPT_DIR, script_name)

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per module
        )
        output = result.stdout + result.stderr
        passed = result.returncode == 0
        return passed, output
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT: Test module exceeded 5 minute limit"
    except Exception as e:
        return False, f"ERROR: {e}"


def main():
    print("=" * 70)
    print("  IONIS V20 — Complete Test Suite")
    print("=" * 70)
    print(f"\n  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model: IONIS V20")
    print(f"  Checkpoint: versions/v20/ionis_v20.safetensors")
    print()

    total_tests = sum(count for _, _, _, count in TEST_MODULES)
    print(f"  Running {len(TEST_MODULES)} test groups ({total_tests} total tests)...")
    print()

    results = []
    all_outputs = []

    for group_id, group_name, script, test_count in TEST_MODULES:
        print(f"  [{group_id}] {group_name} ({test_count} tests)...", end=" ", flush=True)

        passed, output = run_test_module(script)
        results.append((group_id, group_name, test_count, passed))
        all_outputs.append((group_id, output))

        status = "PASS" if passed else "FAIL"
        print(status)

    # Summary
    print()
    print("=" * 70)
    print("  TEST SUITE SUMMARY")
    print("=" * 70)
    print()

    passed_groups = sum(1 for _, _, _, p in results if p)
    passed_tests = sum(count for _, _, count, p in results if p)

    print(f"  {'Group':<10s}  {'Description':<25s}  {'Tests':>6s}  {'Status':>8s}")
    print(f"  {'-'*55}")

    for group_id, group_name, test_count, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {group_id:<10s}  {group_name:<25s}  {test_count:>6d}  {status:>8s}")

    print(f"  {'-'*55}")
    print(f"  {'TOTAL':<10s}  {'':<25s}  {total_tests:>6d}  {passed_tests}/{total_tests}")
    print()

    if passed_groups == len(TEST_MODULES):
        print("  " + "=" * 50)
        print("  ALL TEST GROUPS PASSED")
        print("  " + "=" * 50)
        print()
        print("  IONIS V20 validation complete.")
        print("  Model is ready for production deployment.")
        print()
        return 0
    else:
        failed_groups = [g for g, _, _, p in results if not p]
        print("  " + "=" * 50)
        print(f"  {len(failed_groups)} TEST GROUP(S) FAILED: {', '.join(failed_groups)}")
        print("  " + "=" * 50)
        print()
        print("  Review individual test output for details.")
        print()

        # Show failed group outputs
        for group_id, output in all_outputs:
            if any(g == group_id and not p for g, _, _, p in results):
                print(f"\n  --- {group_id} Output ---")
                print(output)

        return 1


if __name__ == "__main__":
    sys.exit(main())
