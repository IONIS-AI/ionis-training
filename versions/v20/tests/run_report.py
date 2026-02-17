#!/usr/bin/env python3
"""
run_report.py — IONIS V20 Beta Test Report Generator

Generates a structured markdown report for pasting into a GitHub Issue.
Collects system info, runs the test suite, and optionally runs custom
path tests — all in one command.

Usage:
  ionis-validate report                        # system info + test results
  ionis-validate report --custom paths.json    # include custom path results
  ionis-validate report --skip-tests           # system info only (fast)
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import torch

# ── Path Setup ───────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V20_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, V20_DIR)

from model import IonisGate, get_device


def generate_install_token(checkpoint_path):
    """Generate a proof-of-installation token.

    Short hash proving the reporter has the checkpoint installed and ran
    this command. Not cryptographic — just enough to distinguish real
    installs from drive-by troll submissions.
    """
    h = hashlib.sha256()
    h.update(b"ionis-v20-beta")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            h.update(f.read(512))
    h.update(platform.node().encode())
    h.update(sys.version.encode())
    return f"ionis-{h.hexdigest()[:12]}"


def collect_system_info():
    """Collect system and model information."""
    config_path = os.path.join(V20_DIR, "config_v20.json")
    if not os.path.exists(config_path):
        return {"error": f"Config not found: {config_path}"}

    with open(config_path) as f:
        config = json.load(f)

    checkpoint_path = os.path.join(V20_DIR, config["checkpoint"])
    device = get_device()

    model = IonisGate(
        dnn_dim=config["model"]["dnn_dim"],
        sidecar_hidden=config["model"]["sidecar_hidden"],
        sfi_idx=config["model"]["sfi_idx"],
        kp_penalty_idx=config["model"]["kp_penalty_idx"],
    )
    param_count = sum(p.numel() for p in model.parameters())

    info = {
        "version": config.get("version", "unknown"),
        "phase": config.get("phase", "unknown"),
        "architecture": config["model"].get("architecture", "IonisGate"),
        "parameters": param_count,
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "device": str(device),
        "platform": f"{platform.system()} {platform.machine()}",
        "hostname": platform.node(),
        "os_release": "",
    }

    # Try to get OS release info
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    info["os_release"] = line.split("=", 1)[1].strip().strip('"')
                    break
    except FileNotFoundError:
        info["os_release"] = f"{platform.system()} {platform.release()}"

    # Checkpoint info
    if os.path.exists(checkpoint_path):
        size_bytes = os.path.getsize(checkpoint_path)
        info["checkpoint_size"] = f"{size_bytes / 1024:.0f} KB"
        cp = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        info["checkpoint_pearson"] = cp.get("val_pearson")
        info["checkpoint_rmse"] = cp.get("val_rmse")
        info["checkpoint_epoch"] = cp.get("epoch")
    else:
        info["checkpoint_size"] = "NOT FOUND"

    if device.type == "cuda":
        info["cuda_version"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0)

    info["install_token"] = generate_install_token(checkpoint_path)

    return info


def run_test_suite():
    """Run the test suite and capture results."""
    run_all = os.path.join(SCRIPT_DIR, "run_all.py")
    try:
        result = subprocess.run(
            [sys.executable, run_all],
            capture_output=True,
            text=True,
            timeout=600,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT: Test suite exceeded 10 minute limit"
    except Exception as e:
        return False, f"ERROR: {e}"


def run_custom_paths(json_path):
    """Run custom path tests and capture results."""
    run_custom = os.path.join(SCRIPT_DIR, "run_custom.py")
    try:
        result = subprocess.run(
            [sys.executable, run_custom, json_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT: Custom path tests exceeded 2 minute limit"
    except Exception as e:
        return False, f"ERROR: {e}"


def generate_report(info, test_passed=None, test_output=None,
                    custom_passed=None, custom_output=None, custom_json=None):
    """Generate markdown report for GitHub Issue."""
    lines = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines.append("## IONIS V20 Beta Test Report")
    lines.append(f"*Generated: {now}*")
    lines.append("")

    # System Info
    lines.append("### System Info")
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append(f"| **Model** | {info.get('version', '?')} ({info.get('phase', '?')}) |")
    lines.append(f"| **Architecture** | {info.get('architecture', '?')} ({info.get('parameters', '?'):,} params) |")

    if info.get("checkpoint_pearson") is not None:
        lines.append(f"| **Pearson** | {info['checkpoint_pearson']:+.4f} |")
    if info.get("checkpoint_rmse") is not None:
        lines.append(f"| **RMSE** | {info['checkpoint_rmse']:.4f} sigma |")

    lines.append(f"| **Checkpoint** | {info.get('checkpoint_size', '?')} |")
    lines.append(f"| **Python** | {info.get('python', '?')} |")
    lines.append(f"| **PyTorch** | {info.get('pytorch', '?')} |")
    lines.append(f"| **Device** | {info.get('device', '?')} |")

    if info.get("gpu"):
        lines.append(f"| **GPU** | {info['gpu']} (CUDA {info.get('cuda_version', '?')}) |")

    lines.append(f"| **OS** | {info.get('os_release', info.get('platform', '?'))} |")
    lines.append(f"| **Hostname** | {info.get('hostname', '?')} |")
    lines.append(f"| **Install Token** | `{info.get('install_token', 'unknown')}` |")
    lines.append("")

    # Test Suite Results
    if test_output is not None:
        status = "PASS" if test_passed else "FAIL"
        lines.append(f"### Test Suite: {status}")
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>Full test output</summary>")
        lines.append("")
        lines.append("```")
        lines.append(test_output.rstrip())
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # Custom Path Results
    if custom_output is not None:
        status = "PASS" if custom_passed else "FAIL"
        lines.append(f"### Custom Paths: {status}")
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>Custom path output</summary>")
        lines.append("")
        lines.append("```")
        lines.append(custom_output.rstrip())
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")

        # Include JSON file contents
        if custom_json:
            lines.append("<details>")
            lines.append("<summary>JSON test file</summary>")
            lines.append("")
            lines.append("```json")
            lines.append(custom_json.rstrip())
            lines.append("```")
            lines.append("")
            lines.append("</details>")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a structured beta test report for GitHub Issues",
    )
    parser.add_argument(
        "--custom",
        metavar="FILE",
        help="Run custom path tests from JSON file and include results",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip the test suite (system info only)",
    )
    args = parser.parse_args()

    print("Collecting system info...", file=sys.stderr)
    info = collect_system_info()

    if "error" in info:
        print(f"ERROR: {info['error']}", file=sys.stderr)
        return 1

    test_passed = None
    test_output = None
    custom_passed = None
    custom_output = None
    custom_json = None

    if not args.skip_tests:
        print("Running test suite...", file=sys.stderr)
        test_passed, test_output = run_test_suite()
        status = "PASS" if test_passed else "FAIL"
        print(f"Test suite: {status}", file=sys.stderr)

    if args.custom:
        if not os.path.exists(args.custom):
            print(f"ERROR: File not found: {args.custom}", file=sys.stderr)
            return 1
        print(f"Running custom paths: {args.custom}", file=sys.stderr)
        custom_passed, custom_output = run_custom_paths(args.custom)
        with open(args.custom) as f:
            custom_json = f.read()
        status = "PASS" if custom_passed else "FAIL"
        print(f"Custom paths: {status}", file=sys.stderr)

    report = generate_report(
        info,
        test_passed=test_passed,
        test_output=test_output,
        custom_passed=custom_passed,
        custom_output=custom_output,
        custom_json=custom_json,
    )

    # Report goes to stdout (for piping/copying)
    print(report)

    print("", file=sys.stderr)
    print("Report printed to stdout. Copy and paste into a GitHub Issue:", file=sys.stderr)
    print("  https://github.com/IONIS-AI/ionis-training/issues/new/choose", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
