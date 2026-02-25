#!/usr/bin/env python3
"""
run_info.py — IONIS V22-gamma System and Model Information

Displays model version, checkpoint details, PhysicsOverrideLayer info,
system configuration, and installed package paths.

Usage:
  python run_info.py
  ionis-validate info
"""

import json
import os
import platform
import sys

import torch
from safetensors.torch import load_file as load_safetensors

# -- Path Setup ----------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V22_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, V22_DIR)

from model import IonisGate, get_device
from physics_override import PhysicsOverrideLayer


def main():
    config_path = os.path.join(V22_DIR, "config_v22.json")
    if not os.path.exists(config_path):
        print(f"  ERROR: Config not found: {config_path}", file=sys.stderr)
        return 1

    with open(config_path) as f:
        config = json.load(f)

    checkpoint_path = os.path.join(V22_DIR, config["checkpoint"])
    device = get_device()

    print()
    print("=" * 60)
    print("  IONIS V22-gamma + PhysicsOverrideLayer — Info")
    print("=" * 60)

    # Model info
    print("\n  MODEL")
    print(f"  {'─' * 50}")
    print(f"  Version:       {config['version']}-{config['variant']} ({config['phase']})")
    print(f"  Architecture:  {config['model']['architecture']}")
    print(f"  DNN dim:       {config['model']['dnn_dim']}")
    print(f"  Hidden dim:    {config['model']['hidden_dim']}")
    print(f"  Input dim:     {config['model']['input_dim']}")
    print(f"  Sidecar hidden: {config['model']['sidecar_hidden']}")

    model = IonisGate(
        dnn_dim=config["model"]["dnn_dim"],
        sidecar_hidden=config["model"]["sidecar_hidden"],
        sfi_idx=config["model"]["sfi_idx"],
        kp_penalty_idx=config["model"]["kp_penalty_idx"],
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:    {param_count:,}")

    # PhysicsOverrideLayer
    override = PhysicsOverrideLayer()
    print(f"\n  PHYSICS OVERRIDE")
    print(f"  {'─' * 50}")
    print(f"  {override.describe()}")

    # Checkpoint info
    print(f"\n  CHECKPOINT")
    print(f"  {'─' * 50}")
    print(f"  Path:          {checkpoint_path}")

    if os.path.exists(checkpoint_path):
        size_bytes = os.path.getsize(checkpoint_path)
        if size_bytes > 1024 * 1024:
            size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            size_str = f"{size_bytes / 1024:.1f} KB"
        print(f"  File size:     {size_str}")

        meta_path = checkpoint_path.replace(".safetensors", "_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}

        epoch = metadata.get("epoch", None)
        val_pearson = metadata.get("val_pearson", None)
        val_rmse = metadata.get("val_rmse", None)
        tst900 = metadata.get("tst900_score", None)
        ki7mt = metadata.get("ki7mt_hard_pass", None)

        if epoch is not None:
            print(f"  Epoch:         {epoch}")
        if val_pearson is not None:
            print(f"  Pearson:       {val_pearson:+.4f}")
        if val_rmse is not None:
            print(f"  RMSE:          {val_rmse:.4f} sigma")
        if tst900 is not None:
            print(f"  TST-900:       {tst900}")
        if ki7mt is not None:
            print(f"  KI7MT:         {ki7mt}")

        date_range = metadata.get("date_range", None)
        sample_size = metadata.get("sample_size", None)
        if date_range:
            print(f"  Date range:    {date_range}")
        if sample_size:
            print(f"  Sample size:   {sample_size:,}")
    else:
        print(f"  Status:        NOT FOUND")

    # Features
    features = config.get("features", [])
    if features:
        print(f"\n  FEATURES ({len(features)})")
        print(f"  {'─' * 50}")
        for i, feat in enumerate(features):
            print(f"  [{i:>2d}] {feat}")

    # System info
    print(f"\n  SYSTEM")
    print(f"  {'─' * 50}")
    print(f"  Python:        {sys.version.split()[0]}")
    print(f"  PyTorch:       {torch.__version__}")
    print(f"  Device:        {device}")
    print(f"  Platform:      {platform.system()} {platform.machine()}")
    print(f"  Hostname:      {platform.node()}")

    if device.type == "cuda":
        print(f"  CUDA version:  {torch.version.cuda}")
        print(f"  GPU:           {torch.cuda.get_device_name(0)}")
    elif device.type == "mps":
        print(f"  MPS:           available")

    # Install paths
    print(f"\n  INSTALL PATHS")
    print(f"  {'─' * 50}")
    print(f"  Config:        {config_path}")
    print(f"  Tests:         {SCRIPT_DIR}")
    print(f"  Model:         {V22_DIR}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
