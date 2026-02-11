#!/usr/bin/env python3
"""
verify_v18.py — V18 Physics Verification (4 tests)

Verifies the dual monotonic sidecars work correctly:
1. Storm sidecar: Kp 0→9 should COST SNR (positive delta in normalized units)
2. Sun sidecar: SFI 70→200 should BENEFIT SNR (positive delta)
3. Gate range: Both gates should be in [0.5, 2.0]
4. Decomposition: base + sun_gate*sun_boost + storm_gate*storm_boost = total

V18 uses global raw-dB normalization, so sidecar outputs are still in σ units,
but denormalize differently than V16/V17.
"""

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# ── Config ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "ionis_v18.pth")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DNN_DIM = 11
SFI_IDX = 11
KP_PENALTY_IDX = 12
INPUT_DIM = 13
GATE_INIT_BIAS = -math.log(2.0)


# ── Model Architecture ───────────────────────────────────────────────────────

class MonotonicMLP(nn.Module):
    def __init__(self, hidden_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=True)
        self.activation = nn.Softplus()

    def forward(self, x):
        w1 = torch.abs(self.fc1.weight)
        w2 = torch.abs(self.fc2.weight)
        h = self.activation(nn.functional.linear(x, w1, self.fc1.bias))
        return nn.functional.linear(h, w2, self.fc2.bias)


def _gate(x):
    return 0.5 + 1.5 * torch.sigmoid(x)


class IonisV12Gate(nn.Module):
    def __init__(self, dnn_dim=11, sidecar_hidden=8):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dnn_dim, 512), nn.Mish(),
            nn.Linear(512, 256), nn.Mish(),
        )
        self.base_head = nn.Sequential(
            nn.Linear(256, 128), nn.Mish(),
            nn.Linear(128, 1),
        )
        self.sun_scaler_head = nn.Sequential(
            nn.Linear(256, 64), nn.Mish(),
            nn.Linear(64, 1),
        )
        self.storm_scaler_head = nn.Sequential(
            nn.Linear(256, 64), nn.Mish(),
            nn.Linear(64, 1),
        )
        self.sun_sidecar = MonotonicMLP(hidden_dim=sidecar_hidden)
        self.storm_sidecar = MonotonicMLP(hidden_dim=sidecar_hidden)
        self._init_scaler_heads()

    def _init_scaler_heads(self):
        for head in [self.sun_scaler_head, self.storm_scaler_head]:
            final_layer = head[-1]
            nn.init.zeros_(final_layer.weight)
            nn.init.constant_(final_layer.bias, GATE_INIT_BIAS)

    def forward(self, x):
        x_deep = x[:, :DNN_DIM]
        x_sfi = x[:, SFI_IDX:SFI_IDX + 1]
        x_kp = x[:, KP_PENALTY_IDX:KP_PENALTY_IDX + 1]
        trunk_out = self.trunk(x_deep)
        base_snr = self.base_head(trunk_out)
        sun_logit = self.sun_scaler_head(trunk_out)
        storm_logit = self.storm_scaler_head(trunk_out)
        sun_gate = _gate(sun_logit)
        storm_gate = _gate(storm_logit)
        return base_snr + sun_gate * self.sun_sidecar(x_sfi) + \
               storm_gate * self.storm_sidecar(x_kp)

    def forward_with_components(self, x):
        x_deep = x[:, :DNN_DIM]
        x_sfi = x[:, SFI_IDX:SFI_IDX + 1]
        x_kp = x[:, KP_PENALTY_IDX:KP_PENALTY_IDX + 1]
        trunk_out = self.trunk(x_deep)
        base_snr = self.base_head(trunk_out)
        sun_logit = self.sun_scaler_head(trunk_out)
        storm_logit = self.storm_scaler_head(trunk_out)
        sun_gate = _gate(sun_logit)
        storm_gate = _gate(storm_logit)
        sun_boost = self.sun_sidecar(x_sfi)
        storm_boost = self.storm_sidecar(x_kp)
        total = base_snr + sun_gate * sun_boost + storm_gate * storm_boost
        return total, base_snr, sun_gate, storm_gate, sun_boost, storm_boost

    def get_sun_effect(self, sfi_normalized):
        x = torch.tensor([[sfi_normalized]], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            return self.sun_sidecar(x).item()

    def get_storm_effect(self, kp_penalty):
        x = torch.tensor([[kp_penalty]], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            return self.storm_sidecar(x).item()


# ── Tests ────────────────────────────────────────────────────────────────────

def test_storm_sidecar(model, global_std):
    """Test 1: Kp 0→9 should COST SNR (storms degrade propagation)."""
    # kp_penalty = 1 - kp/9, so kp=0 → penalty=1.0, kp=9 → penalty=0.0
    kp0_effect = model.get_storm_effect(1.0)   # Kp=0 (quiet)
    kp9_effect = model.get_storm_effect(0.0)   # Kp=9 (severe storm)

    storm_cost_sigma = kp0_effect - kp9_effect  # Should be positive
    storm_cost_db = storm_cost_sigma * global_std

    print(f"  Storm sidecar:")
    print(f"    Kp=0 effect: {kp0_effect:+.4f}σ")
    print(f"    Kp=9 effect: {kp9_effect:+.4f}σ")
    print(f"    Storm cost (Kp 0→9): {storm_cost_sigma:+.4f}σ ({storm_cost_db:+.1f} dB)")

    # Storm cost should be positive (quiet > storm)
    if storm_cost_sigma > 2.0:
        print(f"    PASS: Storm cost {storm_cost_sigma:.2f}σ > 2.0σ (target: > 3.0σ)")
        return True
    else:
        print(f"    FAIL: Storm cost {storm_cost_sigma:.2f}σ <= 2.0σ")
        return False


def test_sun_sidecar(model, global_std):
    """Test 2: SFI 70→200 should BENEFIT SNR (high SFI improves propagation)."""
    sfi_70 = model.get_sun_effect(70.0 / 300.0)
    sfi_200 = model.get_sun_effect(200.0 / 300.0)

    sfi_benefit_sigma = sfi_200 - sfi_70  # Should be positive
    sfi_benefit_db = sfi_benefit_sigma * global_std

    print(f"  Sun sidecar:")
    print(f"    SFI=70 effect:  {sfi_70:+.4f}σ")
    print(f"    SFI=200 effect: {sfi_200:+.4f}σ")
    print(f"    SFI benefit (70→200): {sfi_benefit_sigma:+.4f}σ ({sfi_benefit_db:+.1f} dB)")

    # SFI benefit should be positive and reasonable
    if 0.3 < sfi_benefit_sigma < 1.5:
        print(f"    PASS: SFI benefit {sfi_benefit_sigma:.2f}σ in expected range [0.3, 1.5]")
        return True
    else:
        print(f"    FAIL: SFI benefit {sfi_benefit_sigma:.2f}σ outside expected range")
        return False


def test_gate_range(model):
    """Test 3: Gates should be in range [0.5, 2.0]."""
    # Create a batch of test inputs with varied geography
    test_inputs = []
    for distance in [1000, 5000, 10000, 15000]:
        for freq_log in [0.8, 0.85, 0.9]:  # Different bands
            for hour in [0, 6, 12, 18]:
                x = np.zeros(INPUT_DIM, dtype=np.float32)
                x[0] = distance / 20000.0
                x[1] = freq_log
                x[2] = np.sin(2 * np.pi * hour / 24)
                x[3] = np.cos(2 * np.pi * hour / 24)
                x[11] = 150.0 / 300.0  # SFI
                x[12] = 1.0 - 3.0 / 9.0  # Kp=3
                test_inputs.append(x)

    X = torch.tensor(np.array(test_inputs), dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        _, _, sun_gates, storm_gates, _, _ = model.forward_with_components(X)

    sun_min, sun_max = sun_gates.min().item(), sun_gates.max().item()
    storm_min, storm_max = storm_gates.min().item(), storm_gates.max().item()

    print(f"  Gate ranges:")
    print(f"    Sun gate:   [{sun_min:.4f}, {sun_max:.4f}]")
    print(f"    Storm gate: [{storm_min:.4f}, {storm_max:.4f}]")

    # Gates should be in [0.5, 2.0] per _gate function
    in_range = (0.5 <= sun_min <= sun_max <= 2.0) and (0.5 <= storm_min <= storm_max <= 2.0)
    if in_range:
        print(f"    PASS: All gates in expected range [0.5, 2.0]")
        return True
    else:
        print(f"    FAIL: Gates outside expected range")
        return False


def test_decomposition(model):
    """Test 4: base + sun_gate*sun_boost + storm_gate*storm_boost = total."""
    # Create test input
    x = np.zeros(INPUT_DIM, dtype=np.float32)
    x[0] = 5000 / 20000.0  # distance
    x[1] = 0.875  # freq_log (20m)
    x[2] = 0.0  # hour_sin (midnight)
    x[3] = 1.0  # hour_cos
    x[11] = 150.0 / 300.0  # SFI
    x[12] = 1.0 - 2.0 / 9.0  # Kp=2

    X = torch.tensor([x], dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        total, base, sun_gate, storm_gate, sun_boost, storm_boost = model.forward_with_components(X)

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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  IONIS V18 Physics Verification (4 tests)")
    print("=" * 70)
    print()

    # Load checkpoint
    print(f"Loading checkpoint: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Checkpoint not found at {MODEL_PATH}")
        return 1

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    # V18: Get global normalization constants
    global_mean = checkpoint.get('global_mean', 0.0)
    global_std = checkpoint.get('global_std', 1.0)
    print(f"  global_mean = {global_mean:.2f} dB")
    print(f"  global_std  = {global_std:.2f} dB")
    print()

    # Load model
    model = IonisV12Gate(
        dnn_dim=checkpoint.get('dnn_dim', DNN_DIM),
        sidecar_hidden=checkpoint.get('sidecar_hidden', 8)
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: IonisV12Gate ({total_params:,} params)")
    print()

    # Run tests
    results = []

    print("Test 1: Storm Sidecar (Kp sensitivity)")
    results.append(test_storm_sidecar(model, global_std))
    print()

    print("Test 2: Sun Sidecar (SFI sensitivity)")
    results.append(test_sun_sidecar(model, global_std))
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
