"""
model.py — Standalone IONIS model definition (inference-only)

This module contains everything needed to load and run IonisGate
for inference. Zero ClickHouse dependency.

Extracted from train_common.py for beta validation packaging.

Classes:
    MonotonicMLP — Monotonically increasing MLP for physics constraints
    IonisGate — Production model architecture

Functions:
    get_device() — Universal device selection (CUDA > MPS > CPU)
    load_model() — Load checkpoint and return (model, metadata)

Security:
    Uses safetensors format — no pickle, no arbitrary code execution.
"""

import json
import math
import os
import re

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors


# ── Device Selection ─────────────────────────────────────────────────────────

def get_device():
    """Select the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Grid Utilities ───────────────────────────────────────────────────────────

GRID_RE = re.compile(r'[A-Ra-r]{2}[0-9]{2}')


def grid4_to_latlon(g):
    """Convert a single 4-char Maidenhead grid to (lat, lon) centroid."""
    s = str(g).strip().rstrip('\x00').upper()
    m = GRID_RE.search(s)
    g4 = m.group(0) if m else 'JJ00'
    lon = (ord(g4[0]) - ord('A')) * 20.0 - 180.0 + int(g4[2]) * 2.0 + 1.0
    lat = (ord(g4[1]) - ord('A')) * 10.0 - 90.0 + int(g4[3]) * 1.0 + 0.5
    return lat, lon


# ── Geo Helpers ──────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def azimuth_deg(lat1, lon1, lat2, lon2):
    """Initial bearing (azimuth) in degrees from point 1 to point 2."""
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2_r)
    y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


# ── Band Lookup ──────────────────────────────────────────────────────────────

# WSPR dial frequencies in Hz, keyed by common name
BAND_FREQ_HZ = {
    "160m": 1_836_600,
    "80m":  3_568_600,
    "60m":  5_287_200,
    "40m":  7_038_600,
    "30m": 10_138_700,
    "20m": 14_097_100,
    "17m": 18_104_600,
    "15m": 21_094_600,
    "12m": 24_924_600,
    "10m": 28_124_600,
}


# ── Feature Builder ──────────────────────────────────────────────────────────

def build_features(tx_lat, tx_lon, rx_lat, rx_lon, freq_hz, sfi, kp,
                   hour_utc, month):
    """Build a 13-element normalized feature vector for a single path.

    Returns:
        np.ndarray of shape (13,), dtype float32
    """
    distance_km = haversine_km(tx_lat, tx_lon, rx_lat, rx_lon)
    az = azimuth_deg(tx_lat, tx_lon, rx_lat, rx_lon)
    midpoint_lat = (tx_lat + rx_lat) / 2.0
    midpoint_lon = (tx_lon + rx_lon) / 2.0
    local_solar_h = hour_utc + midpoint_lon / 15.0

    return np.array([
        distance_km / 20000.0,                              # 0: distance
        np.log10(freq_hz) / 8.0,                            # 1: freq_log
        np.sin(2.0 * np.pi * hour_utc / 24.0),              # 2: hour_sin
        np.cos(2.0 * np.pi * hour_utc / 24.0),              # 3: hour_cos
        np.sin(2.0 * np.pi * az / 360.0),                   # 4: az_sin
        np.cos(2.0 * np.pi * az / 360.0),                   # 5: az_cos
        abs(tx_lat - rx_lat) / 180.0,                       # 6: lat_diff
        midpoint_lat / 90.0,                                 # 7: midpoint_lat
        np.sin(2.0 * np.pi * month / 12.0),                 # 8: season_sin
        np.cos(2.0 * np.pi * month / 12.0),                 # 9: season_cos
        np.cos(2.0 * np.pi * local_solar_h / 24.0),         # 10: day_night_est
        sfi / 300.0,                                         # 11: sfi
        1.0 - kp / 9.0,                                     # 12: kp_penalty
    ], dtype=np.float32)


# ── Model Architecture ───────────────────────────────────────────────────────

class MonotonicMLP(nn.Module):
    """Monotonically increasing MLP for physics constraints."""

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
    """Gate function: range 0.5 to 2.0"""
    return 0.5 + 1.5 * torch.sigmoid(x)


class IonisGate(nn.Module):
    """
    IONIS Production Model — gated sidecars for physics-constrained SNR prediction.

    Architecture:
        - Trunk: geography/time features (11-dim) → 256-dim representation
        - Gates from trunk output (256-dim), not raw input
        - Gate range 0.5-2.0
        - Separate base_head (256→128→1) and scaler_heads (256→64→1)
        - Uses Mish activation, no LayerNorm or Dropout
        - Requires defibrillator init and weight clamping to keep sidecars alive

    Args:
        dnn_dim: Number of geography/time features (default 11)
        sidecar_hidden: Hidden units in MonotonicMLP (default 8)
        sfi_idx: Index of SFI feature in input (default 11)
        kp_penalty_idx: Index of Kp penalty feature in input (default 12)
        gate_init_bias: Initial bias for scaler heads (default -ln(2))
    """

    def __init__(self, dnn_dim=11, sidecar_hidden=8, sfi_idx=11, kp_penalty_idx=12,
                 gate_init_bias=None):
        super().__init__()

        if gate_init_bias is None:
            gate_init_bias = -math.log(2.0)

        self.dnn_dim = dnn_dim
        self.sfi_idx = sfi_idx
        self.kp_penalty_idx = kp_penalty_idx

        # Trunk: geography/time features → 256-dim representation
        self.trunk = nn.Sequential(
            nn.Linear(dnn_dim, 512), nn.Mish(),
            nn.Linear(512, 256), nn.Mish(),
        )

        # Base head: trunk → SNR prediction
        self.base_head = nn.Sequential(
            nn.Linear(256, 128), nn.Mish(),
            nn.Linear(128, 1),
        )

        # Scaler heads: trunk → gate logits (256-dim input, expressive)
        self.sun_scaler_head = nn.Sequential(
            nn.Linear(256, 64), nn.Mish(),
            nn.Linear(64, 1),
        )
        self.storm_scaler_head = nn.Sequential(
            nn.Linear(256, 64), nn.Mish(),
            nn.Linear(64, 1),
        )

        # Physics sidecars (monotonic)
        self.sun_sidecar = MonotonicMLP(hidden_dim=sidecar_hidden)
        self.storm_sidecar = MonotonicMLP(hidden_dim=sidecar_hidden)

        # Initialize scaler heads
        self._init_scaler_heads(gate_init_bias)

    def _init_scaler_heads(self, gate_init_bias):
        """Initialize scaler head biases for balanced gates."""
        for head in [self.sun_scaler_head, self.storm_scaler_head]:
            final_layer = head[-1]
            nn.init.zeros_(final_layer.weight)
            nn.init.constant_(final_layer.bias, gate_init_bias)

    def forward(self, x):
        x_deep = x[:, :self.dnn_dim]
        x_sfi = x[:, self.sfi_idx:self.sfi_idx + 1]
        x_kp = x[:, self.kp_penalty_idx:self.kp_penalty_idx + 1]

        trunk_out = self.trunk(x_deep)
        base_snr = self.base_head(trunk_out)

        sun_logit = self.sun_scaler_head(trunk_out)
        storm_logit = self.storm_scaler_head(trunk_out)
        sun_gate = _gate(sun_logit)
        storm_gate = _gate(storm_logit)

        return base_snr + sun_gate * self.sun_sidecar(x_sfi) + \
               storm_gate * self.storm_sidecar(x_kp)

    def forward_with_gates(self, x):
        """Forward pass returning gate values for variance loss."""
        x_deep = x[:, :self.dnn_dim]
        x_sfi = x[:, self.sfi_idx:self.sfi_idx + 1]
        x_kp = x[:, self.kp_penalty_idx:self.kp_penalty_idx + 1]

        trunk_out = self.trunk(x_deep)
        base_snr = self.base_head(trunk_out)

        sun_logit = self.sun_scaler_head(trunk_out)
        storm_logit = self.storm_scaler_head(trunk_out)
        sun_gate = _gate(sun_logit)
        storm_gate = _gate(storm_logit)

        sun_boost = self.sun_sidecar(x_sfi)
        storm_boost = self.storm_sidecar(x_kp)

        return base_snr + sun_gate * sun_boost + storm_gate * storm_boost, \
               sun_gate, storm_gate

    def get_sun_effect(self, sfi_normalized, device):
        """Get raw sun sidecar output for a given SFI value."""
        with torch.no_grad():
            x = torch.tensor([[sfi_normalized]], dtype=torch.float32, device=device)
            return self.sun_sidecar(x).item()

    def get_storm_effect(self, kp_penalty, device):
        """Get raw storm sidecar output for a given Kp penalty value."""
        with torch.no_grad():
            x = torch.tensor([[kp_penalty]], dtype=torch.float32, device=device)
            return self.storm_sidecar(x).item()

    def get_gates(self, x):
        """Get gate values without gradient tracking."""
        x_deep = x[:, :self.dnn_dim]
        with torch.no_grad():
            trunk_out = self.trunk(x_deep)
            sun_logit = self.sun_scaler_head(trunk_out)
            storm_logit = self.storm_scaler_head(trunk_out)
        return _gate(sun_logit), _gate(storm_logit)


# ── Model Loading ────────────────────────────────────────────────────────────

def load_model(config_path=None, checkpoint_path=None, device=None):
    """Load an IONIS model from config + safetensors checkpoint.

    Args:
        config_path: Path to config JSON. If None, auto-discovers V20 config.
        checkpoint_path: Path to .safetensors file. If None, derived from config.
        device: torch.device. If None, auto-selects via get_device().

    Returns:
        (model, config, metadata) tuple.
        model is in eval mode on the specified device.
        metadata dict contains training info (val_rmse, val_pearson, etc).

    Security:
        Uses safetensors format — pure tensor data, no pickle, no code execution.
    """
    if device is None:
        device = get_device()

    # Auto-discover V20 if no paths given
    if config_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(here, "..", "v20", "config_v20.json")

    with open(config_path) as f:
        config = json.load(f)

    config_dir = os.path.dirname(os.path.abspath(config_path))

    if checkpoint_path is None:
        checkpoint_path = os.path.join(config_dir, config["checkpoint"])

    # Load weights via safetensors (no pickle)
    state_dict = load_safetensors(checkpoint_path, device=str(device))

    # Load metadata from companion JSON
    meta_path = checkpoint_path.replace(".safetensors", "_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    model = IonisGate(
        dnn_dim=config["model"]["dnn_dim"],
        sidecar_hidden=config["model"]["sidecar_hidden"],
        sfi_idx=config["model"]["sfi_idx"],
        kp_penalty_idx=config["model"]["kp_penalty_idx"],
        gate_init_bias=config["model"].get("gate_init_bias"),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, config, metadata
