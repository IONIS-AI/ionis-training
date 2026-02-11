#!/usr/bin/env python3
"""
common.py — Shared utilities for IONIS validation templates

Provides:
  - Config loading from version's config.json
  - Model architecture definitions
  - Feature engineering functions
  - Normalization/denormalization helpers

Usage:
  from templates.common import load_config, load_model, engineer_features
"""

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ── Device ───────────────────────────────────────────────────────────────────

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ── Model Constants ──────────────────────────────────────────────────────────

DNN_DIM = 11
SFI_IDX = 11
KP_PENALTY_IDX = 12
INPUT_DIM = 13
GATE_INIT_BIAS = -math.log(2.0)

BAND_TO_HZ = {
    102:  1_836_600,   103:  3_568_600,   104:  5_287_200,
    105:  7_038_600,   106: 10_138_700,   107: 14_097_100,
    108: 18_104_600,   109: 21_094_600,   110: 24_924_600,
    111: 28_124_600,
}

FREQ_MHZ = {
    102: 1.8, 103: 3.5, 104: 5.3, 105: 7.0, 106: 10.1,
    107: 14.0, 108: 18.1, 109: 21.0, 110: 24.9, 111: 28.0,
}

MHZ_TO_BAND = {v: k for k, v in FREQ_MHZ.items()}


# ── Config Loading ───────────────────────────────────────────────────────────

@dataclass
class VersionConfig:
    """Configuration for a specific model version."""
    version: str
    checkpoint: str
    architecture: str
    normalization: str
    norm_keys: Dict[str, str]
    thresholds: Dict[str, float]
    validation: Dict[str, float]
    baselines: Dict[str, float]
    config_path: Path

    @classmethod
    def load(cls, version_dir: Path) -> "VersionConfig":
        """Load config from version directory."""
        config_path = version_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path) as f:
            data = json.load(f)

        return cls(
            version=data["version"],
            checkpoint=data["checkpoint"],
            architecture=data["architecture"],
            normalization=data.get("normalization", "global"),
            norm_keys=data.get("norm_keys", {"mean": "global_mean", "std": "global_std"}),
            thresholds=data.get("thresholds", {}),
            validation=data.get("validation", {}),
            baselines=data.get("baselines", {}),
            config_path=config_path,
        )

    @property
    def checkpoint_path(self) -> Path:
        """Full path to checkpoint file."""
        return self.config_path.parent / self.checkpoint

    def get_threshold(self, mode: str) -> float:
        """Get SNR threshold for a mode."""
        mode_lower = mode.lower()
        if mode_lower in self.thresholds:
            return self.thresholds[mode_lower]
        # Fallback mappings
        fallbacks = {
            "cw": self.thresholds.get("cw_machine", -18.0),
            "ph": self.thresholds.get("ssb", 5.0),
            "ry": self.thresholds.get("rtty", -5.0),
            "dg": self.thresholds.get("ft8", -20.0),
        }
        return fallbacks.get(mode_lower, -20.0)


def load_config(version: str) -> VersionConfig:
    """Load config for a version by name (e.g., 'v18')."""
    templates_dir = Path(__file__).parent
    versions_dir = templates_dir.parent
    version_dir = versions_dir / version
    return VersionConfig.load(version_dir)


# ── Model Architectures ──────────────────────────────────────────────────────

class MonotonicMLP(nn.Module):
    """Monotonic MLP for physics-constrained sidecars."""

    def __init__(self, hidden_dim: int = 8):
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
    """Gate function: maps logits to [0.5, 2.0] range."""
    return 0.5 + 1.5 * torch.sigmoid(x)


class IonisV12Gate(nn.Module):
    """IONIS V12+ architecture with gated dual monotonic sidecars."""

    def __init__(self, dnn_dim: int = 11, sidecar_hidden: int = 8):
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
        """Forward pass returning all components for verification."""
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

    def get_sun_effect(self, sfi_normalized: float) -> float:
        """Get sun sidecar effect for a given normalized SFI."""
        x = torch.tensor([[sfi_normalized]], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            return self.sun_sidecar(x).item()

    def get_storm_effect(self, kp_penalty: float) -> float:
        """Get storm sidecar effect for a given Kp penalty."""
        x = torch.tensor([[kp_penalty]], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            return self.storm_sidecar(x).item()


# Architecture registry
ARCHITECTURES = {
    "IonisV12Gate": IonisV12Gate,
}


def get_architecture(name: str) -> type:
    """Get model class by name."""
    if name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {name}. Available: {list(ARCHITECTURES.keys())}")
    return ARCHITECTURES[name]


# ── Model Loading ────────────────────────────────────────────────────────────

@dataclass
class LoadedModel:
    """Container for loaded model and metadata."""
    model: nn.Module
    config: VersionConfig
    checkpoint: dict
    norm_mean: float
    norm_std: float

    def denormalize(self, predictions_sigma: np.ndarray) -> np.ndarray:
        """Convert predictions from σ to dB."""
        return predictions_sigma * self.norm_std + self.norm_mean

    def predict_db(self, features: torch.Tensor) -> np.ndarray:
        """Run inference and return predictions in dB."""
        with torch.no_grad():
            pred_sigma = self.model(features).cpu().numpy().flatten()
        return self.denormalize(pred_sigma)


def load_model(config: VersionConfig) -> LoadedModel:
    """Load model from config."""
    checkpoint_path = config.checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    # Get architecture
    arch_class = get_architecture(config.architecture)
    model = arch_class(
        dnn_dim=checkpoint.get("dnn_dim", DNN_DIM),
        sidecar_hidden=checkpoint.get("sidecar_hidden", 8),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Get normalization constants
    mean_key = config.norm_keys.get("mean", "global_mean")
    std_key = config.norm_keys.get("std", "global_std")
    norm_mean = checkpoint.get(mean_key, 0.0)
    norm_std = checkpoint.get(std_key, 1.0)

    return LoadedModel(
        model=model,
        config=config,
        checkpoint=checkpoint,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )


# ── Feature Engineering ──────────────────────────────────────────────────────

GRID_RE = re.compile(r"[A-Ra-r]{2}[0-9]{2}")


def grid4_to_latlon(g: str) -> Tuple[float, float]:
    """Convert 4-char Maidenhead grid to (lat, lon) centroid."""
    s = str(g).strip().rstrip("\x00").upper()
    m = GRID_RE.search(s)
    g4 = m.group(0) if m else "JJ00"
    lon = (ord(g4[0]) - ord("A")) * 20.0 - 180.0 + int(g4[2]) * 2.0 + 1.0
    lat = (ord(g4[1]) - ord("A")) * 10.0 - 90.0 + int(g4[3]) * 1.0 + 0.5
    return lat, lon


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float]:
    """Calculate distance (km) and bearing (degrees) between two points."""
    R = 6371.0  # Earth radius in km

    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    distance = R * c

    y = math.sin(dlon) * math.cos(lat2_r)
    x = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
    bearing = (math.degrees(math.atan2(y, x)) + 360) % 360

    return distance, bearing


def build_features(
    lat_tx: float, lon_tx: float,
    lat_rx: float, lon_rx: float,
    freq_mhz: float,
    sfi: float, kp: float,
    hour: float, month: float,
) -> torch.Tensor:
    """Build feature vector for a single path."""
    distance_km, bearing_deg = haversine(lat_tx, lon_tx, lat_rx, lon_rx)

    midpoint_lat = (lat_tx + lat_rx) / 2.0
    midpoint_lon = (lon_tx + lon_rx) / 2.0

    # Get frequency in Hz
    band_id = MHZ_TO_BAND.get(freq_mhz)
    if band_id:
        freq_hz = BAND_TO_HZ[band_id]
    else:
        freq_hz = freq_mhz * 1e6

    kp_penalty = 1.0 - kp / 9.0

    features = np.array([
        distance_km / 20000.0,
        math.log10(freq_hz) / 8.0,
        math.sin(2.0 * math.pi * hour / 24.0),
        math.cos(2.0 * math.pi * hour / 24.0),
        math.sin(2.0 * math.pi * bearing_deg / 360.0),
        math.cos(2.0 * math.pi * bearing_deg / 360.0),
        abs(lat_tx - lat_rx) / 180.0,
        midpoint_lat / 90.0,
        math.sin(2.0 * math.pi * month / 12.0),
        math.cos(2.0 * math.pi * month / 12.0),
        math.cos(2.0 * math.pi * (hour + midpoint_lon / 15.0) / 24.0),
        sfi / 300.0,
        kp_penalty,
    ], dtype=np.float32)

    return torch.tensor([features], dtype=torch.float32, device=DEVICE)


def build_features_batch(
    tx_grids: np.ndarray,
    rx_grids: np.ndarray,
    band_ids: np.ndarray,
    hours: np.ndarray,
    months: np.ndarray,
    sfis: np.ndarray,
    kps: np.ndarray,
    distances: np.ndarray,
    azimuths: np.ndarray,
) -> np.ndarray:
    """Build feature matrix for a batch of paths."""
    n = len(tx_grids)
    features = np.zeros((n, INPUT_DIM), dtype=np.float32)

    for i in range(n):
        tx_lat, tx_lon = grid4_to_latlon(tx_grids[i])
        rx_lat, rx_lon = grid4_to_latlon(rx_grids[i])

        midpoint_lat = (tx_lat + rx_lat) / 2.0
        midpoint_lon = (tx_lon + rx_lon) / 2.0

        band_id = int(band_ids[i])
        freq_hz = BAND_TO_HZ.get(band_id, 14_097_100)

        kp_penalty = 1.0 - kps[i] / 9.0

        features[i] = [
            distances[i] / 20000.0,
            np.log10(freq_hz) / 8.0,
            np.sin(2.0 * np.pi * hours[i] / 24.0),
            np.cos(2.0 * np.pi * hours[i] / 24.0),
            np.sin(2.0 * np.pi * azimuths[i] / 360.0),
            np.cos(2.0 * np.pi * azimuths[i] / 360.0),
            abs(tx_lat - rx_lat) / 180.0,
            midpoint_lat / 90.0,
            np.sin(2.0 * np.pi * months[i] / 12.0),
            np.cos(2.0 * np.pi * months[i] / 12.0),
            np.cos(2.0 * np.pi * (hours[i] + midpoint_lon / 15.0) / 24.0),
            sfis[i] / 300.0,
            kp_penalty,
        ]

    return features
