#!/usr/bin/env python3
"""
oracle_v19.py — IONIS V19 Rosetta Stone Oracle

The V19 oracle implements the "Rosetta Stone" decoder:
    - Model outputs Z-score (σ) = signal quality relative to average
    - Denormalize using source-appropriate constants based on mode

Mode mapping (from config.json):
    - WSPR/FT8/FT4/JT65/JT9/JS8 → use WSPR constants (weak signal)
    - CW_machine/RTTY → use RBN constants (skimmer)
    - CW_human/SSB → use Contest constants (human ear)

Usage:
    from oracle_v19 import IonisOracle
    oracle = IonisOracle()
    result = oracle.predict(tx_grid, rx_grid, band, hour, month, sfi, kp, mode='ft8')
    print(f"SNR: {result['snr_db']:.1f} dB, Band open: {result['band_open']}")

CLI usage:
    python oracle_v19.py --tx-grid FN20 --rx-grid JN48 --band 20m --mode ft8
"""

import argparse
import json
import math
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "ionis_v19.pth")
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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

BAND_NAME_TO_ID = {
    '160m': 102, '80m': 103, '60m': 104, '40m': 105, '30m': 106,
    '20m': 107, '17m': 108, '15m': 109, '12m': 110, '10m': 111,
}

GRID_RE = re.compile(r'[A-Ra-r]{2}[0-9]{2}')


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


class IonisV12Gate(nn.Module):
    def __init__(self, dnn_dim=DNN_DIM, hidden=256, sidecar_hidden=8):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(dnn_dim, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

        self.sun_sidecar = MonotonicMLP(sidecar_hidden)
        self.storm_sidecar = MonotonicMLP(sidecar_hidden)

        self.sun_gate = nn.Sequential(
            nn.Linear(dnn_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.storm_gate = nn.Sequential(
            nn.Linear(dnn_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x_dnn = x[:, :DNN_DIM]
        sfi_in = x[:, SFI_IDX:SFI_IDX+1]
        kp_penalty = x[:, KP_PENALTY_IDX:KP_PENALTY_IDX+1]

        base = self.trunk(x_dnn)
        sun_effect = self.sun_sidecar(sfi_in)
        storm_effect = self.storm_sidecar(kp_penalty)

        sun_g = torch.sigmoid(self.sun_gate(x_dnn)) + 0.5
        storm_g = torch.sigmoid(self.storm_gate(x_dnn)) + 0.5

        return base + sun_g * sun_effect + storm_g * storm_effect

    def get_sun_effect(self, sfi_normalized):
        with torch.no_grad():
            inp = torch.tensor([[sfi_normalized]], dtype=torch.float32, device=DEVICE)
            return self.sun_sidecar(inp).item()

    def get_storm_effect(self, kp_penalty):
        with torch.no_grad():
            inp = torch.tensor([[kp_penalty]], dtype=torch.float32, device=DEVICE)
            return self.storm_sidecar(inp).item()


# ── Utilities ────────────────────────────────────────────────────────────────

def grid4_to_latlon(g):
    """Convert 4-char Maidenhead grid to (lat, lon) centroid."""
    s = str(g).strip().rstrip('\x00').upper()
    m = GRID_RE.search(s)
    g4 = m.group(0) if m else 'JJ00'
    lon = (ord(g4[0]) - ord('A')) * 20.0 - 180.0 + int(g4[2]) * 2.0 + 1.0
    lat = (ord(g4[1]) - ord('A')) * 10.0 - 90.0 + int(g4[3]) * 1.0 + 0.5
    return lat, lon


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in km."""
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def compute_azimuth(lat1, lon1, lat2, lon2):
    """Calculate initial bearing (azimuth) from point 1 to point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


# ── Oracle ───────────────────────────────────────────────────────────────────

class IonisOracle:
    """
    IONIS V19 Oracle with Rosetta Stone decoder.

    The model outputs Z-score (σ) representing signal quality.
    The Rosetta Stone maps modes to their appropriate normalization constants
    to produce physically meaningful dB predictions.
    """

    def __init__(self, model_path=None, config_path=None):
        self.model_path = model_path or MODEL_PATH
        self.config_path = config_path or CONFIG_PATH

        # Load config
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

        self.norm_constants = self.config['norm_constants']
        self.mode_map = self.config['mode_map']
        self.thresholds = self.config['thresholds']

        # Load model
        checkpoint = torch.load(self.model_path, map_location=DEVICE, weights_only=False)

        self.model = IonisV12Gate(dnn_dim=DNN_DIM, hidden=256, sidecar_hidden=8)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(DEVICE)
        self.model.eval()

        # Override config with checkpoint constants if available
        if 'norm_constants' in checkpoint:
            self.norm_constants = checkpoint['norm_constants']

        self.version = checkpoint.get('version', 'v19')

    def _compute_features(self, tx_lat, tx_lon, rx_lat, rx_lon, band_id, hour, month, sfi, kp):
        """Compute 13 input features for the model."""
        freq_hz = BAND_TO_HZ.get(band_id, 14_097_100)
        distance = haversine_km(tx_lat, tx_lon, rx_lat, rx_lon)
        azimuth = compute_azimuth(tx_lat, tx_lon, rx_lat, rx_lon)
        midpoint_lat = (tx_lat + rx_lat) / 2.0
        midpoint_lon = (tx_lon + rx_lon) / 2.0

        features = np.zeros(INPUT_DIM, dtype=np.float32)
        features[0] = distance / 20000.0
        features[1] = np.log10(freq_hz) / 8.0
        features[2] = np.sin(2 * np.pi * hour / 24.0)
        features[3] = np.cos(2 * np.pi * hour / 24.0)
        features[4] = np.sin(2 * np.pi * azimuth / 360.0)
        features[5] = np.cos(2 * np.pi * azimuth / 360.0)
        features[6] = abs(tx_lat - rx_lat) / 180.0
        features[7] = midpoint_lat / 90.0
        features[8] = np.sin(2 * np.pi * month / 12.0)
        features[9] = np.cos(2 * np.pi * month / 12.0)
        features[10] = np.cos(2 * np.pi * (hour + midpoint_lon / 15.0) / 24.0)
        features[11] = sfi / 300.0
        features[12] = 1.0 - kp / 9.0

        return features

    def _rosetta_decode(self, sigma, mode):
        """
        Rosetta Stone decoder: Map Z-score to dB using mode-appropriate constants.

        A +1.0σ prediction means "this path is 1 std better than average."
        - For WSPR: +1σ → -11 dB (great weak-signal path)
        - For RBN:  +1σ → +26 dB (booming CW signal)
        - For SSB:  +1σ → +12 dB (solid voice copy)
        """
        mode_lower = mode.lower().replace('-', '_')

        # Map mode to source
        source = self.mode_map.get(mode_lower, 'wspr')
        constants = self.norm_constants[source]

        snr_db = sigma * constants['std'] + constants['mean']
        return snr_db, source

    def predict(self, tx_grid, rx_grid, band, hour, month, sfi, kp, mode='ft8'):
        """
        Predict propagation for a path.

        Args:
            tx_grid: Transmitter grid (e.g., 'FN20')
            rx_grid: Receiver grid (e.g., 'JN48')
            band: Band ID (102-111) or name ('20m')
            hour: UTC hour (0-23)
            month: Month (1-12)
            sfi: Solar Flux Index (70-300)
            kp: Kp index (0-9)
            mode: Mode for threshold selection ('ft8', 'cw_machine', 'ssb', etc.)

        Returns:
            dict with:
                - sigma: Raw Z-score prediction
                - snr_db: Predicted SNR in dB (Rosetta decoded)
                - source: Which constants used ('wspr', 'rbn', 'contest')
                - threshold: Band-open threshold for this mode
                - band_open: True if snr_db >= threshold
                - modes: Dict of SNR in all receiver scales
        """
        # Parse inputs
        tx_lat, tx_lon = grid4_to_latlon(tx_grid)
        rx_lat, rx_lon = grid4_to_latlon(rx_grid)

        if isinstance(band, str):
            band_id = BAND_NAME_TO_ID.get(band.lower(), 107)
        else:
            band_id = band

        # Compute features
        features = self._compute_features(
            tx_lat, tx_lon, rx_lat, rx_lon,
            band_id, hour, month, sfi, kp
        )

        # Run model
        X = torch.tensor(np.array([features]), dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            sigma = self.model(X).cpu().item()

        # Rosetta decode for requested mode
        mode_lower = mode.lower().replace('-', '_')
        snr_db, source = self._rosetta_decode(sigma, mode)

        # Get threshold
        threshold = self.thresholds.get(mode_lower, -15.0)
        band_open = snr_db >= threshold

        # Compute SNR in all scales for reference
        modes_snr = {}
        for mode_name, src in self.mode_map.items():
            c = self.norm_constants[src]
            modes_snr[mode_name] = sigma * c['std'] + c['mean']

        return {
            'sigma': sigma,
            'snr_db': snr_db,
            'source': source,
            'threshold': threshold,
            'band_open': band_open,
            'modes': modes_snr,
        }

    def predict_all_modes(self, tx_grid, rx_grid, band, hour, month, sfi, kp):
        """
        Predict propagation and return band-open status for all modes.

        Returns dict mapping mode → {snr_db, threshold, band_open}
        """
        # Get base prediction
        result = self.predict(tx_grid, rx_grid, band, hour, month, sfi, kp, mode='ft8')
        sigma = result['sigma']

        # Compute for all modes
        all_modes = {}
        for mode_name, src in self.mode_map.items():
            c = self.norm_constants[src]
            snr_db = sigma * c['std'] + c['mean']
            threshold = self.thresholds.get(mode_name, -15.0)
            all_modes[mode_name] = {
                'snr_db': snr_db,
                'threshold': threshold,
                'band_open': snr_db >= threshold,
                'source': src,
            }

        return {
            'sigma': sigma,
            'modes': all_modes,
        }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='IONIS V19 Oracle (Rosetta Stone)')
    parser.add_argument('--tx-grid', required=True, help='Transmitter grid (e.g., FN20)')
    parser.add_argument('--rx-grid', required=True, help='Receiver grid (e.g., JN48)')
    parser.add_argument('--band', required=True, help='Band (20m, 40m, etc. or ID 102-111)')
    parser.add_argument('--hour', type=int, default=12, help='UTC hour (0-23)')
    parser.add_argument('--month', type=int, default=2, help='Month (1-12)')
    parser.add_argument('--sfi', type=float, default=150.0, help='Solar Flux Index (70-300)')
    parser.add_argument('--kp', type=float, default=2.0, help='Kp index (0-9)')
    parser.add_argument('--mode', default='ft8', help='Mode (ft8, wspr, cw_machine, ssb, etc.)')
    parser.add_argument('--all-modes', action='store_true', help='Show all modes')

    args = parser.parse_args()

    oracle = IonisOracle()

    print(f"IONIS {oracle.version} Oracle (Rosetta Stone)")
    print(f"Path: {args.tx_grid} → {args.rx_grid} on {args.band}")
    print(f"Conditions: SFI={args.sfi}, Kp={args.kp}, {args.hour:02d}Z, Month {args.month}")
    print()

    if args.all_modes:
        result = oracle.predict_all_modes(
            args.tx_grid, args.rx_grid, args.band,
            args.hour, args.month, args.sfi, args.kp
        )
        print(f"Signal Quality (σ): {result['sigma']:+.3f}")
        print()
        print("Mode            SNR (dB)   Threshold   Band Open   Source")
        print("-" * 60)
        for mode, data in sorted(result['modes'].items()):
            status = "YES" if data['band_open'] else "NO"
            print(f"{mode:15s} {data['snr_db']:+8.1f}   {data['threshold']:+8.1f}   {status:^9s}   {data['source']}")
    else:
        result = oracle.predict(
            args.tx_grid, args.rx_grid, args.band,
            args.hour, args.month, args.sfi, args.kp, args.mode
        )
        print(f"Mode: {args.mode.upper()}")
        print(f"Signal Quality (σ): {result['sigma']:+.3f}")
        print(f"Predicted SNR: {result['snr_db']:+.1f} dB (using {result['source']} scale)")
        print(f"Threshold: {result['threshold']:+.1f} dB")
        print(f"Band Open: {'YES' if result['band_open'] else 'NO'}")


if __name__ == '__main__':
    main()
