#!/usr/bin/env python3
"""
oracle_v19.py — IONIS V19 Oracle (Config-driven)

This is a thin wrapper that:
1. Loads configuration from config_v19_3.json
2. Imports IonisModel from versions/common/train_common.py
3. Provides the Rosetta Stone decoder for mode-appropriate SNR predictions

The V19 oracle implements the "Rosetta Stone" decoder:
    - Model outputs Z-score (sigma) = signal quality relative to average
    - Denormalize using source-appropriate constants based on mode

Mode mapping (from config.json):
    - WSPR/FT8/FT4/JT65/JT9/JS8 -> use WSPR constants (weak signal)
    - CW_machine/RTTY -> use RBN constants (skimmer)
    - CW_human/SSB -> use Contest constants (human ear)

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

# Add parent directory to path for common imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VERSIONS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, VERSIONS_DIR)

from common.train_common import (
    IonisModel,
    grid4_to_latlon,
)

# ── Configuration ─────────────────────────────────────────────────────────────

CONFIG_FILE = os.path.join(SCRIPT_DIR, "config_v19_4.json")
MODEL_FILE = os.path.join(SCRIPT_DIR, "ionis_v19_4.pth")

with open(CONFIG_FILE) as f:
    CONFIG = json.load(f)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Extract model dimensions from config
DNN_DIM = CONFIG["model"]["dnn_dim"]
INPUT_DIM = CONFIG["model"]["input_dim"]
SFI_IDX = CONFIG["model"]["sfi_idx"]
KP_PENALTY_IDX = CONFIG["model"]["kp_penalty_idx"]

# Convert band_to_hz keys to int (JSON keys are strings)
BAND_TO_HZ = {int(k): v for k, v in CONFIG["band_to_hz"].items()}

BAND_NAME_TO_ID = {
    '160m': 102, '80m': 103, '60m': 104, '40m': 105, '30m': 106,
    '20m': 107, '17m': 108, '15m': 109, '12m': 110, '10m': 111,
}

GRID_RE = re.compile(r'[A-Ra-r]{2}[0-9]{2}')


# ── Utilities ─────────────────────────────────────────────────────────────────

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


# ── Oracle ────────────────────────────────────────────────────────────────────

class IonisOracle:
    """
    IONIS V19 Oracle with Rosetta Stone decoder.

    The model outputs Z-score (sigma) representing signal quality.
    The Rosetta Stone maps modes to their appropriate normalization constants
    to produce physically meaningful dB predictions.
    """

    def __init__(self, model_path=None, config=None):
        self.model_path = model_path or MODEL_FILE
        self.config = config or CONFIG

        self.norm_constants = self.config['norm_constants']
        self.mode_map = self.config['mode_map']
        self.thresholds = self.config['thresholds']

        # Load model
        checkpoint = torch.load(self.model_path, map_location=DEVICE, weights_only=False)

        self.model = IonisModel(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(DEVICE)
        self.model.eval()

        # Override config with checkpoint constants if available
        if 'norm_constants' in checkpoint:
            self.norm_constants = checkpoint['norm_constants']

        self.version = self.config.get('version', 'v19')

    def _compute_features(self, tx_lat, tx_lon, rx_lat, rx_lon, band_id, hour, month, sfi, kp):
        """Compute input features for the model."""
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
        features[SFI_IDX] = sfi / 300.0
        features[KP_PENALTY_IDX] = 1.0 - kp / 9.0

        return features

    def _rosetta_decode(self, sigma, mode):
        """
        Rosetta Stone decoder: Map Z-score to dB using mode-appropriate constants.

        A +1.0 sigma prediction means "this path is 1 std better than average."
        - For WSPR: +1 sigma -> -11 dB (great weak-signal path)
        - For RBN:  +1 sigma -> +26 dB (booming CW signal)
        - For SSB:  +1 sigma -> +12 dB (solid voice copy)
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
        snr_db, source = self._rosetta_decode(sigma, mode)

        # Get threshold
        mode_lower = mode.lower().replace('-', '_')
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

        Returns dict mapping mode -> {snr_db, threshold, band_open}
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


# ── CLI ───────────────────────────────────────────────────────────────────────

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
    print(f"Path: {args.tx_grid} -> {args.rx_grid} on {args.band}")
    print(f"Conditions: SFI={args.sfi}, Kp={args.kp}, {args.hour:02d}Z, Month {args.month}")
    print()

    if args.all_modes:
        result = oracle.predict_all_modes(
            args.tx_grid, args.rx_grid, args.band,
            args.hour, args.month, args.sfi, args.kp
        )
        print(f"Signal Quality (sigma): {result['sigma']:+.3f}")
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
        print(f"Signal Quality (sigma): {result['sigma']:+.3f}")
        print(f"Predicted SNR: {result['snr_db']:+.1f} dB (using {result['source']} scale)")
        print(f"Threshold: {result['threshold']:+.1f} dB")
        print(f"Band Open: {'YES' if result['band_open'] else 'NO'}")


if __name__ == '__main__':
    main()
