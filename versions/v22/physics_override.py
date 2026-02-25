"""
physics_override.py — Deterministic Physics Override Layer for IONIS Inference

Post-model inference clamp for physically impossible predictions.
V22-gamma passes 16/17 KI7MT tests. The single failure (acid test: 10m EU
night, +0.540σ) cannot be fixed by retraining — eight architectures tried,
all destroyed global physics. This module applies a deterministic rule at
inference time.

Rule: IF freq >= 21 MHz AND tx_solar < -6° AND rx_solar < -6° AND pred > -1.0σ
      THEN clamp pred to -1.0σ

Physics: No F2 skip above 21 MHz when both terminators are below civil
twilight (-6°). The F2 MUF has dropped below 21 MHz. Sporadic E could
theoretically open 15m briefly, but Es is localized and unpredictable —
not something a statistical model should predict as "open."

The -1.0σ threshold (~-24.5 dB WSPR) is 3.5 dB above the WSPR decode floor
(-28 dB). It predicts "below reliable threshold," not "impossible."

Usage:
    from physics_override import PhysicsOverrideLayer, apply_override_to_prediction

    # Single prediction
    sigma, was_overridden = apply_override_to_prediction(
        sigma, freq_mhz, tx_solar_deg, rx_solar_deg)

    # Batch (numpy arrays)
    override = PhysicsOverrideLayer()
    sigmas, audit = override(sigmas, freq_mhz, tx_solar, rx_solar)
"""

import numpy as np


# ── Constants ────────────────────────────────────────────────────────────────

FREQ_THRESHOLD_MHZ = 21.0      # 15m and above (21, 24, 28 MHz)
SOLAR_THRESHOLD_DEG = -6.0     # Civil twilight
CLAMP_SIGMA = -1.0             # ~-24.5 dB WSPR, 3.5 dB above decode floor


# ── Single Prediction ────────────────────────────────────────────────────────

def apply_override_to_prediction(sigma, freq_mhz, tx_solar_deg, rx_solar_deg):
    """Apply physics override to a single prediction.

    Args:
        sigma: Model prediction in z-score units
        freq_mhz: Operating frequency in MHz
        tx_solar_deg: TX solar elevation in degrees (negative = below horizon)
        rx_solar_deg: RX solar elevation in degrees

    Returns:
        (clamped_sigma, was_overridden) tuple
    """
    if (freq_mhz >= FREQ_THRESHOLD_MHZ and
            tx_solar_deg < SOLAR_THRESHOLD_DEG and
            rx_solar_deg < SOLAR_THRESHOLD_DEG and
            sigma > CLAMP_SIGMA):
        return CLAMP_SIGMA, True
    return sigma, False


# ── Batch Override ────────────────────────────────────────────────────────────

class PhysicsOverrideLayer:
    """Deterministic physics clamp applied after model inference.

    Not a torch.nn.Module — this is pure numpy, no gradients, no GPU.
    Applied between inference and output/denormalization.
    """

    def __init__(self, freq_threshold=FREQ_THRESHOLD_MHZ,
                 solar_threshold=SOLAR_THRESHOLD_DEG,
                 clamp_sigma=CLAMP_SIGMA):
        self.freq_threshold = freq_threshold
        self.solar_threshold = solar_threshold
        self.clamp_sigma = clamp_sigma

    def __call__(self, sigmas, freq_mhz, tx_solar_deg, rx_solar_deg):
        """Apply physics override to a batch of predictions.

        Args:
            sigmas: np.ndarray of shape (N,) — model predictions in z-score
            freq_mhz: np.ndarray of shape (N,) — operating frequency in MHz
            tx_solar_deg: np.ndarray of shape (N,) — TX solar elevation
            rx_solar_deg: np.ndarray of shape (N,) — RX solar elevation

        Returns:
            (clamped_sigmas, audit_dict) tuple
            audit_dict contains: n_overridden, override_mask, original_sigmas
        """
        sigmas = np.asarray(sigmas, dtype=np.float64)
        original = sigmas.copy()

        mask = ((freq_mhz >= self.freq_threshold) &
                (tx_solar_deg < self.solar_threshold) &
                (rx_solar_deg < self.solar_threshold) &
                (sigmas > self.clamp_sigma))

        sigmas[mask] = self.clamp_sigma

        audit = {
            "n_total": len(sigmas),
            "n_overridden": int(mask.sum()),
            "override_mask": mask,
            "original_sigmas": original,
        }
        return sigmas, audit

    def describe(self):
        """Human-readable description of the override rule."""
        return (
            f"PhysicsOverrideLayer: "
            f"IF freq >= {self.freq_threshold} MHz "
            f"AND tx_solar < {self.solar_threshold}° "
            f"AND rx_solar < {self.solar_threshold}° "
            f"AND pred > {self.clamp_sigma}σ "
            f"THEN clamp to {self.clamp_sigma}σ"
        )
