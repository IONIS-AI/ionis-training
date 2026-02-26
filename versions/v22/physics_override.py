"""
physics_override.py — Deterministic Physics Override Layer for IONIS Inference

Post-model inference clamp for physically impossible predictions.
V22-gamma passes 16/17 KI7MT tests. The single failure (acid test: 10m EU
night, +0.540σ) cannot be fixed by retraining — eight architectures tried,
all destroyed global physics. This module applies deterministic rules at
inference time.

Rule A (both dark):
    IF freq >= 21 MHz AND tx_solar < -6° AND rx_solar < -6° AND pred > -2.0σ
    THEN clamp pred to -2.0σ

Rule B (TX deep darkness):
    IF freq >= 21 MHz AND tx_solar < -18° AND pred > -2.0σ
    THEN clamp pred to -2.0σ

Physics: No F2 skip above 21 MHz when the ionosphere above the transmitter
has fully decayed. Rule A handles mutual darkness (civil twilight -6°).
Rule B handles asymmetric paths where the TX is past astronomical twilight
(-18°) — the signal cannot get UP into the ionosphere for the first hop.
RX-dark with TX in daylight is NOT clamped: multi-hop paths carry the
signal across sunlit ionosphere to arrive at the dark receiver. Proven by
KI7MT QSO data (301 QSOs 15m JA at 17z, 568 QSOs 15m EU at 18z Nov).

The -2.0σ clamp = -30.9 dB, well below the WSPR decode floor (-28 dB).
Display shows "—" (closed) with clear margin.

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
SOLAR_THRESHOLD_DEG = -6.0     # Civil twilight (Rule A: both endpoints)
DEEP_DARK_THRESHOLD_DEG = -18.0  # Astronomical twilight (Rule B: single endpoint)
CLAMP_SIGMA = -2.0             # -30.9 dB, below WSPR -28 dB decode floor


# ── Single Prediction ────────────────────────────────────────────────────────

def apply_override_to_prediction(sigma, freq_mhz, tx_solar_deg, rx_solar_deg):
    """Apply physics override to a single prediction.

    Two rules (either triggers the clamp):
        Rule A: freq >= 21 MHz AND tx_solar < -6° AND rx_solar < -6°
        Rule B: freq >= 21 MHz AND tx_solar < -18°

    Args:
        sigma: Model prediction in z-score units
        freq_mhz: Operating frequency in MHz
        tx_solar_deg: TX solar elevation in degrees (negative = below horizon)
        rx_solar_deg: RX solar elevation in degrees

    Returns:
        (clamped_sigma, was_overridden) tuple
    """
    if freq_mhz >= FREQ_THRESHOLD_MHZ and sigma > CLAMP_SIGMA:
        # Rule A: both endpoints in civil twilight or deeper
        both_dark = (tx_solar_deg < SOLAR_THRESHOLD_DEG and
                     rx_solar_deg < SOLAR_THRESHOLD_DEG)
        # Rule B: TX past astronomical twilight (signal can't get UP)
        tx_deep_dark = (tx_solar_deg < DEEP_DARK_THRESHOLD_DEG)
        if both_dark or tx_deep_dark:
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
                 deep_dark_threshold=DEEP_DARK_THRESHOLD_DEG,
                 clamp_sigma=CLAMP_SIGMA):
        self.freq_threshold = freq_threshold
        self.solar_threshold = solar_threshold
        self.deep_dark_threshold = deep_dark_threshold
        self.clamp_sigma = clamp_sigma

    def __call__(self, sigmas, freq_mhz, tx_solar_deg, rx_solar_deg):
        """Apply physics override to a batch of predictions.

        Two rules (either triggers the clamp):
            Rule A: freq >= 21 MHz AND tx_solar < -6° AND rx_solar < -6°
            Rule B: freq >= 21 MHz AND tx_solar < -18°

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

        freq_ok = (freq_mhz >= self.freq_threshold)
        above_clamp = (sigmas > self.clamp_sigma)

        # Rule A: both endpoints in civil twilight or deeper
        both_dark = ((tx_solar_deg < self.solar_threshold) &
                     (rx_solar_deg < self.solar_threshold))

        # Rule B: TX past astronomical twilight (signal can't get UP)
        tx_deep_dark = (tx_solar_deg < self.deep_dark_threshold)

        mask = freq_ok & above_clamp & (both_dark | tx_deep_dark)

        sigmas[mask] = self.clamp_sigma

        audit = {
            "n_total": len(sigmas),
            "n_overridden": int(mask.sum()),
            "override_mask": mask,
            "original_sigmas": original,
        }
        return sigmas, audit

    def describe(self):
        """Human-readable description of the override rules."""
        return (
            f"PhysicsOverrideLayer: "
            f"Rule A: freq >= {self.freq_threshold} MHz "
            f"AND tx_solar < {self.solar_threshold}° "
            f"AND rx_solar < {self.solar_threshold}° → clamp {self.clamp_sigma}σ | "
            f"Rule B: freq >= {self.freq_threshold} MHz "
            f"AND tx_solar < {self.deep_dark_threshold}° → clamp {self.clamp_sigma}σ"
        )
