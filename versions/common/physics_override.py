"""
physics_override.py — Deterministic Physics Override Layer for IONIS Inference

Post-model inference clamp for physically impossible predictions.
V22-gamma passes 16/17 KI7MT tests. The single failure (acid test: 10m EU
night, +0.540σ) cannot be fixed by retraining — eight architectures tried,
all destroyed global physics. This module applies deterministic rules at
inference time.

Rule A (both dark — F-layer collapse):
    IF freq >= 21 MHz AND tx_solar < -6° AND rx_solar < -6° AND pred > -2.0σ
    THEN clamp pred to -2.0σ

Rule B (TX deep darkness — F-layer collapse):
    IF freq >= 21 MHz AND tx_solar < -18° AND pred > -2.0σ
    THEN clamp pred to -2.0σ

Rule C (D-layer daytime absorption):
    IF freq <= 7.5 MHz AND tx_solar > 0° AND rx_solar > 0°
    AND distance > 1500 km AND pred > -2.0σ
    THEN clamp pred to -2.0σ

Physics (Rules A/B): No F2 skip above 21 MHz when the ionosphere above the
transmitter has fully decayed. Rule A handles mutual darkness (civil twilight
-6°). Rule B handles asymmetric paths where the TX is past astronomical
twilight (-18°) — the signal cannot get UP into the ionosphere for the first
hop. RX-dark with TX in daylight is NOT clamped: multi-hop paths carry the
signal across sunlit ionosphere to arrive at the dark receiver. Proven by
KI7MT QSO data (301 QSOs 15m JA at 17z, 568 QSOs 15m EU at 18z Nov).

Physics (Rule C): Daytime D-layer absorption scales as 1/f² (Appleton-Hartree).
On 160m (1.8 MHz) it is a wall of plasma; on 80m (3.5 MHz) ~4x less severe;
on 40m (7 MHz) still significant for DX paths. When BOTH endpoints are in
daylight (solar > 0°), the entire propagation path passes through ionized
D-layer. Distance > 1500 km clears the maximum single-hop NVIS range for 40m
and protects ground-wave on 160m. Greyline/terminator DX windows (one end
dark) are preserved because the rule requires both ends in daylight.

The -2.0σ clamp = -30.9 dB, well below the WSPR decode floor (-28 dB).
Display shows "—" (closed) with clear margin.

Usage:
    from physics_override import PhysicsOverrideLayer, apply_override_to_prediction

    # Single prediction
    sigma, was_overridden = apply_override_to_prediction(
        sigma, freq_mhz, tx_solar_deg, rx_solar_deg, distance_km=8500)

    # Batch (numpy arrays)
    override = PhysicsOverrideLayer()
    sigmas, audit = override(sigmas, freq_mhz, tx_solar, rx_solar, distance_km=dists)
"""

import numpy as np


# ── Constants ────────────────────────────────────────────────────────────────

# Rules A/B: High-band night closure (F-layer collapse)
FREQ_THRESHOLD_MHZ = 21.0      # 15m and above (21, 24, 28 MHz)
SOLAR_THRESHOLD_DEG = -6.0     # Civil twilight (Rule A: both endpoints)
DEEP_DARK_THRESHOLD_DEG = -18.0  # Astronomical twilight (Rule B: single endpoint)

# Rule C: Low-band day closure (D-layer absorption)
LOW_FREQ_THRESHOLD_MHZ = 7.5   # 40m and below (7, 5.3, 3.5, 1.8 MHz)
DLAYER_SOLAR_THRESHOLD_DEG = 0.0  # Both endpoints above horizon
DLAYER_DISTANCE_KM = 1500.0    # Clears max NVIS range for 40m

CLAMP_SIGMA = -2.0             # -30.9 dB, below WSPR -28 dB decode floor


# ── Single Prediction ────────────────────────────────────────────────────────

def apply_override_to_prediction(sigma, freq_mhz, tx_solar_deg, rx_solar_deg,
                                 distance_km=None):
    """Apply physics override to a single prediction.

    Three rules (any triggers the clamp):
        Rule A: freq >= 21 MHz AND tx_solar < -6° AND rx_solar < -6°
        Rule B: freq >= 21 MHz AND tx_solar < -18°
        Rule C: freq <= 7.5 MHz AND tx_solar > 0° AND rx_solar > 0°
                AND distance > 1500 km

    Args:
        sigma: Model prediction in z-score units
        freq_mhz: Operating frequency in MHz
        tx_solar_deg: TX solar elevation in degrees (negative = below horizon)
        rx_solar_deg: RX solar elevation in degrees
        distance_km: Great-circle distance in km (required for Rule C)

    Returns:
        (clamped_sigma, was_overridden) tuple
    """
    if sigma > CLAMP_SIGMA:
        # Rules A/B: High-band night closure (F-layer collapse)
        if freq_mhz >= FREQ_THRESHOLD_MHZ:
            both_dark = (tx_solar_deg < SOLAR_THRESHOLD_DEG and
                         rx_solar_deg < SOLAR_THRESHOLD_DEG)
            tx_deep_dark = (tx_solar_deg < DEEP_DARK_THRESHOLD_DEG)
            if both_dark or tx_deep_dark:
                return CLAMP_SIGMA, True

        # Rule C: Low-band day closure (D-layer absorption)
        if (distance_km is not None and
                freq_mhz <= LOW_FREQ_THRESHOLD_MHZ and
                tx_solar_deg > DLAYER_SOLAR_THRESHOLD_DEG and
                rx_solar_deg > DLAYER_SOLAR_THRESHOLD_DEG and
                distance_km > DLAYER_DISTANCE_KM):
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
                 low_freq_threshold=LOW_FREQ_THRESHOLD_MHZ,
                 dlayer_solar_threshold=DLAYER_SOLAR_THRESHOLD_DEG,
                 dlayer_distance=DLAYER_DISTANCE_KM,
                 clamp_sigma=CLAMP_SIGMA):
        self.freq_threshold = freq_threshold
        self.solar_threshold = solar_threshold
        self.deep_dark_threshold = deep_dark_threshold
        self.low_freq_threshold = low_freq_threshold
        self.dlayer_solar_threshold = dlayer_solar_threshold
        self.dlayer_distance = dlayer_distance
        self.clamp_sigma = clamp_sigma

    def __call__(self, sigmas, freq_mhz, tx_solar_deg, rx_solar_deg,
                 distance_km=None):
        """Apply physics override to a batch of predictions.

        Three rules (any triggers the clamp):
            Rule A: freq >= 21 MHz AND tx_solar < -6° AND rx_solar < -6°
            Rule B: freq >= 21 MHz AND tx_solar < -18°
            Rule C: freq <= 7.5 MHz AND tx_solar > 0° AND rx_solar > 0°
                    AND distance > 1500 km

        Args:
            sigmas: np.ndarray of shape (N,) — model predictions in z-score
            freq_mhz: np.ndarray of shape (N,) — operating frequency in MHz
            tx_solar_deg: np.ndarray of shape (N,) — TX solar elevation
            rx_solar_deg: np.ndarray of shape (N,) — RX solar elevation
            distance_km: np.ndarray of shape (N,) — great-circle distance
                         (required for Rule C)

        Returns:
            (clamped_sigmas, audit_dict) tuple
            audit_dict contains: n_overridden, override_mask, original_sigmas
        """
        sigmas = np.asarray(sigmas, dtype=np.float64)
        original = sigmas.copy()
        above_clamp = (sigmas > self.clamp_sigma)

        # Rules A/B: High-band night closure (F-layer collapse)
        high_freq = (freq_mhz >= self.freq_threshold)
        both_dark = ((tx_solar_deg < self.solar_threshold) &
                     (rx_solar_deg < self.solar_threshold))
        tx_deep_dark = (tx_solar_deg < self.deep_dark_threshold)
        night_mask = high_freq & above_clamp & (both_dark | tx_deep_dark)

        # Rule C: Low-band day closure (D-layer absorption)
        if distance_km is not None:
            distance_km = np.asarray(distance_km, dtype=np.float64)
            low_freq = (freq_mhz <= self.low_freq_threshold)
            both_day = ((tx_solar_deg > self.dlayer_solar_threshold) &
                        (rx_solar_deg > self.dlayer_solar_threshold))
            far_path = (distance_km > self.dlayer_distance)
            day_mask = low_freq & above_clamp & both_day & far_path
        else:
            day_mask = np.zeros(len(sigmas), dtype=bool)

        mask = night_mask | day_mask

        sigmas[mask] = self.clamp_sigma

        audit = {
            "n_total": len(sigmas),
            "n_overridden": int(mask.sum()),
            "n_night_override": int(night_mask.sum()),
            "n_day_override": int(day_mask.sum()),
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
            f"AND tx_solar < {self.deep_dark_threshold}° → clamp {self.clamp_sigma}σ | "
            f"Rule C: freq <= {self.low_freq_threshold} MHz "
            f"AND tx_solar > {self.dlayer_solar_threshold}° "
            f"AND rx_solar > {self.dlayer_solar_threshold}° "
            f"AND dist > {self.dlayer_distance} km → clamp {self.clamp_sigma}σ"
        )
