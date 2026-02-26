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

Rule C (D-layer daytime absorption — two tiers):
    Severe (freq <= 4.0 MHz, 80m/160m):
        IF EITHER tx_solar > 0° OR rx_solar > 0°
        AND distance > 1500 km AND pred > -2.0σ
        THEN clamp pred to -2.0σ
    Moderate (freq 4.0–7.5 MHz, 40m/60m):
        IF tx_solar > 0° AND rx_solar > 0°
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
Two tiers reflect the severity difference:
  - 80m/160m (≤ 4 MHz): D-layer at 1.8 MHz is a plasma wall; at 3.5 MHz ~4x
    less but still lethal. If EITHER endpoint has a sunlit D-layer, the signal
    cannot punch through on the first hop up or last hop down. One sunlit
    endpoint is enough to kill the path.
  - 40m/60m (4–7.5 MHz): D-layer at 7 MHz is ~15x less severe than 160m.
    Single-sided daylight paths (greyline) work on 40m — the signal can punch
    through one moderate D-layer. Only blocked when BOTH endpoints are in
    daylight (full D-layer path absorption).
Distance > 1500 km clears the maximum single-hop NVIS range for 40m and
protects ground-wave on 160m.

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

# Rule C: Low-band day closure (D-layer absorption — two tiers)
LOW_FREQ_THRESHOLD_MHZ = 7.5   # 40m and below (7, 5.3, 3.5, 1.8 MHz)
SEVERE_DLAYER_FREQ_MHZ = 4.0   # 80m and below (3.5, 1.8 MHz) — severe 1/f² absorption
DLAYER_SOLAR_THRESHOLD_DEG = 0.0  # Endpoint(s) above horizon
DLAYER_DISTANCE_KM = 1500.0    # Clears max NVIS range for 40m

CLAMP_SIGMA = -2.0             # -30.9 dB, below WSPR -28 dB decode floor


# ── Single Prediction ────────────────────────────────────────────────────────

def apply_override_to_prediction(sigma, freq_mhz, tx_solar_deg, rx_solar_deg,
                                 distance_km=None):
    """Apply physics override to a single prediction.

    Rules (any triggers the clamp):
        Rule A: freq >= 21 MHz AND tx_solar < -6° AND rx_solar < -6°
        Rule B: freq >= 21 MHz AND tx_solar < -18°
        Rule C (severe): freq <= 4.0 MHz AND EITHER solar > 0° AND dist > 1500 km
        Rule C (moderate): freq 4.0-7.5 MHz AND BOTH solar > 0° AND dist > 1500 km

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

        # Rule C: Low-band day closure (D-layer absorption — two tiers)
        if (distance_km is not None and
                freq_mhz <= LOW_FREQ_THRESHOLD_MHZ and
                distance_km > DLAYER_DISTANCE_KM):
            if freq_mhz <= SEVERE_DLAYER_FREQ_MHZ:
                # 80m/160m: EITHER endpoint in daylight → D-layer kills signal
                if (tx_solar_deg > DLAYER_SOLAR_THRESHOLD_DEG or
                        rx_solar_deg > DLAYER_SOLAR_THRESHOLD_DEG):
                    return CLAMP_SIGMA, True
            else:
                # 40m/60m: BOTH endpoints in daylight → path absorption
                if (tx_solar_deg > DLAYER_SOLAR_THRESHOLD_DEG and
                        rx_solar_deg > DLAYER_SOLAR_THRESHOLD_DEG):
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
                 severe_dlayer_freq=SEVERE_DLAYER_FREQ_MHZ,
                 dlayer_solar_threshold=DLAYER_SOLAR_THRESHOLD_DEG,
                 dlayer_distance=DLAYER_DISTANCE_KM,
                 clamp_sigma=CLAMP_SIGMA):
        self.freq_threshold = freq_threshold
        self.solar_threshold = solar_threshold
        self.deep_dark_threshold = deep_dark_threshold
        self.low_freq_threshold = low_freq_threshold
        self.severe_dlayer_freq = severe_dlayer_freq
        self.dlayer_solar_threshold = dlayer_solar_threshold
        self.dlayer_distance = dlayer_distance
        self.clamp_sigma = clamp_sigma

    def __call__(self, sigmas, freq_mhz, tx_solar_deg, rx_solar_deg,
                 distance_km=None):
        """Apply physics override to a batch of predictions.

        Rules (any triggers the clamp):
            Rule A: freq >= 21 MHz AND tx_solar < -6° AND rx_solar < -6°
            Rule B: freq >= 21 MHz AND tx_solar < -18°
            Rule C (severe): freq <= 4.0 MHz AND EITHER solar > 0° AND dist > 1500 km
            Rule C (moderate): freq 4.0-7.5 MHz AND BOTH solar > 0° AND dist > 1500 km

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

        # Rule C: Low-band day closure (D-layer absorption — two tiers)
        if distance_km is not None:
            distance_km = np.asarray(distance_km, dtype=np.float64)
            far_path = (distance_km > self.dlayer_distance)
            tx_day = (tx_solar_deg > self.dlayer_solar_threshold)
            rx_day = (rx_solar_deg > self.dlayer_solar_threshold)

            # Severe tier (80m/160m): EITHER endpoint in daylight
            severe_freq = (freq_mhz <= self.severe_dlayer_freq)
            severe_mask = severe_freq & above_clamp & (tx_day | rx_day) & far_path

            # Moderate tier (40m/60m): BOTH endpoints in daylight
            moderate_freq = ((freq_mhz > self.severe_dlayer_freq) &
                             (freq_mhz <= self.low_freq_threshold))
            moderate_mask = moderate_freq & above_clamp & tx_day & rx_day & far_path

            day_mask = severe_mask | moderate_mask
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
            f"Rule C (severe): freq <= {self.severe_dlayer_freq} MHz "
            f"AND EITHER solar > {self.dlayer_solar_threshold}° "
            f"AND dist > {self.dlayer_distance} km → clamp {self.clamp_sigma}σ | "
            f"Rule C (moderate): freq {self.severe_dlayer_freq}-{self.low_freq_threshold} MHz "
            f"AND BOTH solar > {self.dlayer_solar_threshold}° "
            f"AND dist > {self.dlayer_distance} km → clamp {self.clamp_sigma}σ"
        )
