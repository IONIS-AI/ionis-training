# V22 Implementation Plan — Solar Depression Angles + Band×Darkness Interaction

**Author:** Claude-M3 (Sage Node)
**Date:** 2026-02-21
**Status:** APPROVED — Reviewed by 9975 and Gemini (2026-02-21)

**Gemini Notes:**
- Cross-product mechanism confirmed as solution to gradient cancellation
- Signage correct: solar_elevation positive=day, negative=night
- CUDA port: use single-precision trig (sinf/cosf/asinf) for Tensor Core efficiency

## Executive Summary

V21-beta proved that endpoint-specific darkness features improve peak hour detection (TST-908/909 pass) but fail to teach magnitude of band closure (TST-901: 0.0 dB delta). The root cause is that `mutual_darkness` and `frequency` have opposing physical effects that the trunk averages out.

V22 introduces **solar depression angles** with **explicit band×darkness cross-products** to give the model mathematical permission to apply band-specific time penalties.

---

## V21-beta Post-Mortem

### What Worked
- TST-908: 10m peaks at 11:00 UTC (daylight) ✓
- TST-909: 160m peaks at 03:00 UTC (night) ✓
- TST-910: 40m gray line enhancement ✓
- RMSE improved: 0.831σ (better than V20's 0.862σ)
- Kp sidecar shed temporal load: +3.49σ → +1.29σ

### What Failed
- TST-901: 10m day/night delta = **0.0 dB** (should be 15-20 dB)
- TST-903/904: Mutual darkness effect too weak (3.5-4.6 dB vs 6+ dB needed)
- TST-907: Time sensitivity only 2.5 dB (need 10+ dB)

### Root Cause Analysis (Gemini)

The sigmoid-based `mutual_darkness` feature provides a smooth 0-1 curve. The optimizer used it to align peaks (improving Pearson) but never learned the magnitude cliff.

**Critical insight:** 160m and 10m have *opposite* reactions to darkness:
- 160m WANTS darkness (D-layer collapse enables propagation)
- 10m WANTS daylight (F-layer ionization required)

Without explicit cross-products, the trunk averages weights to avoid punishing either band. The opposing gradients cancel out.

---

## V22 Architecture

### New Features

| Index | Feature | Formula | Range | Purpose |
|-------|---------|---------|-------|---------|
| 10 | vertex_lat | arccos(\|sin(az)×cos(tx_lat)\|) / 90 | [0, 1] | Polar exposure (V21) |
| 11 | tx_solar_dep | solar_depression(tx_lat, tx_lon, hour, doy) / 90 | [-1, 1] | TX sun angle |
| 12 | rx_solar_dep | solar_depression(rx_lat, rx_lon, hour, doy) / 90 | [-1, 1] | RX sun angle |
| 13 | freq_x_tx_dark | freq_log × tx_solar_dep | [-1, 1] | Band×TX darkness |
| 14 | freq_x_rx_dark | freq_log × rx_solar_dep | [-1, 1] | Band×RX darkness |
| 15 | sfi | sfi / 300 | [0, 1] | Solar flux (sidecar) |
| 16 | kp_penalty | 1 - kp/9 | [0, 1] | Storm cost (sidecar) |

### Removed Features
- `mutual_darkness` (V21-beta feature 10)
- `mutual_daylight` (V21-beta feature 11)

### Retained Features (indices 0-9)
```
0: distance / 20000
1: freq_log / 8
2: hour_sin
3: hour_cos
4: az_sin
5: az_cos
6: lat_diff / 180
7: midpoint_lat / 90
8: season_sin
9: season_cos
```

### Config Changes

```json
{
  "model": {
    "architecture": "IonisGate",
    "dnn_dim": 15,
    "hidden_dim": 256,
    "sidecar_hidden": 8,
    "input_dim": 17,
    "sfi_idx": 15,
    "kp_penalty_idx": 16,
    "gate_init_bias": -0.693
  },
  "features": [
    "distance", "freq_log", "hour_sin", "hour_cos",
    "az_sin", "az_cos", "lat_diff", "midpoint_lat",
    "season_sin", "season_cos", "vertex_lat",
    "tx_solar_dep", "rx_solar_dep",
    "freq_x_tx_dark", "freq_x_rx_dark",
    "sfi", "kp_penalty"
  ]
}
```

---

## Solar Depression Computation

### Algorithm

```python
import math

def solar_depression_deg(lat, lon, hour_utc, day_of_year):
    """
    Compute solar depression angle in degrees.

    Positive = sun above horizon (daylight)
    Negative = sun below horizon (night)

    Args:
        lat: Latitude in degrees (-90 to 90)
        lon: Longitude in degrees (-180 to 180)
        hour_utc: Hour of day in UTC (0-23, can be float)
        day_of_year: Day of year (1-366)

    Returns:
        Solar elevation angle in degrees (-90 to +90)
    """
    # Solar declination (simplified equation)
    # Max +23.44° at summer solstice, min -23.44° at winter solstice
    dec = -23.44 * math.cos(math.radians(360 / 365 * (day_of_year + 10)))
    dec_r = math.radians(dec)
    lat_r = math.radians(lat)

    # Hour angle: degrees from solar noon
    # Solar noon occurs when sun is directly south (northern hem) or north (southern)
    solar_hour = hour_utc + lon / 15.0  # Local solar time
    hour_angle = (solar_hour - 12.0) * 15.0  # 15 deg per hour from noon
    ha_r = math.radians(hour_angle)

    # Solar elevation (altitude) formula
    sin_elev = (math.sin(lat_r) * math.sin(dec_r) +
                math.cos(lat_r) * math.cos(dec_r) * math.cos(ha_r))

    # Clamp to valid range for arcsin
    sin_elev = max(-1.0, min(1.0, sin_elev))
    elevation = math.degrees(math.asin(sin_elev))

    return elevation
```

### Physical Thresholds

| Elevation | Twilight Zone | D-layer | F-layer | Low Bands | High Bands |
|-----------|---------------|---------|---------|-----------|------------|
| > 0° | Daylight | Absorbing | Fully ionized | Blocked | Open |
| 0° to -6° | Civil | Weakening | Ionized | Opening | Open |
| -6° to -12° | Nautical | Collapsed | Residual | **Sweet spot** | Closing |
| -12° to -18° | Astronomical | Gone | Fading | Open | Closed |
| < -18° | Night | Gone | Decayed | Open | Dead |

The model should learn these thresholds from data. We don't hardcode them.

### Vectorized Implementation (train_common.py)

```python
def solar_depression_vectorized(lat, lon, hour_utc, day_of_year):
    """Vectorized solar depression for arrays."""
    dec = -23.44 * np.cos(np.radians(360 / 365 * (day_of_year + 10)))
    dec_r = np.radians(dec)
    lat_r = np.radians(lat)

    solar_hour = hour_utc + lon / 15.0
    ha_r = np.radians((solar_hour - 12.0) * 15.0)

    sin_elev = (np.sin(lat_r) * np.sin(dec_r) +
                np.cos(lat_r) * np.cos(dec_r) * np.cos(ha_r))
    sin_elev = np.clip(sin_elev, -1.0, 1.0)

    return np.degrees(np.arcsin(sin_elev))
```

---

## Cross-Product Rationale

### The Problem
Dense layers receive both `freq_log` and `darkness` features but learn averaged weights. When training sees:
- 160m + dark → good SNR
- 10m + dark → bad SNR

The gradients for the darkness feature cancel out.

### The Solution
Explicit cross-products:
```
freq_x_tx_dark = freq_log * tx_solar_dep
freq_x_rx_dark = freq_log * rx_solar_dep
```

Now the model has separate weights for:
- "high frequency × darkness" (should be negative)
- "low frequency × darkness" (should be positive)

The gradients no longer cancel.

### Normalization

- `freq_log` is already normalized: `log10(freq_hz) / 8` → ~[0.78, 0.93] for HF
- `tx_solar_dep / 90` → [-1, 1]
- Cross-product: `(freq_log/8) × (solar_dep/90)` → roughly [-1, 1]

Both inputs are already normalized, so the cross-product stays in reasonable range.

---

## Implementation Checklist

### Phase 1: Model Code (model.py)

- [ ] Add `solar_depression_deg()` function
- [ ] Update `build_features()` to accept `day_of_year` parameter
- [ ] Add `include_solar_depression` flag (similar to `include_physics_gates`)
- [ ] Compute: tx_solar_dep, rx_solar_dep, freq_x_tx_dark, freq_x_rx_dark
- [ ] Update docstrings with V22 feature indices

### Phase 2: Training Code (train_common.py)

- [ ] Add `solar_depression_vectorized()` function
- [ ] Update `engineer_features()` to check `dnn_dim >= 15`
- [ ] Extract `day_of_year` from signature data (need to add to query or compute from month)
- [ ] Compute vectorized: tx_solar_dep, rx_solar_dep, cross-products

### Phase 3: Configuration (config_v22.json)

- [ ] Create `versions/v22/` directory
- [ ] Create `config_v22.json` with new dimensions
- [ ] Copy `train_v22.py` from v21 template
- [ ] Update feature list in config

### Phase 4: Testing

- [ ] Update TST-900 test file to detect V22 checkpoint
- [ ] Update `predict()` to pass `day_of_year` parameter
- [ ] Run TST-900 baseline against V21-beta (document current failures)

### Phase 5: Training

- [ ] Train V22-alpha with rbn_full_sample=0
- [ ] Monitor Kp sidecar (expect further reduction from +1.29)
- [ ] Run TST-900 — TST-901 is the gate (10m delta must be significant)

### Phase 6: Validation

- [ ] If TST-900 passes, train V22-gamma with rbn_full_sample=20M (Pressure Vessel)
- [ ] Compare to V21-beta, V21-alpha, V20

---

## Data Considerations

### Day of Year
**Decision:** Use exact day-of-year from ClickHouse timestamp.

Solar declination changes ~0.4°/day near equinoxes. A 15-day mid-month approximation introduces up to 6° declination error — unacceptable when exact values are available.

**Implementation:**
- Training: Add `toDayOfYear(timestamp) AS day_of_year` to signature queries
- Inference: Compute from actual date (always available)

### Signature Table Query Update
```sql
SELECT
    tx_grid_4, rx_grid_4, band, hour, month,
    toDayOfYear(timestamp) AS day_of_year,  -- NEW
    median_snr, spot_count, snr_std, reliability,
    avg_sfi, avg_kp, avg_distance, avg_azimuth
FROM wspr.signatures_v2_terrestrial
WHERE avg_sfi > 0
```

Note: Signature tables aggregate by hour/month. The `timestamp` field represents the bucket start. Day-of-year from this is exact for the aggregation period.

---

## Success Criteria

### Primary Gate: TST-901
```
10m at midday:   expect > -15 dB
10m at midnight: expect < -25 dB (below WSPR floor)
Day/Night Delta: expect > 10 dB
```

### Secondary Gates
- TST-903: 160m mutual darkness >= 6 dB
- TST-907: Time sensitivity >= 10 dB dynamic range
- TST-906: Low bands beat high bands at night

### Metrics (relative to V21-beta)
| Metric | V21-beta | V22 Target |
|--------|----------|------------|
| Pearson | +0.464 | >= +0.46 (maintain) |
| RMSE | 0.831σ | <= 0.83σ (maintain or improve) |
| SFI | +0.48σ | >= +0.4σ |
| Kp | +1.29σ | May drop further (expected) |
| TST-900 | 4/10 | >= 8/10 |

---

## Risks and Mitigations

### Risk 1: Cross-products dominate, trunk ignores base features
**Mitigation:** Monitor trunk weights during training. If collapsing, reduce cross-product learning rate.

### Risk 2: Model overfits to cross-product patterns
**Mitigation:** Maintain validation split, watch for train/val divergence.

### Risk 3: Parameter count increases significantly
**Calculation:** IonisGate trunk grows from 13→15 input dims.
- Current: 13×512 + 512×256 + 256×128 + 128×1 = 6,656 + 131,072 + 32,768 + 128 = 170,624
- V22: 15×512 + same = 7,680 + 131,072 + 32,768 + 128 = 171,648
- Delta: +1,024 params (0.5% increase)
**Assessment:** Negligible impact.

---

## Review Decisions (9975 Feedback — 2026-02-21)

| Question | Decision | Rationale |
|----------|----------|-----------|
| Day-of-year | **Exact from ClickHouse** | 15-day mid-month = 6° declination error near equinoxes. No reason to approximate when exact is available. |
| hour_sin/hour_cos | **Keep for V22-alpha** | UTC encodes non-solar signals (operator activity, ionospheric tides). Test removing in V22-beta. |
| season_sin/season_cos | **Keep for V22-alpha** | Winter anomaly, equinoctial enhancements may not reduce to sun angle. Test removing later. |
| Feature ordering | **Logical groups** | `[geo][time][vertex][solar_dep][cross][sidecar]` — readability, doesn't affect training. |
| Version naming | **New V22 directory** | V21 archived like V17-V19. New feature architecture warrants new version. |

**Critical principle:** One variable at a time. V22 is surgical — swap day/night features, add cross-products, keep everything else identical.

---

## Appendix: Feature Evolution

| Version | dnn_dim | Key Features | TST-900 |
|---------|---------|--------------|---------|
| V20 | 11 | day_night_est (midpoint) | 2/10 |
| V21-alpha | 12 | + vertex_lat | 2/10 |
| V21-beta | 13 | + mutual_darkness, mutual_daylight | 4/10 |
| **V22** | **15** | **+ tx/rx_solar_dep, freq×dark cross-products** | **Target: 8/10** |

---

*Document ready for review. Awaiting feedback from 9975 and Gemini before implementation.*
