# IONIS V23 TST-900 Results

**Date**: 2026-02-23
**Checkpoint**: `versions/v23/ionis_v23.safetensors`
**Architecture**: IonisGate (207,157 params, dnn_dim=18)

## Training Metrics

| Metric | Value |
|--------|-------|
| RMSE | 0.8126 sigma |
| Pearson | +0.4997 |
| SFI Benefit | +0.482 sigma |
| Storm Cost | +2.569 sigma |

## TST-900 Results: 4/11 PASSED (FAIL - below threshold)

| Test | Description | Result |
|------|-------------|--------|
| TST-901 | 10m Band Closure (Winter) | **FAIL** (+3.2 dB vs 10 dB needed) |
| TST-901b | 10m Summer Twilight | PASS |
| TST-902 | 15m Band Closure (Winter) | **FAIL** (+0.6 dB vs 8 dB needed) |
| TST-903 | 160m Mutual Darkness | **FAIL** (+0.1 dB vs 5 dB needed) |
| TST-904 | 80m Mutual Darkness | **FAIL** (+0.3 dB vs 4 dB needed) |
| TST-905 | Band Order Day | PASS |
| TST-906 | Band Order Night | **FAIL** (reversed) |
| TST-907 | Time Sensitivity | **FAIL** (+3.4 dB vs 6 dB needed) |
| TST-908 | 10m Peak Hour | PASS |
| TST-909 | 160m Peak Hour | **FAIL** (08:00 UTC vs night) |
| TST-910 | 40m Gray Line | PASS |

## Comparison to V22-gamma Baseline

| Metric | V22-gamma | V23 | Delta |
|--------|-----------|-----|-------|
| TST-900 Passed | 9/11 | 4/11 | **-5 tests** |
| Pearson | +0.4917 | +0.4997 | +0.008 |
| RMSE | 0.821 sigma | 0.8126 sigma | -0.008 |

## Regression Analysis

V23 regressed on 5 tests compared to V22-gamma:
- TST-901: 10m Band Closure (Winter) PASS -> FAIL
- TST-902: 15m Band Closure (Winter) PASS -> FAIL
- TST-906: Band Order Night PASS -> FAIL
- TST-907: Time Sensitivity PASS -> FAIL
- TST-909: 160m Peak Hour PASS -> FAIL

## Root Cause Hypotheses

1. **IRI Feature Interference**: The 3 new IRI features (foF2_freq_ratio, foE_mid, hmF2_mid) may be providing alternative signals that allow the model to achieve good Pearson without learning proper band x time physics.

2. **SFI Sidecar Bug**: During training, SFI sidecar weights were discovered pinned at clamp floor [0.5, 2.0]. The +0.482 sigma value appears to be an architectural artifact, not learned behavior.

3. **Gradient Competition**: The IRI features may be capturing variance that should flow through the band x darkness cross-products (freq_x_tx_dark, freq_x_rx_dark).

## Recommendations

1. **V23-alpha is NOT production-ready** - fails TST-900 threshold (8/11 minimum)
2. **V22-gamma remains production baseline** - passed 9/11 TST-900
3. **Proceed with AUDIT-01/02/03** to investigate SFI sidecar bug
4. **Consider V23-beta** with IRI features as auxiliary (not trunk) inputs

## Next Steps (from Einstein/Patton AUDIT plan)

- AUDIT-01: Widen sidecar clamp [0.5, 2.0] to [0.1, 5.0] (9975WX)
- AUDIT-02: Replace Defibrillator init with Xavier Uniform (9975WX)
- AUDIT-03: Double IRI buckets from 18 to 36 (M3)
