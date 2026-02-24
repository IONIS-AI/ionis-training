# V24 Report: The Subtraction

**Date:** 2026-02-23
**Status:** COMPLETE — REGRESSION
**Result:** TST-900 4/11 (vs V22-gamma 9/11)

## Hypothesis

Removing the sun sidecar (which contributed only a fixed +0.48σ clamp floor artifact) would free trunk capacity and improve band×time physics discrimination.

## Results

### Aggregate Metrics (Record-Breaking)

| Metric | V24 | V22-gamma | V23-alpha | Delta |
|--------|-----|-----------|-----------|-------|
| RMSE | **0.777σ** | 0.823σ | 0.815σ | -0.046σ |
| Pearson | **+0.543** | +0.492 | +0.500 | +0.051 |
| Kp9- | +1.07σ | +3.15σ | +3.00σ | -2.08σ |
| Training | 325 min | ~320 min | ~330 min | — |

### TST-900 Physics Tests (Catastrophic Failure)

| Test | Result | Measured | Threshold | Notes |
|------|--------|----------|-----------|-------|
| TST-901 | **FAIL** | +3.8 dB | 10 dB | 10m winter closure compressed |
| TST-901b | PASS | -1.9 dB | <5 dB | Summer twilight correct |
| TST-902 | **FAIL** | +2.5 dB | 8 dB | 15m winter closure compressed |
| TST-903 | **FAIL** | +3.9 dB | 5 dB | 160m mutual darkness weak |
| TST-904 | **FAIL** | +2.9 dB | 4 dB | 80m mutual darkness weak |
| TST-905 | PASS | +3.0 dB | >0 | Band order day correct |
| TST-906 | **FAIL** | -0.0 dB | >0 | Band order night FLAT |
| TST-907 | **FAIL** | +2.5 dB | 6 dB | Time sensitivity compressed |
| TST-908 | PASS | 12:00 | daylight | 10m peak hour correct |
| TST-909 | **FAIL** | 07:00 | night | 160m peaks at wrong time |
| TST-910 | PASS | +1.4 dB | >0 | Gray line correct |

**TST-900: 4/11 — REGRESSION from V22-gamma's 9/11**

## Root Cause Analysis

### The Dynamic Range Compression Problem

V24 achieved record Pearson correlation (+0.543) by **compressing the prediction range**. Instead of learning the physics extremes (band closure at night, mutual darkness enhancement), the trunk found a statistical middle ground that minimizes aggregate loss.

**Evidence:**
- Day/night deltas compressed from 10+ dB to ~3 dB
- Band ordering at night is flat (delta = -0.002σ)
- 160m peaks at 07:00 UTC instead of mutual darkness (00-06 UTC)
- Dynamic range only 2.5 dB instead of 6+ dB

### Why Aggregate Metrics Improved

Without sidecars forcing physical extremes, the trunk converged to predictions that:
1. Stay close to the mean (minimizes MSE/Huber loss)
2. Track the overall correlation pattern (maximizes Pearson)
3. Avoid the "risky" extreme predictions that physics requires

This is a classic **bias-variance tradeoff**: the sidecars added bias (monotonic physics constraints) that hurt aggregate metrics but improved physics discrimination.

### The Sidecar Inductive Bias

Even though the sun sidecar was at clamp floor (+0.48σ constant), it was still:
1. Providing a **dedicated SFI pathway** into the model
2. Forcing **some positive solar contribution** on every prediction
3. Creating **gradient signal** that the trunk had to accommodate

Removing it entirely eliminated this inductive bias. The trunk had no pressure to maintain SFI sensitivity, so it optimized for aggregate correlation instead.

## Key Finding

**Aggregate metrics and physics discrimination are inversely correlated in this architecture.**

| Direction | RMSE | Pearson | TST-900 |
|-----------|------|---------|---------|
| Better physics | Higher | Lower | Higher |
| Better aggregates | Lower | Higher | Lower |

The sidecars were not vestigial — they were essential structural constraints that forced the model to respect ionospheric physics even when the loss function would prefer flatter predictions.

## Implications for V25+

1. **Sidecars are mandatory** — Even if they contribute a fixed constant, they provide essential inductive bias

2. **The sun sidecar needs redesign, not removal** — A band-aware vector (9×1) that can express the frequency-dependent SFI curve

3. **Aggregate metrics are deceptive** — TST-900 must remain the primary gate, not RMSE/Pearson

4. **Consider regularization** — Dynamic range loss or physics-aware objectives that penalize compressed predictions

## Storm Sidecar Status

Both fc1 and fc2 converged to clamp floor (0.5000) by epoch 60:
```
Epoch 60:
  fc1: min=0.5000, max=0.5000  (dead)
  fc2: min=0.5000, max=0.5000  (dead)
```

Without the sun sidecar competing for gradient signal, the optimizer killed the storm sidecar too. The Kp9- locked at +1.07σ (clamp floor artifact, same pattern as sun sidecar in V22).

## Files

- Checkpoint: `versions/v24/ionis_v24.safetensors`
- Metadata: `versions/v24/ionis_v24_meta.json`
- Config: `versions/v24/config_v24.json`
- Training script: `versions/v24/train_v24.py`
- TST-900 tests: `versions/v24/tests/test_tst900_band_time.py`
- Training log: `results/v24_training.log`

## ClickHouse Recording

Results recorded to:
- `validation.sfi_audit_runs` — V24-ALPHA summary
- `validation.tst900_results` — Individual test results

## Conclusion

**V24 FAILED.** The subtraction hypothesis was incorrect.

The sun sidecar, even at clamp floor, provided essential inductive bias that the trunk cannot discover independently. Removing it allowed the trunk to optimize for aggregate metrics at the expense of physics discrimination.

**Recommendation:** Redesign the sun sidecar rather than remove it. The 9×1 band-aware vector sidecar remains the correct direction for V25.
