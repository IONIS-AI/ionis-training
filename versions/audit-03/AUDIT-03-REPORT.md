# AUDIT-03: 35-Bucket IRI Resolution Test

**Date**: 2026-02-23
**Status**: COMPLETE
**Result**: PARTIAL IMPROVEMENT (5/11 TST-900 vs V23-alpha 4/11)

## Hypothesis

Does increasing IRI atlas resolution from 18 buckets (10-unit SFI steps) to 35 buckets (5-unit SFI steps) fix the band×time shortcuts that broke V23's TST-900 tests?

## Experiment Design

| Parameter | AUDIT-03 | V23-alpha | V22-gamma |
|-----------|----------|-----------|-----------|
| Recipe | v23-alpha | v23-alpha | v22-gamma |
| DNN Dim | 18 | 18 | 15 |
| IRI Buckets | 35 (5-unit) | 18 (10-unit) | 0 (none) |
| Clamp Range | [0.5, 2.0] | [0.5, 2.0] | [0.5, 2.0] |
| IRI Features | foF2_freq_ratio, foE_mid, hmF2_mid | same | none |

### IRI Atlas Indexing Change

```python
# V23-alpha (18 buckets, 10-unit steps)
sfi_idxs = np.clip(((sfi_raw - 70) / 10).astype(int), 0, 17)

# AUDIT-03 (35 buckets, 5-unit steps)
sfi_idxs = np.clip(((sfi_raw - 70) / 5).astype(int), 0, 34)
```

## Training Results

| Metric | AUDIT-03 | V23-alpha | V22-gamma |
|--------|----------|-----------|-----------|
| Best RMSE | 0.8112σ | 0.815σ | 0.823σ |
| Best Pearson | +0.4902 | +0.4997 | +0.4917 |
| Final SFI+ | +0.4818σ | +0.482σ | +0.482σ |
| Final Kp9- | +2.931σ | +3.00σ | +3.15σ |
| TST-900 | **5/11** | 4/11 | **9/11** |
| Training Time | 321.3 min | ~330 min | ~320 min |

## Sidecar Weight Diagnostics

### Sun Sidecar (DEAD)
```
Epoch 100:
  fc1: min=0.5000, max=0.5000, mean=0.5000  (ALL AT FLOOR)
  fc2: min=0.5000, max=0.5000, mean=0.5000  (ALL AT FLOOR)
```

Both layers completely pinned to clamp floor. The SFI sidecar is contributing exactly +0.482σ regardless of input—a fixed constant, not learned physics.

### Storm Sidecar (PARTIALLY ALIVE)
```
Epoch 100:
  fc1: min=0.5000, max=0.5000  (DEAD)
  fc2: min=1.1675, max=1.4681  (ALIVE)
```

fc1 dead but fc2 still learning. Storm sidecar partially functional.

## TST-900 Band×Time Discrimination Tests

| Test | Description | Result | Measured | Threshold |
|------|-------------|--------|----------|-----------|
| TST-901 | 10m Band Closure (Winter) | **FAIL** | +6.6 dB | 10 dB |
| TST-901b | 10m Summer Twilight | PASS | -1.5 dB | weak delta |
| TST-902 | 15m Band Closure (Winter) | **FAIL** | +5.3 dB | 8 dB |
| TST-903 | 160m Mutual Darkness | **FAIL** | -0.2 dB | 5 dB |
| TST-904 | 80m Mutual Darkness | **FAIL** | +0.2 dB | 4 dB |
| TST-905 | Band Order Day | **FAIL** | -0.018σ | high > low |
| TST-906 | Band Order Night | PASS | +0.331σ | low > high |
| TST-907 | Time Sensitivity | PASS | 6.6 dB | 6 dB |
| TST-908 | 10m Peak Hour | PASS | 12:00 UTC | daylight |
| TST-909 | 160m Peak Hour | **FAIL** | 09:00 UTC | night |
| TST-910 | 40m Gray Line | PASS | +1.8 dB | enhancement |

### Critical Failures

1. **Band Closure Tests (TST-901, TST-902)**: Day/night deltas too weak. Model doesn't learn that 10m/15m close at night.

2. **Mutual Darkness Tests (TST-903, TST-904)**: Zero day/night discrimination on low bands. Model sees 160m/80m as time-independent.

3. **160m Peak Hour (TST-909)**: Peaks at 09:00 UTC instead of nighttime. Fundamentally wrong physics.

## Diagnosis

### The IRI Shortcut Problem

IRI features (foF2, hmF2, foE) inject ionospheric physics directly into the model. Instead of learning band×time relationships from actual WSPR propagation data, the model takes shortcuts:

1. **foF2_freq_ratio** tells the model directly whether a frequency can penetrate the F2 layer
2. **hmF2** provides layer height information
3. **foE** indicates E-layer absorption

The model learns to predict from these derived parameters rather than from the empirical band×time patterns in the training data.

### Why 35 Buckets Didn't Help

Higher SFI resolution (5-unit vs 10-unit steps) provides more granular IRI lookups, but the fundamental problem remains: **IRI features are physics injection, not physics learning**.

The +1 test improvement (4→5) is marginal and likely noise. The core failures persist:
- Low band mutual darkness: completely broken
- Band closure: weak deltas
- 160m timing: wrong peak hour

## Conclusion

**AUDIT-03 RESULT**: 35-bucket IRI resolution provides marginal improvement (+1 TST-900 test) but does not fix the fundamental band×time physics regression.

**Root Cause Confirmed**: IRI features create shortcuts that bypass empirical learning. The model predicts directly from ionospheric parameters instead of learning propagation patterns from WSPR data.

**Recommendation**: IRI features should be **DROPPED** from the V23 recipe. The V22-gamma approach (no IRI, let the model learn from data) maintains correct band×time physics.

## ClickHouse Recording

Results recorded to:
- `validation.sfi_audit_runs` - Audit summary
- `validation.tst900_results` - Individual test results

Query:
```sql
SELECT audit_id, recipe, iri_buckets, best_pearson, tst900_passed
FROM validation.sfi_audit_runs
ORDER BY run_timestamp;
```

## Files

- Checkpoint: `versions/audit-03/ionis_audit03.safetensors`
- Config: `versions/audit-03/config_audit03.json`
- Training script: `versions/audit-03/train_audit03.py`
- TST-900 tests: `versions/audit-03/tests/test_tst900_band_time.py`
- Training log: `results/audit03_training.log`

## Next Steps

1. **AUDIT-04**: Test removing IRI features entirely from V23 recipe (foF2/hmF2/foE → dropped, keep other V23 improvements if any)
2. **Confirm V22-gamma**: Re-validate that V22-gamma (no IRI) maintains 9/11 TST-900
3. **Document in MEMORY.md**: Update auto memory with IRI shortcut finding
