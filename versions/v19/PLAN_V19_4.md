# V19.4 Training Plan

## Summary
V16 recipe + Rosetta Stone per-source normalization.

## Rationale

### V19.0-V19.3 Post-Mortem
All V19 variants with RBN Full data killed the storm sidecar:

| Version | RBN Full | Kp9- @ Epoch 3 | Outcome |
|---------|----------|----------------|---------|
| V19.0 | 20M | +0.05σ | DEAD |
| V19.1 | 20M | +0.14σ (frozen trunk) | DEAD after unfreeze |
| V19.2 | 20M | +0.07σ | DEAD |
| V19.3 | 5M | +0.16σ | Declining, killed early |

**Root cause**: RBN Full data (24/7 quiet-time skimmer spots) drowns the storm signal. Even at 5M rows, the trunk absorbs the gradient and sidecars starve.

### V16 Baseline (Production)
V16 achieved stable physics with this data mix:
- WSPR: 20M (floor, weak signal)
- RBN DXpedition: 91K × 50 (rare DXCC, high-value paths)
- Contest: 6.34M (ceiling, proven paths)
- **RBN Full: 0**

Results:
- RMSE: 0.860σ
- Pearson: +0.4873
- SFI+: +0.51σ (~3.2 dB)
- Kp9-: +3.66σ (~23.1 dB) — **ALIVE**
- PSK Reporter acid test: 84.14% recall

## V19.4 Configuration

```json
{
  "data": {
    "wspr_sample": 20000000,
    "rbn_full_sample": 0,        // <-- The fix
    "rbn_dx_upsample": 50,
    "contest_upsample": 1,
    "storm_upsample": 20,
    "normalization": "per_source"
  }
}
```

## Expected Outcomes

| Metric | V16 | V19.4 Expected |
|--------|-----|----------------|
| RMSE | 0.860σ | ~0.86σ |
| Pearson | +0.4873 | ~+0.48 |
| SFI+ | +0.51σ | ~+0.5σ |
| Kp9- | +3.66σ | ~+3.5σ (ALIVE) |

The Rosetta Stone per-source normalization is an **inference-time feature** that doesn't affect training dynamics. The model learns in Z-score space; the decoder maps predictions back to mode-appropriate dB scales.

## Validation Plan

1. **During training**: Monitor Kp9- stays above +0.10σ red line
2. **After training**: Run oracle test suite (35 physics tests)
3. **Acid test**: Validate against PSK Reporter fire hose (16.5M+ spots)

## Files

- `config_v19_4.json` — Configuration (source of truth)
- `train_v19_4.py` — Training script (thin wrapper)
- `oracle_v19.py` — Inference (imports from common/, reads config)

## Command

```bash
python versions/v19/train_v19_4.py 2>&1 | tee versions/v19/train_v19_4.log
```
