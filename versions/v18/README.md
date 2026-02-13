# IONIS V18 — Global Normalization Fix

**Date:** 2026-02-10
**Status:** Ready for Training
**Purpose:** Fix V17 normalization calibration issue

## Problem Statement

V17 used per-source per-band Z-normalization, which erased the physical dB relationship between sources:
- WSPR -18 dB → 0σ
- RBN +17 dB → 0σ
- Model saw them as identical targets

V17 physics was correct (monotonicity preserved), but absolute dB scale was lost. Denormalized predictions landed at 0-7 dB, above all thresholds → 99.99% "recall" (meaningless).

## V18 Fix (Two Changes)

### 1. Global Raw-dB Normalization

Replace per-source per-band Z-score with single global Z-score in raw dB:

```python
# Concatenate all sources in raw dB
all_snr_db = np.concatenate([wspr_snr_raw, rbn_snr_raw, contest_snr_raw, ...])

# Compute ONE global mean and std
global_mean = float(all_snr_db.mean())  # Expected: -5 to +5 dB
global_std = float(all_snr_db.std())    # Expected: 15-25 dB

# Z-normalize
targets = (all_snr_db - global_mean) / global_std

# Denormalize (in validation)
snr_db = prediction_sigma * global_std + global_mean
```

Physical ordering preserved:
- WSPR -18 dB → z ≈ -1.0σ (weak signal)
- RBN +17 dB → z ≈ +0.8σ (strong signal)

### 2. Storm Upsample

RBN data is biased toward quiet conditions (skimmers decode fewer signals during storms). This diluted V17's storm sidecar from +3.4σ to +2.5σ.

Fix: Upsample storm-time signatures (Kp >= 5) by 10x to give the storm sidecar clean gradient signal.

## Expected Outcomes

| Metric | V16 | V17 (broken) | V18 (expected) |
|--------|-----|--------------|----------------|
| global_mean | N/A | 0.44σ (mixed) | -5 to +5 dB |
| global_std | N/A | 1.50σ | 15-25 dB |
| Pearson | +0.487 | +0.384 | > +0.45 |
| Kp storm | +3.4σ | +2.5σ | > +3.0σ |
| PSKR recall | 84.14% | 99.99% | 80-95% |
| Step I recall | 96.38% | 100% | 85-95% |

## Files

| File | Purpose | Status |
|------|---------|--------|
| `train_v18.py` | Training with global normalization | Ready |
| `ionis_v18.pth` | Checkpoint | Pending training |
| `oracle_v18.py` | 35-test physics suite | Ready |
| `verify_v18.py` | 4 sidecar verification tests | Ready |
| `validate_v18.py` | Step I recall vs contest paths | Ready |
| `validate_pskr.py` | PSK Reporter acid test | Ready |

## Verification Checklist

After training, verify:
1. `global_mean` is in dB (-5 to +5, NOT ~0.4)
2. `global_std` is in dB (15-25, NOT ~1.5)
3. Denormalized predictions span -30 to +30 dB
4. PSKR recall is realistic (80-95%, NOT 99%)
5. Step I recall is realistic (85-95%, NOT 100%)
6. Kp storm cost > +3.0σ
7. SFI benefit > +0.3σ
8. Physics tests 35/35 pass

## Training Command

```bash
cd /Users/gbeam/workspace/ionis-ai/ionis-training
.venv/bin/python versions/v18/train_v18.py 2>&1 | tee versions/v18/train_v18.log
```

## What NOT Changed

- Architecture: IonisV12Gate (203,573 params)
- Feature engineering: 13 features
- Source mix: WSPR + RBN Full + RBN DX (50x) + Contest
- Sample sizes: 20M + 20M + 91K×50 + 6.3M
- Loss: HuberLoss (delta=1.0)
- Optimizer: AdamW + CosineAnnealing + differential LR
- Gate variance regularization (LAMBDA_VAR=0.001)
- Weight by spot_count per source

## Reference

- Root cause analysis: `shared-context/v18-normalization-fix.md`
- V17 postmortem: `planning/v17/POST_V17_ROADMAP_DRAFT.md` (Rev 8)
