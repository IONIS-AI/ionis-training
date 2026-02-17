# IONIS V20 — Production

## Purpose

V20 is the production model checkpoint. It validates that the training pipeline
produces consistent, reproducible results.

## Success Criteria

| Metric | Target | V20 Reference |
|--------|--------|---------------|
| Pearson | > +0.48 | +0.4879 |
| Kp sidecar | > +3.0σ | +3.487σ |
| SFI sidecar | > +0.4σ | +0.482σ |
| RMSE | < 0.87σ | 0.862σ |

## Architecture Constraints

These constraints are **non-negotiable**:

1. **Architecture**: IonisGate — gates from trunk output (256-dim)
2. **Loss**: HuberLoss(delta=1.0) — robust to synthetic contest anchors
3. **Regularization**: Gate variance loss — forces context-dependent behavior
4. **Init**: Defibrillator — weights uniform(0.8-1.2), fc2.bias=-10.0, fc1.bias frozen
5. **Constraint**: Weight clamp [0.5, 2.0] after EVERY optimizer.step()
6. **Data**: WSPR + RBN DXpedition + Contest (no RBN Full)

## Data Recipe

| Source | Volume | Role |
|--------|--------|------|
| WSPR signatures | 20M | Floor (-28 dB) |
| RBN DXpedition | 91K × 50 = 4.55M | Rare paths (152 DXCC) |
| Contest | ~6M | Ceiling (SSB +10 dB) |
| RBN Full | 0 | Not used in V20 |

## Files

- `train_v20.py` — Training script
- `config_v20.json` — Training configuration
- `ionis_v20.pth` — Checkpoint
- `verify_v20.py` — Physics verification tests
- `test_v20.py` — Sensitivity analysis
- `validate_v20.py` — Step I recall validation
- `validate_v20_pskr.py` — Live PSKR validation

## Running

```bash
cd $IONIS_WORKSPACE/ionis-training
python versions/v20/train_v20.py
python versions/v20/verify_v20.py
python versions/v20/test_v20.py
```
