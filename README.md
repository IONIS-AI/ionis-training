# IONIS — Training

PyTorch training and validation for the **IONIS** (Ionospheric Neural Inference System) propagation model.

## Current Model

**IONIS V20** — Production (IonisGate architecture)
- 203,573 parameters
- Trained on WSPR (floor) + DXpedition (rare paths) + Contest (ceiling)
- Config-driven: `versions/v20/config_v20.json`

| Metric | Value |
|--------|-------|
| Pearson | **+0.4879** |
| RMSE | 0.862σ |
| SFI sidecar | +0.482σ |
| Kp sidecar | +3.487σ |
| PSK Reporter recall | 84.14% |

## Repository Structure

```
ionis-training/
├── versions/           # Self-contained version folders
│   ├── v20/           # IONIS V20 production
│   ├── common/        # Shared training infrastructure
│   └── templates/     # New version templates
├── scripts/           # Shared utilities
│   ├── signature_search.py   # kNN search over 93.4M signatures
│   ├── predict.py            # Generic prediction interface
│   ├── voacap_batch_runner.py # VOACAP comparison harness
│   └── coverage_heatmap.py   # Grid coverage visualization
├── tools/             # Development tools
├── results/           # Validation results
├── GOAL.md           # Project vision
└── README.md         # This file
```

Each version folder is self-contained:
```
versions/v20/
├── train_v20.py       # Training script
├── validate_v20.py    # Step I recall validation
├── verify_v20.py      # Physics verification
├── test_v20.py        # Sensitivity analysis
├── config_v20.json    # All hyperparameters
├── ionis_v20.pth      # Model checkpoint
└── README.md          # Checklist and summary
```

## Architecture

```
IonisGate (203,573 params)
├── Trunk: 11 geography/time features → 512 → 256
├── Base Head: 256 → 128 → 1 (baseline SNR)
├── Sun Scaler Head: 256 → 64 → 1 (geographic gate)
├── Storm Scaler Head: 256 → 64 → 1 (geographic gate)
├── Sun Sidecar: MonotonicMLP (SFI → SNR boost)
└── Storm Sidecar: MonotonicMLP (Kp → SNR penalty)
```

**Key innovation:** Gated monotonic sidecars enforce physics constraints (SFI+, Kp-) while allowing geographic modulation of sensitivity.

## Quick Start

### Prerequisites
- Python 3.10+ with PyTorch 2.x
- ClickHouse access (10.60.1.1 via DAC or 192.168.1.90 LAN)
- Required tables: `wspr.signatures_v2_terrestrial`, `rbn.dxpedition_signatures`, `contest.signatures`

### Reproduce V20
```bash
cd versions/v20
python train_v20.py          # ~4h 16m on M3 Ultra
python validate_v20.py       # Step I recall
python verify_v20.py         # Physics verification
python test_v20.py           # Sensitivity analysis
```

## Data Sources

| Source | Table | Rows | Purpose |
|--------|-------|------|---------|
| WSPR | `wspr.signatures_v2_terrestrial` | 93.3M | Floor (-28 dB) |
| RBN | `rbn.dxpedition_signatures` | 91K | Rare paths (152 DXCC) |
| Contest | `contest.signatures` | 6.34M | Ceiling (+10 dB SSB) |

## Related Repositories

| Repository | Purpose |
|------------|---------|
| [ionis-core](https://github.com/IONIS-AI/ionis-core) | DDL schemas, SQL |
| [ionis-apps](https://github.com/IONIS-AI/ionis-apps) | Go data ingesters |
| [ionis-cuda](https://github.com/IONIS-AI/ionis-cuda) | CUDA signature engine |
| [ionis-validate](https://github.com/IONIS-AI/ionis-validate) | Model validation suite (PyPI) |
| [ionis-docs](https://github.com/IONIS-AI/ionis-docs) | Documentation site |

## License

GPLv3 — See [COPYING](COPYING) for details.

## Author

Greg Beam, KI7MT
