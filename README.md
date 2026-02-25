# IONIS — Training

PyTorch training and validation for the **IONIS** (Ionospheric Neural Inference System) propagation model.

## Current Model

**IONIS V22-gamma + PhysicsOverrideLayer** — Production (Phase 4.0)
- 205,621 parameters (IonisGate architecture)
- 17 features: geography, time, solar depression, freq x dark cross-products
- Trained on 38.7M rows: WSPR (floor) + DXpedition 50x (rare paths) + Contest (ceiling)
- PhysicsOverrideLayer: deterministic clamp for high-band night closure
- Config-driven: `versions/v22/config_v22.json`

| Metric | Value |
|--------|-------|
| Pearson | **+0.492** |
| RMSE | 0.821σ |
| KI7MT | 17/17 (with PhysicsOverrideLayer) |
| TST-900 | 9/11 |
| SFI sidecar | +0.482σ |
| Kp sidecar | +3.02σ (distilled) |

## Repository Structure

```
ionis-training/
├── versions/           # Self-contained version folders
│   ├── v22/           # V22-gamma production (locked)
│   ├── v20/           # V20 (archived)
│   ├── common/        # Shared training infrastructure (model.py, train_common.py)
│   └── v23-v27/       # Experimental (all failed — see CLAUDE.md)
├── scripts/           # Shared utilities
│   ├── signature_search.py    # kNN search over 93.6M signatures
│   ├── voacap_batch_runner.py # VOACAP comparison harness
│   └── coverage_heatmap.py    # Grid coverage visualization
├── tools/             # Development and validation tools
├── results/           # Validation results
└── README.md          # This file
```

Each version folder is self-contained:
```
versions/v22/
├── train_v22.py                    # Training script
├── model.py                        # IonisGate architecture (locked)
├── physics_override.py             # PhysicsOverrideLayer (deterministic)
├── config_v22.json                 # All hyperparameters
├── ionis_v22_gamma.safetensors     # Model checkpoint (805 KB)
├── ionis_v22_gamma_meta.json       # Training metadata
└── tests/                          # KI7MT + TST-900 test suite
```

## Architecture

```
IonisGate (205,621 params)
├── Trunk: 15 geography/time features → 512 → 256
│   (distance, freq_log, hour_sin/cos, az_sin/cos, lat_diff, midpoint_lat,
│    season_sin/cos, vertex_lat, tx/rx_solar_dep, freq_x_tx/rx_dark)
├── Base Head: 256 → 128 → 1 (baseline SNR)
├── Sun Scaler Head: 256 → 64 → 1 (geographic gate)
├── Storm Scaler Head: 256 → 64 → 1 (geographic gate)
├── Sun Sidecar: MonotonicMLP (SFI → SNR boost)
├── Storm Sidecar: MonotonicMLP (Kp → SNR penalty)
└── PhysicsOverrideLayer (post-inference, deterministic):
    IF freq >= 21 MHz AND both endpoints < -6° solar → clamp to -1.0σ
```

**Key innovation:** Gated monotonic sidecars enforce physics constraints (SFI+, Kp-) while allowing geographic modulation of sensitivity. The PhysicsOverrideLayer adds a deterministic post-inference clamp for high-band night closure that the neural network cannot learn from data alone.

## Quick Start

### Prerequisites
- Python 3.10+ with PyTorch 2.x
- ClickHouse access (10.60.1.1 via DAC or 192.168.1.90 LAN)
- Required tables: `wspr.signatures_v2_terrestrial`, `rbn.dxpedition_signatures`, `contest.signatures`

### Validate (no ClickHouse needed)
```bash
pip install ionis-validate
ionis-validate test              # KI7MT 18/18, TST-900 9/11
ionis-validate predict \
    --tx-grid DN46 --rx-grid JN48 --band 10m \
    --sfi 150 --kp 2 --hour 2 --month 2 --day-of-year 45
```

### Reproduce V22-gamma (requires ClickHouse)
```bash
cd versions/v22
python train_v22.py          # ~4h on M3 Ultra (MPS)
```

## Data Sources

| Source | Table | Rows | Purpose |
|--------|-------|------|---------|
| WSPR | `wspr.signatures_v2_terrestrial` | 93.6M | Floor (-28 dB) |
| DXpedition | `rbn.dxpedition_signatures` | 260K (x50 upsample) | Rare paths (152 DXCC) |
| Contest | `contest.signatures` | 5.7M | Ceiling (+10 dB SSB) |

## Related Repositories

| Repository | Purpose |
|------------|---------|
| [ionis-core](https://github.com/IONIS-AI/ionis-core) | DDL schemas, SQL |
| [ionis-apps](https://github.com/IONIS-AI/ionis-apps) | Go data ingesters |
| [ionis-cuda](https://github.com/IONIS-AI/ionis-cuda) | CUDA signature engine |
| [ionis-validate](https://github.com/IONIS-AI/ionis-validate) | Model validation suite (PyPI) |
| [ionis-hamstats](https://github.com/IONIS-AI/ionis-hamstats) | ham-stats.com publishing |
| [ionis-docs](https://github.com/IONIS-AI/ionis-docs) | Documentation site |

## License

GPLv3 — See [COPYING](COPYING) for details.

## Author

Greg Beam, KI7MT
