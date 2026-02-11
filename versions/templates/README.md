# IONIS Validation Templates

Parameterized validation scripts that work with any IONIS version via `config.json`.

## Usage

```bash
# Run verification on V18
python versions/templates/verify_template.py v18

# Run Step I validation on V18
python versions/templates/validate_template.py v18

# Works with any version that has a config.json
python versions/templates/verify_template.py v16
```

## Files

| File | Purpose |
|------|---------|
| `common.py` | Shared utilities (config loading, model loading, feature engineering) |
| `verify_template.py` | 4-test physics verification (storm, sun, gates, decomposition) |
| `validate_template.py` | Step I recall validation against contest paths |
| `config_schema.json` | JSON schema for version config files |

## Version Config

Each version needs a `config.json` in its folder:

```json
{
  "version": "v18",
  "checkpoint": "ionis_v18.pth",
  "architecture": "IonisV12Gate",
  "normalization": "global",
  "norm_keys": {
    "mean": "global_mean",
    "std": "global_std"
  },
  "thresholds": {
    "wspr": -28.0,
    "ft8": -20.0,
    "cw_machine": -18.0,
    "ssb": 5.0
  },
  "validation": {
    "step_i_recall_min": 80.0,
    "step_i_recall_max": 99.0,
    "sfi_benefit_min": 0.3,
    "kp_storm_min": 3.0
  },
  "baselines": {
    "voacap": 75.82,
    "previous_version": "v16",
    "previous_recall": 96.38
  }
}
```

## Benefits

1. **No per-version editing**: Templates load config, no hardcoded values
2. **Consistent validation**: Same tests across all versions
3. **Easy comparison**: Baselines stored in config
4. **Normalization aware**: Handles global, per-source, per-band normalization
5. **Threshold flexibility**: Mode thresholds configurable per version

## Adding a New Version

1. Create `versions/vXX/` folder
2. Copy a training script and modify
3. Create `config.json` with version-specific values
4. Run templates:
   ```bash
   python versions/templates/verify_template.py vXX
   python versions/templates/validate_template.py vXX
   ```

## Architecture Registry

Currently supported architectures in `common.py`:
- `IonisV12Gate` — V12-V18 (203,573 params)

To add new architectures, update `ARCHITECTURES` dict in `common.py`.
