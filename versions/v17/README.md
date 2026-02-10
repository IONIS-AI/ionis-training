# IONIS V17

**Date:** 2026-02-10
**Status:** Planning
**Goal:** RBN grid-enriched training

## Summary

V17 improves RBN data quality by enriching missing grids. Currently only 24% of RBN spots have geocoded grids — the rest are dropped. Grid enrichment recovers these observations.

## The Problem

| Source | Grid Coverage | Notes |
|--------|---------------|-------|
| WSPR | 100% | Grid is required in protocol |
| Contest | 100% | Logs include grid/QTH |
| RBN | **24%** | Only if spotter has grid in database |

V16 uses 91K RBN DXpedition signatures (50x upsampled). With grid enrichment, we could have 4-10x more RBN data.

## Grid Enrichment Approaches

### Option 1: Callsign → Grid Lookup

Use external databases to map callsigns to grids:
- QRZ.com API (requires subscription)
- HamQTH (free, less complete)
- FCC ULS database (US only)
- Club Log (DXpedition-focused)

**Pro:** High accuracy for fixed stations
**Con:** Doesn't work for /P, /M, contest expeditions

### Option 2: DXCC → Country Centroid

For spots without grid, use DXCC entity to assign approximate location:
- Map callsign prefix → DXCC entity
- Assign country centroid as grid
- Lower precision but 100% coverage

**Pro:** Works for all callsigns
**Con:** 500-2000 km error for large countries

### Option 3: Hybrid

1. Try callsign lookup first (high precision)
2. Fall back to DXCC centroid (full coverage)

## Training Data Impact

| Version | RBN Signatures | Method |
|---------|---------------|--------|
| V16 | 91K | Geocoded only (24%) |
| V17 | ~400K+ | Grid-enriched (est.) |

## Checklist

- [ ] Analyze RBN grid coverage gaps
- [ ] Implement grid enrichment (Go tool on 9975WX)
- [ ] Create `rbn.enriched_signatures` table
- [ ] Train V17 with enriched RBN data
- [ ] Validate Step I recall
- [ ] Compare V17 vs V16

## Validation: PSK Reporter (Separate Track)

PSK Reporter provides independent validation data:
- 14.3M spots collected (15 hours)
- ~321 spots/sec current rate
- Files: `/mnt/pskr-data/2026/02/10/spots-*.jsonl.gz`

Quick Python validation script (Option B):
- Read JSONL.gz files directly over DAC link
- Filter to spots with both grids present
- Compare IONIS predictions vs actual observations
- No ClickHouse required for validation

## Files

| File | Purpose |
|------|---------|
| `train_v17.py` | Training with enriched RBN |
| `validate_v17.py` | Step I recall validation |
| `validate_pskr.py` | PSK Reporter live validation |
| `REPORT_v17.md` | Final report |

## Prerequisites

- RBN grid enrichment tool (9975WX)
- `rbn.enriched_signatures` table
- V16 checkpoint (baseline)
