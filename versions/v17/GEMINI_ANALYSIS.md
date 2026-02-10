# V17 Gemini Pro Analysis

> Captured during V17 training run (2026-02-10). Review after final evaluation.

## Training Observations (Epoch 15)

### RMSE Structural Win
- V17 beat V16's final RMSE (0.860σ) by epoch 15 (0.840σ)
- Indicates faster convergence with real machine-measured SNR
- Model finding "global minimum" more efficiently with 20M RBN points

### The "RBN Advantage" over V16 Contest Logs
- **V16 (Ceiling Only)**: Contest logs are binary (QSO or no QSO) — model had to "guess" SNR between 0-40 dB
- **V17 (Enriched Middle)**: 20M real SNR values from RBN skimmers provide "gray scale"
- Model can now see the ramp-up and fade-out of a path, not just the peak

### Pearson Velocity
- Epoch 11: +0.2668 (crossed V15 Diamond threshold)
- Epoch 15: +0.2957
- Velocity: +3.3 pp in 4 epochs
- Projection: May push toward +0.50 Pearson — transformative leap if achieved

---

## Post-Training Audits

### 1. The "Pearson Peak" Audit
If V17 settles above +0.45:
- Indicates "High-Fidelity Climatology" — beyond simple regression
- May determine if "Gated" architecture (203K params) is sufficient
- Or if hidden layers need expansion for 50M+ row complexity

### 2. Geographic "Dead-Zone" Analysis
Re-run 32,400-grid heatmap to verify:
- Did 20M RBN rows "light up" maritime and polar paths?
- Did 50x DXpedition upsampling add signal or noise?
- Objective: Verify physics of rare paths, not just geographic memorization

### 3. Storm Sensitivity Calibration
Verify Positive Kp9- (Storm Cost) remains stable around +3.0σ:
- **Too low**: RBN skimmers "too good" — masking storm effects
- **Too high**: Model being too conservative
- Target: +3.0σ to +3.5σ (consistent with V16's +3.445σ)

---

## Potential V18 Enhancements

| Path | Name | Description | Impact |
|------|------|-------------|--------|
| A | PSK Firehose | Integrate 22M spots/day from live MQTT | Climatology → Live State |
| B | Power-Level Normalization | Model TX power (QRP 5W vs QRO 1500W) | Answer "what SNR at my power?" |
| C | Grey-Line Specialization | High-resolution time-of-day feature | Target sunrise/sunset choke points |

### Path A: PSK Firehose (Real-Time)
- Source: `pskr-collector` MQTT feed (~26M HF spots/day)
- Transform from "Climatology" tool to "Live State" tool
- Requires: Real-time solar index integration (`wspr.live_conditions`)
- Risk: Data freshness vs training stability tradeoff

### Path B: Power-Level Normalization
- Use Contest (100W-1500W) and RBN (varied) data to model TX power
- WSPR is 5W fixed — provides QRP baseline
- Allows: "What SNR if I run QRP vs QRO?"
- Requires: Power level metadata (available in some contest logs)

### Path C: Grey-Line Specialization
- Add high-resolution time-of-day feature (minute-level, not just hour)
- Target sunrise/sunset "choke points" where propagation shifts rapidly
- Current model uses hour + midpoint_lon for day/night estimate
- Enhancement: Explicit solar terminator distance feature

---

## Epoch Milestones

| Epoch | Pearson Target | RMSE Target | SFI+ Target | Notes |
|-------|----------------|-------------|-------------|-------|
| 11 | +0.26 | — | — | **PASSED** — RBN value confirmed |
| 15 | — | < 0.860σ | — | **PASSED** — Beat V16 RMSE |
| 30 | +0.35 | < 0.800σ | ~0.85 | Convergence check |
| 50 | +0.40 | — | — | Heatmap comparison point |
| 100 | +0.48+ | — | — | Match/exceed V16 Pearson |

---

*Analysis by Gemini Pro (Chief Architect role), 2026-02-10*
