### **IONIS: Ionospheric Neural Inference System**

**Phase 2/3: High-Fidelity Radio Propagation Modeling**

#### **Overview**

IONIS (Ionospheric Neural Inference System) is a deep learning framework designed to predict High-Frequency (HF) radio propagation performance using the WSPR (Weak Signal Propagation Reporter) dataset. By integrating real-time solar indices and geographic path data, IONIS provides a high-resolution estimation of Signal-to-Noise Ratio (SNR) for global transatlantic and trans-equatorial paths.

```
Pedigree of this model:
Attribute	        Formal Value
Model ID	        IONIS-V2-P3
Architecture	    Multilayer Perceptron (MLP) / PyTorch 2.x
Training Hardware	Apple M3 Ultra (76-core GPU / 96GB Unified Memory)
Data Engine	        ClickHouse @ AMD Threadripper 9975WX
Feature Set	        13-D (Includes Solar-Hour, SSN-Lat, and Day/Night Est)
Constraint	        Physically Informed (Phase 3 SNR-filtered)
```
#### **Technical Architecture**

* **Core Model:** Residual Neural Network (ResNet-style)
* **Parameters:** ~268,545
* **Input Dimensions:** 13 (Physical & Engineered Features)
* **Hidden Layers:** 256-unit width with Dropout and Batch Normalization.
* **Hardware Stack:** * **Inference/Training:** Apple M3 Ultra (96GB Unified Memory) via Metal Performance Shaders (MPS).
* **Data Engine:** AMD Threadripper 9975WX feeding a ClickHouse OLAP database.


#### **Features (V2.1 "Clean" Specs)**

IONIS uses a 13-feature vector to resolve the complex relationship between the Sun and the Earth's ionosphere:

* **Geographic:** `distance`, `azimuth`, `lat_diff`, `midpoint_lat`
* **Temporal:** `hour_sin/cos`, `season_sin/cos`
* **Solar:** `ssn` (Sunspot Number)
* **Engineered:** * `ssn_lat_interact`: Resolves how solar activity affects different latitudinal zones.
* `day_night_est`: A local solar-hour approximation for D-layer absorption modeling.

> **Note on Raw Data Quality:** IONIS is intentionally developed using "Field-Observed" WSPR data. Previous iterations identified a ~14% artifact rate where actual frequency values (Hz/MHz) leaked into the Band ID column. Phase 3 implements strict range-bound filtering to mitigate these artifacts while maintaining the "noisy" reality of a global distributed sensor network.


#### **Performance Baseline (Phase 2)**

* **Dataset:** 10,000,000 rows (Solar Cycle 25: 2020–2026).
* **Best Validation RMSE:** **8.04 dB**
* **Training Time:** ~30-45 minutes on M3 Ultra.

#### **Current Objectives (Phase 3: The Purge)**

We are currently refining the model to eliminate "inverted solar physics" caused by raw data noise. By implementing strict physical filters, we are moving from a raw archive to a "Clean Signal" dataset:

* **SNR Clipping:** -35 dB to +25 dB (removing software artifacts).
* **Path Filtering:** > 500 km (focusing strictly on Skywave propagation).
* **Band Validation:** 1.8 MHz to 50 MHz (excluding data-leakage artifacts).

---

### **How to Use**

1. **Training:** Run `python scripts/train_v2_pilot.py` to ingest 10M rows from ClickHouse and generate `models/ionis_v2.pth`.
2. **Validation:** Run `python scripts/test_v2_sensitivity.py` to execute a "Solar Sweep" and verify that the model respects ionospheric laws (SSN/SNR correlation).