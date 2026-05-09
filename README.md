# Missingness Mechanism Matters
## Missingness Mechanism Matters: Auditing Clinical Model Selection Stability Under MCAR, MAR, and MNAR Perturbations

**Nattakitti Piyavechvirat · Faizan ul Haq · Qazi Mazhar ul Haq**   
International Bachelor Program in Informatics, Yuan Ze University, Taiwan
 
---
 
## Overview
 
This repository contains the complete code, results, and figures for our IBCAST 2026 paper.  
We audit how three structurally distinct missingness mechanisms affect **clinical model-selection stability** on the MIMIC-IV in-hospital mortality cohort.
 
**Core finding:** At 40% MNAR missingness, winner-flip rate reaches **100%** (stratified split) and **75%** (patient-grouped split), the top-1/top-2 AUROC margin collapses to **0.013 pp**, and mean missingness-indicator AUROC reaches **0.797** — evidence that the apparent winner exploits a structural shortcut rather than stable clinical signal.
 
---
 
## Repository Structure
 
```
Missingness Mechanism Matters/
│
├── Code/
│   ├── mechanism_sweep.py  # Main experiment runner (3×7×2×20 grid)
│   ├── postprocess.py      # Post-processing pipeline (9 stages)
│   └── posthoc_tables.py   # Paper-ready table generation
│
├── Figures/
│   ├── fig1_envelope_flip.*       # Winner-flip rate vs. missingness rate
│   ├── fig2_kendall_tau_heatmap.* # Kendall τ heatmap across all conditions
│   ├── fig3_auroc_boxplot_rate30.*# AUROC distribution by model at 30%
│   ├── fig4_mechanism_divergence.*# Δτ differential stability (MAR/MNAR vs MCAR)
│   ├── fig5_margin_gap.*          # Top-1/top-2 AUROC margin gap
│   └── fig6_per_feature_mnar_leak.*# Per-feature missingness-indicator AUROC
│
├── Postprocess_Result/
│   ├── metrics_raw.csv            # Raw per-seed fold metrics (21 000 rows)
│   ├── envelope_by_mechanism.csv  # Winner-flip % per (mechanism, rate, split)
│   ├── margin_gap_summary.csv     # Top-1/top-2 margin with 95% CI
│   ├── margin_gap_by_seed.csv     # Per-seed margin values
│   ├── per_feature_leak_summary.csv  # Mean/max indicator AUROC per condition
│   ├── per_feature_leak_by_seed.csv  # Per-seed per-feature indicator AUROC
│   ├── winners_by_seed.csv        # Winner model per (seed, condition)
│   ├── summary_by_setting.csv     # Aggregated AUROC/AP/Brier/ECE per setting
│   └── friedman_results.csv       # Friedman test results per condition
│
├── Post_Hoc_Result/
│   ├── summary.csv     # 18-row paper-ready summary table
│   ├── main_mechanism_table.csv    # Full stability metrics table
│   ├── calibration_winner_table.csv# Winner calibration metrics
│   └── shortcut_evidence_table.csv # Indicator AUROC by condition
│
└── nemenyi_posthoc/
    └── nemenyi_<MECH>_rate<RATE>_<SPLIT>.csv  # Nemenyi p-value matrices (42 files)
```
 
---
 
## Experiment Design
 
| Dimension | Values |
|-----------|--------|
| Mechanisms | MCAR, MAR, MNAR |
| Missingness rates | 0, 5, 10, 20, 30, 40, 50 % |
| Split policies | S1: StratifiedKFold · S2: GroupKFold (by `subject_id`) |
| Random seeds | 20 (1001–1020) |
| Models | LR-L2, SVM-Platt, Random Forest, XGBoost, Extra Trees |
| Outer folds | 5 · Inner folds: 3 |
| **Total evaluations** | **3 × 7 × 2 × 20 × 5 = 4 200 fold evaluations** |
 
### Missingness Mechanisms
 
| Mechanism | Formula |
|-----------|---------|
| **MCAR** | P(miss) = r, independently per value, using a separate RNG |
| **MAR** | P(miss) = min(2r, 0.95) if severity proxy > training median; else P(miss) = 0.5r. Proxy = row-mean of training-standardized numeric features, computed on training partition only |
| **MNAR** | P(miss \| y=1) = min(2.5r, 0.95); P(miss \| y=0) = 0.5r. Outcome labels used only to simulate missingness; never provided as model features |
 
### Fold-Safe Protocol (P0)
 
All preprocessing (imputation, scaling, calibration) is **fit on the training partition only**.  
Missingness injection occurs **inside each outer fold** before preprocessing.  
A calibration holdout (25% of outer-train) is carved **before hyperparameter tuning** to prevent data re-use leakage.
 
### Winner Selection
 
Lexicographic ranking per seed: **AUROC ↑ → AP ↑ → Brier ↓**
 
---
 
## Key Results
 
| Condition | Flip % | Kendall τ | Margin (pp) | Mean Ind. AUROC |
|-----------|:------:|:---------:|:-----------:|:---------------:|
| MCAR S1 50% | 20 | 0.83 | 0.385 | 0.465 |
| MCAR S2 50% | 20 | 0.83 | 0.566 | 0.465 |
| MAR S1 50% | 55 | 0.74 | 0.639 | 0.547 |
| MAR S2 50% | 30 | 0.82 | 0.570 | 0.547 |
| **MNAR S1 40%** | **100** | **0.62** | **0.013** | **0.797** |
| MNAR S2 40% | 75 | 0.67 | 0.016 | 0.797 |
 
---
 
## Dataset
 
### MIMIC-IV v3.1
The primary dataset is **not included** in this repository — MIMIC-IV is a restricted dataset governed by the [PhysioNet Data Use Agreement](https://physionet.org/content/mimiciv/).
 
To reproduce experiments:
1. Register and complete CITI training at [physionet.org](https://physionet.org)
2. Apply for MIMIC-IV access: [physionet.org/content/mimiciv/](https://physionet.org/content/mimiciv/)
3. Preprocess to generate `full_analytic_dataset_mortality_all_admissions.csv`
   - Shape: (14 081, 43) · Label: `label_mortality` · Group: `subject_id`
   - Place the file in the same directory as `Code/ibcast_mechanism_sweep.py`
---
 
## Reproducing the Results
 
```bash
# Install dependencies
pip install numpy pandas scikit-learn xgboost scipy scikit-posthocs tqdm matplotlib seaborn
 
# Step 1 — Run mechanism sweep (produces Postprocess_Result/metrics_raw.csv)
python Code/mechanism_sweep.py
 
# Step 2 — Post-process (produces all summary CSVs + Figures/)
python Code/postprocess.py
 
# Step 3 — Generate paper-ready tables (produces Post_Hoc_Result/)
python Code/posthoc_tables.py
```
 
> **Runtime:** Full sweep takes several hours on CPU. Results are checkpointed in `metrics_raw.csv` so interrupted runs can be resumed.
 
---
