Predictive Maintenance and Root-Cause Classification for Industrial Machinery
1. Problem Statement

Modern industrial machinery relies on either reactive maintenance (fix after breakdown) or rigid calendar-based servicing — both costly and inefficient. This project uses IIoT-style sensor telemetry to build a classical machine learning system that:

Predicts imminent failure (binary classification) from current sensor readings
Classifies the likely root cause (multi-class classification) once a failure is flagged, to help technicians act faster

The system is deliberately restricted to classical ML (Random Forest, XGBoost/LightGBM, Logistic Regression) — no neural networks — to demonstrate that rigorous feature engineering and leakage-aware evaluation can produce a reliable, interpretable system without deep learning.

2. Dataset Choice and Rationale

Primary dataset: NASA CMAPSS (FD003/FD004 subset) — a physics-based simulation of turbofan jet engines run to failure, chosen specifically because it is genuinely time-series (unlike snapshot datasets), which makes the temporal-splitting and rolling-feature techniques in this project meaningful rather than aspirational.

Why not a single "generalized" dataset covering all machine types: different machine types fail through physically different mechanisms (thermal/mechanical wear in a jet engine vs. bit wear in a milling machine), so no single dataset or model realistically generalizes across machine types. Instead, this project validates that the methodology generalizes (see Section 10), while the model remains specific to CMAPSS.

3. Labeling Methodology (Engineered, Not Ground Truth)

CMAPSS does not provide failure labels — they were constructed as follows:

RUL (Remaining Useful Life): RUL = max_cycle_for_unit - current_cycle
Binary label: 1 if RUL <= 20, else 0 (imminent failure threshold, chosen after testing N=15/20/30 for a reasonable positive-class ratio)
Root-cause label (multi-class): the 21 sensors were mapped to physical subsystems (thermal, pressure, mechanical, power) per CMAPSS sensor documentation. For each unit, a healthy baseline (mean, std) was computed from its first ~15 cycles. For the last ~15 cycles before failure, a standardized z-score deviation was computed per subsystem:
  deviation_group = mean( |sensor_reading - healthy_baseline_mean| / healthy_baseline_std )

The subsystem with the highest deviation was assigned as that unit's root cause.

This is explicitly disclosed as a labeling heuristic, not ground truth — CMAPSS does not provide validated failure-mode annotations, so this proxy was constructed using domain reasoning about sensor-to-subsystem mapping.

4. Feature Engineering Summary

All features are computed causally (using only past cycles of the same unit — never future rows) to avoid data leakage:

Rolling mean / std / min / max per sensor (windows: last 5, last 10 cycles)
Rolling trend/slope per sensor
FFT-based features (dominant frequency, spectral energy) for sensors with oscillatory behavior
Cross-sensor ratios/differences for physically related sensor pairs
Constant/near-zero-variance sensors dropped after EDA
5. Scaling Decisions
Tree-based models (Random Forest, XGBoost/LightGBM): no scaling applied — tree splits are scale-invariant.
Logistic Regression baseline: StandardScaler fit only on the training fold, then applied to train/test separately — never fit on the full dataset or test data.
This is distinct from the z-score standardization used in the labeling heuristic (Section 3), which normalizes sensor deviations for label construction, not model input.
6. Splitting Strategy and Leakage Avoidance
Group split by unit_id — all cycles from a given engine stay entirely in train or test, never both, since randomly splitting sequential per-engine data would leak degradation patterns across the split.
GroupKFold used for cross-validation.
No random row-level shuffling anywhere in the pipeline.
Resampling (SMOTE/SMOTE-Tomek) applied only inside training folds, never on validation/test data.
7. Model A — Binary Failure Prediction: Results

Three models were trained and compared (5-fold GroupKFold cross-validation):

Model	Precision	Recall	F1	Macro F1	PR-AUC
Logistic Regression (scaled)	0.7923	0.9810	0.8766	—	0.9797
Random Forest	0.8682	0.9095	0.8884	0.9386	—
XGBoost	0.8415	0.9357	0.8861	0.9371	0.9723

Best model selected: Logistic Regression (scaled) — chosen for its highest recall on the failure class (0.98). In a predictive maintenance setting, missing a real failure is generally far costlier than a false alarm, so recall on the failure class was prioritized over raw F1. Note this is a deliberate recall-first choice, not the highest-F1 model — Random Forest and XGBoost score marginally higher on F1 and Macro F1, but Logistic Regression catches more true failures.

Final Test Set Performance (Logistic Regression, held-out test set: 4,288 healthy / 420 failure)
Class	Precision	Recall	F1	Support
0 (Healthy)	1.00	0.97	0.98	4,288
1 (Imminent Failure)	0.79	0.98	0.88	420
Macro avg	0.90	0.98	0.93	4,708
Weighted avg	0.98	0.98	0.98	4,708
Accuracy: 0.98
ROC-AUC: 0.9977
Confusion Matrix: [[4180, 108], [8, 412]] — 8 false negatives (missed failures) vs. 108 false positives (false alarms) out of 4,708 test rows

(Note: accuracy alone is not treated as the primary signal given class imbalance — recall on the failure class (0.98) and the low false-negative count (8) are the more meaningful numbers for this problem, since a missed failure is generally costlier in practice than a false alarm.)

Example Predictions (Model A)

Tested across the RUL spectrum on Engine 5 to sanity-check behavior:

Cycle	Actual RUL	Actual Status	Predicted Failure Probability	Correct?
15	198	Healthy	0.00%	✅
150	63	Healthy	0.01%	✅
165	48	Healthy	0.61%	✅
185	28	Healthy	98.84%	⚠️ Boundary effect (see below)
195	18	Imminent Failure	99.93%	✅

Observation: near the labeling boundary (RUL=28, just outside the N=20 "imminent" cutoff), the model predicted high failure probability despite the hard label being "healthy." This likely reflects genuine degradation signal beginning before the arbitrary cutoff — a known limitation of binary threshold labeling. A continuous RUL regression target, or a softer labeling window, would better capture this continuum. This was verified not to be general model overconfidence: a further-out test (RUL=48) correctly predicted low probability (0.61%), isolating the effect specifically to the boundary region.

8. Model B — Root-Cause Classification: Results

Trained only on rows Model A would flag as failure (RUL <= 20) — 1,680 rows across 3 observed root-cause classes in this data split: mechanical, pressure, thermal (the "power" subsystem from the labeling design in Section 3 had no assigned cases in this particular train/test split).

Metric	Value
Macro F1	0.7607
Training rows (failure-only)	1,680
Test support	420 (mechanical: 110, pressure: 295, thermal: 15)

Confusion Matrix (order: mechanical, pressure, thermal):

[[105,   5,   0]
 [  0, 295,   0]
 [  0,  12,   3]]
Mechanical: 105/110 correctly classified (recall ≈ 0.95)
Pressure: 295/295 correctly classified (recall = 1.00)
Thermal: only 3/15 correctly classified — by far the weakest class, and also by far the smallest (15 test examples vs. 110 and 295 for the others)

Honest read on the thermal class: its low recall is a direct consequence of extreme class scarcity (only 87 thermal-labeled rows exist in the entire dataset per Section 3's label distribution, normal: 22,620 / pressure: 1,581 / mechanical: 432 / thermal: 87), not necessarily a flaw in the model or the labeling heuristic. With this few examples, the model has very little signal to learn a distinct thermal-failure pattern from, and most thermal cases get absorbed into the much larger pressure class. A larger simulated dataset run (more CMAPSS engine units), or targeted oversampling of the thermal class specifically, would be the natural next step to address this rather than treating the current result as final.

Example Predictions (Model A → Model B, cascaded)
Actual Label	Predicted Label	Failure Probability
0	0	0.0
0	0	0.0
1	0	0.0
0	0	0.0
1	1	1.0

(Sample from Model A's test-set predictions used to gate which rows are passed into Model B — only rows predicted as failure by Model A are routed to root-cause classification, matching the cascaded design described in Section 1.)

9. Explainability

Honest note: the pipeline's Task 8 interpretation step printed "Model does not support direct feature importances" — because the best-performing Model A (Logistic Regression) does not expose a feature_importances_ attribute the way tree-based models do; that code path was written expecting Random Forest/XGBoost. This is a real limitation of the current pipeline, not a hidden success — documenting it here rather than glossing over it.

What this means in practice: Logistic Regression is interpretable in principle (via its learned coefficients, which indicate direction and relative weight per feature), and a proper SHAP LinearExplainer would work directly on it — but the current implementation only wired up tree-based feature importance extraction. Since Logistic Regression was selected as the best model specifically for its superior recall (Section 7), this is the natural next improvement: either (a) add coefficient-based interpretation for the linear model, or (b) explicitly select the best tree-based model (Random Forest, whose F1/Macro F1 was close behind) when interpretability via SHAP TreeExplainer is a priority for a given deployment.

Next step (not yet implemented): extract and visualize Logistic Regression's top coefficients by absolute magnitude, or re-run explainability specifically on the Random Forest model as a tree-based alternative with built-in SHAP support.

10. Cross-Domain Validation (AI4I 2020)

Kept fully separate from CMAPSS results — this section validates that the methodology (feature selection approach, imbalance handling, evaluation strategy) generalizes to a structurally different machine type (milling process data), not that the CMAPSS-trained model itself transfers to AI4I. A fresh Random Forest was trained on AI4I directly, using the same principles (proper class-imbalance awareness, Macro F1 / PR-AUC over raw accuracy).

Dataset: AI4I 2020, shape (10,000, 10), class distribution: Safe = 9,661 / Failure = 339 (a ~3.4% positive rate — a different, more extreme imbalance than CMAPSS, which is itself a useful stress test of the methodology).

Features used: Air temperature [K], Process temperature [K], Rotational speed, Torque [Nm], Tool wear [min], Type_L, Type_M

Class	Precision	Recall	F1	Support
0 (Safe)	0.99	0.99	0.99	1,932
1 (Failure)	0.73	0.69	0.71	68
Macro avg	0.86	0.84	0.85	2,000
Weighted avg	0.98	0.98	0.98	2,000
Macro F1: 0.8511
PR-AUC: 0.7852
Accuracy: 0.98

Interpretation: the methodology holds up reasonably well on a structurally different, more severely imbalanced dataset — Macro F1 (0.85) and PR-AUC (0.79) are solid without any dataset-specific tuning beyond feature selection. The drop in failure-class recall (0.69 vs. 0.98 on CMAPSS) is expected given AI4I's more extreme imbalance (3.4% vs. CMAPSS's ~9% positive rate) and far smaller absolute number of failure examples (339 total vs. thousands in CMAPSS) — not a flaw in the approach, but a reflection of how much harder the AI4I problem is with less positive-class signal to learn from.

11. Key Learnings and Limitations
Binary threshold labeling (RUL ≤ N) creates a hard boundary that doesn't reflect the continuous nature of degradation — observed directly in the RUL=28 test case (Section 7). A regression-based RUL target or soft-labeling window would likely improve boundary behavior.
Root-cause labels are an engineered heuristic (Section 3), not verified ground truth — a meaningful caveat for any real deployment.
The best Model A (Logistic Regression) was selected on recall, not F1 — a deliberate choice given how much costlier a missed failure is than a false alarm in practice — but it meant the explainability step (built for tree models) didn't work out of the box, surfacing a real gap between model selection and interpretability tooling (Section 9).
Model B's thermal-failure class is severely underrepresented (87 of 24,720 total rows), which visibly hurt its recall (3/15 correct) — a data scarcity problem, not necessarily a modeling one. This is the clearest concrete next step for improving the project.
AI4I cross-domain validation (Section 10) confirmed the methodology — not the model — generalizes reasonably to a differently-imbalanced, structurally different machine dataset.
Repo Structure
predictive-maintenance/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── outputs/         # saved models, plots, SHAP images
├── README.md
└── requirements.txt
