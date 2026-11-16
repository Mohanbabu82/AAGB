# Where each of the 50 checklist items is addressed

Status key: **Done** = new code or experiment plus a manuscript change. **Partly** = addressed with a stated limitation. **Not possible** = stated as a limitation in the paper.

| # | Item | Status | Where / what was done |
| --- | --- | --- | --- |
| 1 | Mean-reduced gate gradient | Done | Input-level gate, Eq. (4); Proposition 1; gradient check (row-mean 0, input 0.30) |
| 2 | Feature-specific supervision | Done | Proposition 3; the input gate multiplies each feature by its own attention during training |
| 3 | In-sample augmentation | Done | 5-fold cross-fitting (Sec. III-E). In-sample attention lowered Stroke ROC-AUC significantly (Table VII) |
| 4 | Architectural ablation | Done | 14 variants plus 3 supplementary variants on 50 folds (Table VII, Fig. 3) |
| 5 | Run-to-run non-determinism | Done | Cause: the 600-s tuning limit. Removed; bitwise-identical predictions across processes on all datasets (Sec. V-I) |
| 6 | Justification of dual attention | Done | Ablation without sample or without feature attention (no effect); presented honestly as not justified empirically |
| 7 | Attention saturation | Done | Stroke: 50.6% of row-mean attention > 0.95; label–score agreement 0.95; mechanism explained after Proposition 1 (Sec. V-D) |
| 8 | Comparison of gates | Done | Row-mean, projection, input and softmax gates (Table VIII, Fig. 4; Propositions 1–3) |
| 9 | Single CV repetition | Done | 10 × 5 repeated stratified nested CV |
| 10 | Statistical power | Done | Corrected repeated k-fold test; minimum detectable d~z~ 1.04 (Holm 1.74) against 1.86 before (Sec. IV-E) |
| 11 | Confidence intervals | Done | Corrected 95% CIs on all main metrics, differences, ablation and factorial effects |
| 12 | CatBoost, TabNet, FT-Transformer | Done | All in Tables IV–VI. TabNet and FT-Transformer used default settings (stated) |
| 13 | TabPFN | Not possible | The Kaggle account has no internet access, so the package could not be installed (stated in Secs. IV-B and VI-C) |
| 14 | Component comparison | Done | Same as item 4, including random-pair and top-gain products and an untrained module |
| 15 | Explanation stability | Done | Across 50 folds and across 5 seeds with fixed partitions (Tables VIII, IX) |
| 16 | Independent validation | Not possible | No external cohort available (Limitation 2) |
| 17 | Heart Disease duplicate | Done | Audit found the duplicate (rows 163/164), ca = 4 and thal = 0 placeholders and an **inverted label** in heart.csv; the main analysis now uses the curated Cleveland file |
| 18 | Heart Disease categorical encoding | Done | cp, restecg, slope and thal one-hot encoded |
| 19 | Pima zeros | Done | Zeros treated as missing in the main analysis; zeros kept as a sensitivity analysis (Table XI) |
| 20 | CKD near-ceiling | Done | Documented; CKD not used to rank methods |
| 21 | CKD duplicate and integrity audit | Done | 0 duplicates, 158 complete records, missingness reported (Table III) |
| 22 | CKD feature exclusion | Done | Without sc, hemo, pcv and all three: ROC-AUC stays ≥ 0.997 (Table XI) |
| 23 | Stroke imbalance and thresholds | Done | Inner-fold threshold; oversampling and class weighting ± Platt (Table X for Stroke; Supplementary Table S1 for Pima and CKD); factorial with CIs (Supplementary Table S2) |
| 24 | Dataset representativeness | Done | Limitation 1 |
| 25 | Which components add value | Done | None added value; in-sample attention was harmful (Sec. V-C) |
| 26 | No superiority claims | Done | Title, abstract and conclusions framed as a negative/neutral result |
| 27 | Fixed 0.5 threshold | Done | τ\* chosen on inner out-of-fold predictions (Table V) |
| 28 | Calibration curves and metrics | Done | Brier, ECE, slope, intercept, calibration-in-the-large, curves (Tables IV, X, Supplementary Table S1; Fig. 2) |
| 29 | Calibration after resampling | Done | Platt recalibration on inner folds restores calibration (Table X, Supplementary Table S1) |
| 30 | Decision-curve analysis | Done | Fig. 6 |
| 31 | Operating points | Done | Sensitivity, specificity and precision at τ\* (Tables V, X, Supplementary Table S1) |
| 32 | Near-uniform attention | Done | The input gate separates variables; stability and faithfulness tested (Table VIII) |
| 33 | Glucose ranked low on Pima | Done | Explained by Proposition 1; the input gate ranks glucose first and passes the deletion test |
| 34 | SHAP fragmentation | Done | Aggregated to source variables (Sec. III-G, Table IX, Fig. 5) |
| 35 | Duplicate interaction names | Done | Canonical (p < q) pair order |
| 36 | Stability across folds and seeds | Done | Same as item 15 |
| 37 | Faithfulness test | Done | Deletion test against random, least-attended and SHAP rankings (Table VIII) |
| 38 | Causality | Done | Stated in Secs. III-G and V-E and Limitation 6 |
| 39 | Novelty over AttnBoost | Done | Mathematical comparison (Corollary) plus an experimental comparison with AttnBoost-style features, in-sample and cross-fitted (Table VII) |
| 40 | AGBoost and TabNet feature generation | Done | Sec. II-C and Table I |
| 41 | Positioning | Done | Contributions and Sec. VI-B |
| 42 | Extended theory | Done | Propositions 1–3 and the Corollary |
| 43 | No over-generalisation | Done | Conclusions limited to these four datasets |
| 44 | Code availability | Partly | Code is supplementary; **you** must create the public repository or upload it to the journal |
| 45 | Compute benchmark | Done | Table XII with hardware and software versions |
| 46 | Fold-level results | Partly | Checksummed logs included; **you** must download the fold-level CSV/NPZ files from the Kaggle Output tab |
| 47 | Reference verification | Done | Recent references checked online; volume and article numbers fixed for [Rajkomar], [van den Goorbergh] and [Van Calster]; seven references added; all 57 cited in first-cited order |
| 48 | AI acknowledgment wording | Done | Singular wording ("The author") with an accurate description of the AI's role |
| 49 | Figure readability | Done | All figures redrawn at 300 dpi, with variable names and axis units |
| 50 | TRIPOD+AI | Done | Supplementary Table S3 maps the items |
