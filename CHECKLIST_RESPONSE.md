# Where each of the 50 checklist items is addressed

Status key: **Done** = new code or experiment plus a manuscript change. **Partly** = addressed with a stated limitation. **Not possible** = stated as a limitation in the paper.

| # | Item | Status | Where / what was done |
| --- | --- | --- | --- |
| 1 | Mean-reduced gate gradient | Done | Input-level gate, Eq. (4); Observation 1; gradient check (row-mean 0, input 0.30) |
| 2 | Feature-specific supervision | Done | Proposition (input-level gate); the input gate multiplies each feature by its own attention during training |
| 3 | In-sample augmentation | Done | 5-fold cross-fitting (Sec. III-E). In-sample attention lowered Stroke ROC-AUC significantly (Table VII) |
| 4 | Architectural ablation | Done | 14 variants plus 3 supplementary variants on 50 folds (Table VII, Fig. 3) |
| 5 | Run-to-run non-determinism | Done | Cause: the 600-s tuning limit. Removed; bitwise-identical predictions across processes on all datasets (Sec. V-I) |
| 6 | Justification of dual attention | Done | Ablation without sample or without feature attention (no effect); presented honestly as not justified empirically |
| 7 | Attention saturation | Done | Stroke: 50.6% of row-mean attention > 0.95; label–score agreement 0.95; mechanism explained after Observation 1 (Sec. V-D) |
| 8 | Comparison of gates | Done | Row-mean, projection, input and softmax gates (Table VIII, Supplementary Fig. S3; Observations 1–2 and the Proposition) |
| 9 | Single CV repetition | Done | 10 × 5 repeated stratified nested CV |
| 10 | Statistical power | Done | Corrected repeated k-fold test; minimum detectable d~z~ 1.04 (Holm 1.74) against 1.86 before (Sec. IV-E) |
| 11 | Confidence intervals | Done | Corrected 95% CIs on all main metrics, differences, ablation and factorial effects |
| 12 | CatBoost, TabNet, FT-Transformer | Done | All in Tables IV–VI. TabNet and FT-Transformer used default settings (stated) |
| 13 | TabPFN | Done | TabPFN v2 (open v2 weights) on a Kaggle GPU, same folds, all six datasets (Sec. IV-J, Tables X–XI) |
| 14 | Component comparison | Done | Same as item 4, including random-pair and top-gain products and an untrained module |
| 15 | Explanation stability | Done | Across 50 folds and across 5 seeds with fixed partitions (Tables VIII, IX) |
| 16 | Independent validation | Not possible | No external cohort available (Limitation 2) |
| 17 | Heart Disease duplicate | Done | Audit found the duplicate (rows 163/164), ca = 4 and thal = 0 placeholders and an **inverted label** in heart.csv; the main analysis now uses the curated Cleveland file |
| 18 | Heart Disease categorical encoding | Done | cp, restecg, slope and thal one-hot encoded |
| 19 | Pima zeros | Done | Zeros treated as missing in the main analysis; zeros kept as a sensitivity analysis (Supplementary Table S4) |
| 20 | CKD near-ceiling | Done | Documented; CKD not used to rank methods |
| 21 | CKD duplicate and integrity audit | Done | 0 duplicates, 158 complete records, missingness reported (Table III) |
| 22 | CKD feature exclusion | Done | Without sc, hemo, pcv and all three: ROC-AUC stays ≥ 0.997 (Supplementary Table S4) |
| 23 | Stroke imbalance and thresholds | Done | Inner-fold threshold; oversampling and class weighting ± Platt (Supplementary Table S1); factorial with CIs (Supplementary Table S2) |
| 24 | Dataset representativeness | Done | Limitation 1 |
| 25 | Which components add value | Done | None added value; in-sample attention was harmful (Sec. V-C) |
| 26 | No superiority claims | Done | Title, abstract and conclusions framed as a negative/neutral result |
| 27 | Fixed 0.5 threshold | Done | τ\* chosen on inner out-of-fold predictions (Table V) |
| 28 | Calibration curves and metrics | Done | Brier, ECE, slope, intercept, calibration-in-the-large, curves (Tables IV, Supplementary Table S1; Fig. 2) |
| 29 | Calibration after resampling | Done | Platt recalibration on inner folds restores calibration (Supplementary Table S1) |
| 30 | Decision-curve analysis | Done | Supplementary Fig. S1 |
| 31 | Operating points | Done | Sensitivity, specificity and precision at τ\* (Table V, Supplementary Table S1) |
| 32 | Near-uniform attention | Done | The input gate separates variables; stability and faithfulness tested (Table VIII) |
| 33 | Glucose ranked low on Pima | Done | Explained by Observation 1; the input gate ranks glucose first and passes the deletion test |
| 34 | SHAP fragmentation | Done | Aggregated to source variables (Sec. III-G, Table IX, Supplementary Fig. S2) |
| 35 | Duplicate interaction names | Done | Canonical (p < q) pair order |
| 36 | Stability across folds and seeds | Done | Same as item 15 |
| 37 | Faithfulness test | Done | Deletion test against random, least-attended and SHAP rankings (Table VIII) |
| 38 | Causality | Done | Stated in Secs. III-G and V-E and Limitation 6 |
| 39 | Novelty over AttnBoost | Done | Mathematical comparison (Corollary) plus an experimental comparison with AttnBoost-style features, in-sample and cross-fitted (Table VII) |
| 40 | AGBoost and TabNet feature generation | Done | Sec. II-C and Table I |
| 41 | Positioning | Done | Contributions and Sec. VI-B |
| 42 | Extended theory | Done | Observations 1–2 and the Proposition and the Corollary |
| 43 | No over-generalisation | Done | Conclusions limited to these four datasets |
| 44 | Code availability | Partly | Code is supplementary; **you** must create the public repository or upload it to the journal |
| 45 | Compute benchmark | Done | Supplementary Table S5 with hardware and software versions |
| 46 | Fold-level results | Partly | Checksummed logs included; **you** must download the fold-level CSV/NPZ files from the Kaggle Output tab |
| 47 | Reference verification | Done | Recent references checked online; volume and article numbers fixed for [Rajkomar], [van den Goorbergh] and [Van Calster]; seven references added; all cited in first-cited order |
| 48 | AI acknowledgment wording | Done | Singular wording ("The author") with an accurate description of the AI's role |
| 49 | Figure readability | Done | All figures redrawn at 300 dpi, with variable names and axis units |
| 50 | TRIPOD+AI | Done | Supplementary Table S3 maps the items |

## Second review (additional items)

| Item | Status | Where |
|---|---|---|
| Remove revision-history wording ("first submission", "earlier version") | Done | Whole manuscript; history goes in the cover letter |
| Reframe the row-mean design as a natural design rather than "the original method" | Done | Sec. I, III |
| Propositions 1–2 called elementary | Done | Now Observations 1–2; the Proposition and Corollary carry the theory (Sec. III-D) |
| heart.csv inversion presented as a data-quality finding | Done | Sec. IV-A |
| TabPFN | Done | Sec. IV-J, Tables X–XI |
| Larger datasets | Done | Diabetes 130-US Hospitals and CDC Diabetes Health Indicators, 10,000-record stratified samples, 2 × 5 folds (Sec. IV-J, Table XI) |
| Tuned TabNet and FT-Transformer, FT-Transformer on Stroke paired | Done | Same 15-trial TPE budget, single inner hold-out, 2 × 5 folds, paired with XGBoost and AAGB-IG (Table X) |
| Leakage test on a published style of hybrid | Done | TabNet-feature stacking, in-sample vs cross-fitted; no measurable leakage (Supplementary Table S6) |
| AI disclosure with sections and citation | Done | Acknowledgment; reference to the AI system |
| Software versions checked against run logs | Done | CPU logs report Python 3.13.18; TabPFN GPU run Python 3.13.16 (Sec. IV-I) |
| Zenodo DOI for the code | **Author** | Archive the GitHub release on Zenodo and add the DOI to the code-availability statement |
| DOI line year | Done | Template line now reads ACCESS.2026; footers VOLUME XX, 2026 |

## Third review

| Item | Status | Where |
|---|---|---|
| Abstract describes the extended study (TabPFN v2, tuned deep models, two larger datasets, TabNet audit); "a row-mean design"; 235 words, leaving room for revision edits | Done | Abstract |
| Introduction no longer claims published hybrids use the row-mean design or in-sample features | Done | Sec. I, third paragraph (consistent with Table I "Not stated") |
| AI system cited in each section (Sections I–VII, first paragraph) as a standalone note, "(Text in this section was drafted with AI assistance [8].)", not attached to any factual claim | Done | Reference [8] (the AI reference moved from [61] to [8] because references are numbered in order of first citation) |
| DOI line year 2026 | Done | Title page |
| Python and library versions checked | Done | Match the @@VER lines of the v7 logs (CPU 3.13.18, GPU 3.13.16) |
| Zenodo DOI | **Author** | Requires the author's Zenodo account |
| Contribution 3 grammar ("The evaluation also includes") | Done | Sec. I |
| Reason for 10,000-record samples stated | Done | Sec. IV-J (TabPFN v2 training-set size; CPU cost of cross-fitting with tuning) |
| Leakage claim scoped: significant on one of six datasets, same direction on the two larger ones | Done | Abstract, contribution 1, Conclusion |
