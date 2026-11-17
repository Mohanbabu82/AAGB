| Dataset | Records (events, %) | Variables (encoded columns) | Audit findings and handling |
| :-- | :-- | :-- | :-- |
| Heart Disease (processed Cleveland) {{R:uci_heart}} | 297 (137, 46.1%) | 13 (22) | No duplicates or missing values. Nominal cp, restecg, slope and thal one-hot encoded. Redistributed heart.csv: 303 rows, 1 duplicate, 5 with ca = 4, 2 with thal = 0, label inverted (sensitivity analysis). |
| Pima Indians Diabetes {{R:pima}} | 768 (268, 34.9%) | 8 (8) | No duplicates. Zeros set to missing: Glucose 5, BloodPressure 35, SkinThickness 227, Insulin 374, BMI 11 (zeros kept: sensitivity analysis). |
| Chronic Kidney Disease {{R:uci_ckd}} | 400 (250, 62.5%) | 24 (24) | No duplicates. 158 complete records; up to 152 missing values per variable. sc, hemo and pcv excluded individually and jointly in sensitivity analyses. |
| Stroke Prediction {{R:stroke}} | 5,110 (249, 4.9%) | 10 (19) | No duplicates. 201 missing BMI values imputed within folds. One record with gender 'Other' retained. |
| Diabetes 130-US Hospitals {{R:uci_diab130}} | 10,000 sampled from 69,973 (897, 9.0%) | 30 (120) | First encounter per patient; encounters ending in death or hospice discharge removed (101,766 → 69,973 encounters, 9.0% readmitted within 30 days); stratified sample. No duplicates in the sample. |
| CDC Diabetes Health Indicators {{R:uci_cdc}} | 10,000 sampled from 253,680 (1,393, 13.9%) | 21 (21) | No missing values. 24,206 duplicate records in the full file (109 in the sample), kept as distributed; stratified sample. |
