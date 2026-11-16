| Dataset | Records (events, %) | Variables (encoded columns) | Audit findings and handling |
| :-- | :-- | :-- | :-- |
| Heart Disease (processed Cleveland) {{R:uci_heart}} | 297 (137, 46.1%) | 13 (22) | No duplicates or missing values. Nominal cp, restecg, slope and thal one-hot encoded. Redistributed heart.csv used before: 303 rows, 1 duplicate, 5 with ca = 4, 2 with thal = 0, label inverted (sensitivity analysis). |
| Pima Indians Diabetes {{R:pima}} | 768 (268, 34.9%) | 8 (8) | No duplicates. Zeros set to missing: Glucose 5, BloodPressure 35, SkinThickness 227, Insulin 374, BMI 11 (zeros kept: sensitivity analysis). |
| Chronic Kidney Disease {{R:uci_ckd}} | 400 (250, 62.5%) | 24 (24) | No duplicates. 158 complete records; up to 152 missing values per variable. sc, hemo and pcv excluded individually and jointly in sensitivity analyses. |
| Stroke Prediction {{R:stroke}} | 5,110 (249, 4.9%) | 10 (19) | No duplicates. 201 missing BMI values imputed within folds. One record with gender 'Other' retained. |
