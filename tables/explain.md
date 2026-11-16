| Dataset | Top-3, XGBoost SHAP | Top-3, AAGB-IG SHAP (aggregated) | Top-3, AAGB-IG attention | SHAP rank stability XGBoost (folds / seeds) | SHAP rank stability AAGB-IG (folds / seeds) | ρ(SHAP AAGB-IG, SHAP XGBoost) | ρ(attention, SHAP AAGB-IG) | SHAP share *X* / *X*~w~ / *X*~int~ |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Heart Disease | ca, thal, cp | ca, cp, thal | ca, oldpeak, exang | 0.82 / 0.97 | 0.74 / 0.79 | 0.94 | 0.53 | 24 / 66 / 11% |
| Pima | Glucose, BMI, Age | Glucose, BMI, Age | Glucose, BMI, Age | 0.91 / 0.95 | 0.89 / 0.88 | 0.98 | 0.93 | 33 / 46 / 21% |
| CKD | hemo, sg, sc | hemo, sg, sc | hemo, sg, pcv | 0.94 / 0.98 | 0.84 / 0.84 | 0.90 | 0.45 | 44 / 49 / 8% |
| Stroke | age, bmi, avg_glucose_level | age, bmi, hypertension | ever_married, smoking_status, Residence_type | 0.83 / 0.96 | 0.65 / 0.73 | 0.85 | -0.77 | 42 / 55 / 3% |
