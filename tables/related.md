| Method | Where attention or gating acts | How the gate is supervised | Tree model and inputs | Out-of-fold features for training rows |
| :-- | :-- | :-- | :-- | :-- |
| TabNet {{R:tabnet}} | Input features (sequential sparse masks) | End-to-end task loss; mask multiplies inputs | None (replaces trees) | Not applicable |
| L2X, INVASE, STG {{R:l2x, invase, stg}} | Input features (per sample or global) | Task loss through the masked input | None | Not applicable |
| AttnBoost {{R:attnboost}} | Hidden units of a one-layer network (*k* = 128) | BCE through the gated hidden layer | XGBoost on [**x**, hidden or attention representation] | Not stated |
| Xia et al. {{R:xia}} | TabNet features and reweighted inputs | TabNet task loss | Stacked XGBoost / LightGBM | Not stated |
| AGBoost {{R:agboost}} | Boosting iterations (base-learner weights) | Attention over iterations | Gradient boosting machine | Not applicable (no new features) |
| MHASRF {{R:amalina}} | Inside a soft random forest | End-to-end | Soft random forest | Not applicable |
| AAGB-RM (original) | Input-feature matrix *A*~f~ and scalar *a*~s~ | BCE through row-mean gate (no feature-specific gradient) | XGBoost on [*X*, *X*~w~, *X*~int~] | No (in-sample) |
| **AAGB-IG (this work)** | Input-feature matrix *A*~f~ and scalar *a*~s~ | BCE through input-level gate (feature-specific) | XGBoost on [*X*, *X*~w~, *X*~int~] | Yes (5-fold cross-fitting) |
