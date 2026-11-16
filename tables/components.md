| \# | Component | Definition | Used at inference |
| :-: | :-- | :-- | :-: |
| 1 | Shared encoder *f*~θ~ | 2 × (Linear → BatchNorm → ReLU → Dropout), width *m* = 64 | Yes |
| 2 | Feature attention | *A*~f~ ∈ (0,1)^*n*×*d*^, Eq. (1) | Yes |
| 3 | Sample attention | **a**~s~ ∈ (0,1)^*n*^, Eq. (1) | Yes |
| 4 | Gate and auxiliary predictor | Input-level gate, Eq. (4) (AAGB-IG); row-mean gate, Eq. (2) (AAGB-RM) | No |
| 5 | Cross-fitting | 5-fold out-of-fold attention for training rows (AAGB-IG) | Training only |
| 6 | Attention-weighted features | *X*~w~ = *X* ⊙ *A*~f~ ⊙ (**a**~s~**1**~d~^⊤^), Eq. (6) | Yes |
| 7 | Interaction features | Products of all 10 pairs among the 5 most attended columns, canonical order | Yes |
| 8 | Classifier | XGBoost on *X*~aug~ = [*X* ∥ *X*~w~ ∥ *X*~int~] (2*d* + 10 columns), nested tuning | Yes |
| 9 | Explanations | TreeSHAP aggregated to source variables; mean test-row attention | Reporting only |
