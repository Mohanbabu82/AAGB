# AAGB revision – supplementary material

## Contents

| Folder | What it holds |
| --- | --- |
| `code/aagb2.py` | The full revised pipeline: data loading and audit, preprocessing, three attention gates, cross-fitting, TabNet and FT-Transformer implementations, nested tuning, the main benchmark, sensitivity analyses, ablation, factorial design, timing and the determinism check. |
| `code/aagb2x.py` | Supplementary experiments: AttnBoost-style hidden-unit features, the row-normalised (softmax) gate and seed-only stability. |
| `results/v*_log_lines.txt` | Every reported value as a CRC-32-checksummed record (820 records, all verified). v2 = Heart Disease + Pima, v3 = CKD, v4 = Stroke (all models except FT-Transformer), v5 = Stroke FT-Transformer, v6 = supplementary experiments. |
| `results/numbers.json` | Every number quoted in the manuscript, computed from the logs. |
| `analysis/` | `make_final.py` rebuilds all tables and Figs 2–6 from the logs; `lib.py` holds the statistics (corrected repeated k-fold t-test, Bayesian correlated t-test, Holm); `fig1.py` draws Fig. 1. |
| `Supplementary_Tables.docx/.pdf` | Supplementary Tables S1 (imbalance strategies on Pima, CKD and Stroke), S2 (factorial analysis) and S3 (TRIPOD+AI map) cited in the manuscript. |
| `tables/`, `figures/` | Generated tables (Markdown) and 300-dpi figures, numbered as in the manuscript. |

## How the experiments were run

The runs used Kaggle CPU sessions (AMD EPYC 7B12, 4 vCPUs, 31.3 GB RAM; no internet). The software versions were Python 3.13.18, NumPy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1, XGBoost 3.4.1, LightGBM 4.6.0, CatBoost 1.2.10, PyTorch 2.11.0+cpu, Optuna 5.0.0 and imbalanced-learn 0.14.2.

The notebook needs these input files attached:

- `heart_cleveland_upload.csv` (processed Cleveland data)
- `heart.csv` (the redistributed file, for the sensitivity analysis)
- `diabetes.csv`
- `kidney_disease.csv`
- `healthcare-dataset-stroke-data.csv`

The calls used for each run were:

```python
import aagb2
aagb2.run(['heart', 'pima'], dict(R=10, trials=15, n_jobs=4))                      # v2
aagb2.run(['ckd'], dict(R=10, trials=15, n_jobs=4))                                # v3
aagb2.run(['stroke'], dict(R=10, trials=15, n_jobs=4,
          models=['rf','xgb','lgbm','cat','mlp','tabnet','aagb_rm','aagb_ig']))    # v4
aagb2.run(['stroke'], dict(R=10, trials=15, n_jobs=4, models=['ftt'], stages=['main']))  # v5
import aagb2x
aagb2x.run_extra(['heart', 'pima', 'ckd', 'stroke'], dict(R=10, trials=15, n_jobs=4))  # v6
```

Every run logged the same SHA-256 hash of `aagb2.py` (`ff56ca2bb49318015e0f33411c383b3a4268b45e8eae45b904a2dfd421db1d11`).

Run times on Kaggle CPU were 2.1–2.7 h for each of v2–v5 and 0.4 h for v6.

## Fold-level files

Each Kaggle run also wrote the following files to `/kaggle/working/<dataset>/`:

- `fold_metrics_main.csv` and `test_predictions_main.npz`
- `fold_metrics_<variant>.csv` (sensitivity analyses)
- `fold_metrics_ablation.csv` and `fold_metrics_factorial.csv`
- `fold_metrics_extra.csv` (v6)
- `attention_by_variable_*.npy` and `shap_by_variable_*.npy`

These files are in the Output tab of each notebook version. They should be downloaded and added to this package before submission, in a folder such as `fold_level/`.

## Re-running the analysis

To rebuild the tables and figures, put the log files at `results/vN/log_lines.txt`, or edit `ROOT` and the glob in `make_final.py`. Then run `python make_final.py <output folder>`.
