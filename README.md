# AAGB revision – supplementary material

## Contents

| Folder | What it holds |
| --- | --- |
| `code/aagb2.py` | The full revised pipeline: data loading and audit, preprocessing, three attention gates, cross-fitting, TabNet and FT-Transformer implementations, nested tuning, the main benchmark, sensitivity analyses, ablation, factorial design, timing and the determinism check. |
| `code/aagb2x.py` | Supplementary experiments: AttnBoost-style hidden-unit features, the row-normalized (softmax) gate and seed-only stability. |
| `code/aagb3.py` | Additional experiments (Section IV-J): TabPFN v2, TabNet and FT-Transformer tuned with the same 15-trial budget, the TabNet-feature stacking leakage audit, and the Diabetes 130-US and CDC Diabetes Health Indicators datasets (10,000-record stratified samples). |
| `results/v*_log_lines.txt` | Every reported value as a CRC-32-checksummed record (820 records, all verified). v2 = Heart Disease + Pima, v3 = CKD, v4 = Stroke (all models except FT-Transformer), v5 = Stroke FT-Transformer, v6 = supplementary experiments. |
| `results/v7_*_log_lines.txt` | Checksummed records of the additional experiments (597 records, all verified): small datasets on CPU, larger datasets on CPU, TabPFN v2 on a GPU. |
| `results/numbers.json`, `results/numbers_v3.json` | Every number quoted in the manuscript, computed from the logs. |
| `analysis/` | `make_final.py` rebuilds the main tables and figures from the v2–v6 logs; `make_v3.py` builds Tables X–XI and Supplementary Table S6 from the v7 logs; `lib.py` holds the statistics (corrected repeated k-fold t-test, Bayesian correlated t-test, Holm); `fig1.py` draws Fig. 1. |
| `Supplementary_Material.docx/.pdf` | Supplementary Tables S1 (imbalance strategies), S2 (factorial analysis), S3 (TRIPOD+AI map), S4 (dataset sensitivity analyses), S5 (timing), S6 (TabNet-feature stacking audit) and Supplementary Figs S1 (decision curves), S2 (aggregated SHAP) and S3 (attention per gate). |
| `tables/`, `figures/` | Generated tables (Markdown) and 300-dpi figures, numbered as in the manuscript. |

## How the experiments were run

The main runs (v2–v6) used Kaggle CPU sessions (AMD EPYC 7B12, 4 vCPUs, 31.3 GB RAM; no internet). The software versions were Python 3.13.18, NumPy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1, XGBoost 3.4.1, LightGBM 4.6.0, CatBoost 1.2.10, PyTorch 2.11.0+cpu, Optuna 5.0.0 and imbalanced-learn 0.14.2.

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

## Additional experiments (v7)

The v7 runs downloaded `aagb2.py` and `aagb3.py` from this repository at commit `227e9db37fb254560e6dc7b13f12624e60211ae0` (internet on; the larger datasets are fetched from the UCI Machine Learning Repository with `ucimlrepo`). TabPFN (`tabpfn` 10.0.0, open v2 classifier weights) ran on a Kaggle GPU session (Tesla T4, Python 3.13.16).

```python
import aagb2 as A2, aagb3 as A3
C = dict(R=10, R_large=2, trials=15, n_jobs=4, quick=False, leak=True, ctx_small=10000, ctx_large=10000,
         deep_epochs=dict(tabnet=(100, 15), ftt=(100, 10)))
A3.run3([(d, r, f) for d in ('stroke', 'ckd', 'pima', 'heart') for r in range(2) for f in range(5)], C)      # small, CPU (5.4 h)
A3.run3([(d, r, f) for d in ('diab130', 'cdc') for r in range(2) for f in range(5)], C)                       # larger, CPU (0.4 h)
A3.run3([(d, r, f) for d in ('heart', 'pima', 'ckd', 'stroke', 'diab130', 'cdc') for r in range(2) for f in range(5)],
        dict(C, n_jobs=1, leak=False, pfn_only=True))                                                         # TabPFN, GPU (0.5 h)
```

To rebuild the v7 tables: `python analysis/make_v3.py <out> results/v7_*_log_lines.txt`.

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
