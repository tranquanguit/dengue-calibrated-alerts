# Guaranteed versus Experienced False-Alert Rates: A Pre-registered Evaluation of Conformal Selection for Provincial Dengue Outbreak Bulletins

Code, data and results for the paper (IEEE conference version).
This repository is self-contained: open data → forecasts → alert procedures → evaluation → manuscript.

```
data/          open dataset (sources, licences and citations in data/README.md)
pipeline/      scripts that rebuild data/ from the public sources
src/           shared library: loaders, climate, features, rolling-origin CV, evaluation
configs/       benchmark.yaml — countries, targets, folds
forecasts/     probabilistic forecasts consumed by the alert study (produced by scripts/forecasts/)
scripts/       forecasts/ (upstream forecasting) + alert procedures and analyses (tables below)
tests/         unit tests for the alert core (pytest)
results/       all outputs used by the manuscript
docs/          pre-registrations (v1, v2, v3, v3.1 addendum) and pilot notes
manuscript/    IEEE LaTeX source and PDF (ieee_isci2027/)
```

## 1. Data

```bash
pip install -r requirements.txt
python pipeline/fetch_data.py && python pipeline/fetch_power.py && python pipeline/build_datasets.py
```

The processed tables in `data/processed/` are committed, so the later steps run without re-downloading.

## 2. Forecasts (inputs to the alert study)

| Forecast | Script | Output |
|---|---|---|
| LightGBM pooled/local grid, 7 quantiles, h = 1–3 | `scripts/forecasts/unified_grid.py` | `forecasts/unified_grid/blockA.parquet`, `blockB.parquet` |
| Chronos-2 zero-shot quantiles (weights `s3://autogluon/chronos-2`) | `scripts/forecasts/chronos2_zero_shot.py` | `forecasts/chronos2/`, `forecasts/chronos2_grid/blockB/` |
| Rolling-origin benchmark predictions (pilot studies only) | `scripts/forecasts/run_benchmark.py` | `forecasts/benchmark/*/predictions.parquet` |
| In-fold quantile LightGBM (development set) | `scripts/lgbm_quantiles.py` | `results/lgbm_quantiles/` |

The forecasts are committed; re-running them is optional (Chronos-2 needs the model weights; the grid takes hours).

## 3. Alert procedures and evaluation

Core definitions (endemic-channel event, standardised-margin score, conformal p-values, BH selection,
Mondrian CS, fixed-threshold rules, CRC scan, the three FDR notions and the gap decomposition) are in
`scripts/alerts_core.py` and tested in `tests/` (`pytest tests`).

| Stage | Script | Output |
|---|---|---|
| Pilots (exploratory) | `conformal_intervals.py`, `conformal_fdr.py`, `conformal_fdr_adaptive.py`, `quantile_alerts.py`, `quantile_alerts_curve.py`, `block_fdr_alerts.py` | `results/*.csv` |
| Development set (TH, ID, KH, LA, MY) | `dev_alerts.py`, `dev_r4.py`, `dev_r4_gapdiff.py`, `monitored_precision.py` | `results/dev/` |
| **Confirmation set** (PH, VN, LK, BR; run once after the pre-registration was locked) | `confirm.py` | `results/confirmation/` |
| Frozen decision rules → hypotheses H1–H5 | `evaluate_hypotheses.py` | `results/confirmation/hypotheses.md`, `.csv` |
| Disclosed fixes and sensitivity analyses | `confirm_crc_fix.py`, `confirm_ens_arith.py` | `results/confirmation/` |
| Additional exploratory analyses | `revision_r1.py`, `revision_r2.py` | `results/confirmation/revision_r1/` |
| Simulation | `simulation.py` | `results/simulation/` |
| Figures (IEEE version) | `figures_ieee.py` | `manuscript/ieee_isci2027/figures/` |

Run from the repository root, e.g. `python scripts/confirm.py && python scripts/evaluate_hypotheses.py`.
The manuscript compiles with `cd manuscript/ieee_isci2027 && pdflatex main && bibtex main && pdflatex main && pdflatex main`.

Note: the pre-registration documents in `docs/` are kept verbatim as frozen; paths in them refer to the
original project layout (`paper2_calibrated_alerts/<x>` = `<x>` here, `paper1_pooled_forecasting/results/<x>` = `forecasts/<x>`).

## Licence and citation

Code: MIT (see `LICENSE`). Data keep the licences of their original sources (see `data/README.md`);
please cite the original data producers. Citation metadata: `CITATION.cff`.
