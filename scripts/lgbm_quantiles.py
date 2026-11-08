"""In-fold quantile LightGBM (same rolling-origin folds and features as the v2 benchmark).

For every (task, horizon, test year) one LightGBM per quantile level is trained on
log1p(cases) with the pinball loss, using AR features + raw climate lags; the predicted
quantiles are sorted (no crossing) and stored on the count scale.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.cv2 import BASE, Y_ANOM  # noqa: E402
from src.features import add_seasonal_profile, build_horizon_frame, complete_grid  # noqa: E402
from src.power import POWER_VARS  # noqa: E402
from src.stars import anomaly_windows  # noqa: E402

LEVELS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, max_bin=63, force_col_wise=True)
OUT = ROOT / "results/lgbm_quantiles"


def fold(frames, groups, panel, test_year):
    A = anomaly_windows(panel, ["ylog"], test_year)
    rows = []
    for h, fr in frames.items():
        f = fr.merge(A, on=["region", "origin"], how="left")
        mask = (f.target_date.dt.year < test_year).to_numpy()
        f = add_seasonal_profile(f, mask)
        tr, te = f[mask], f[(f.target_date.dt.year == test_year).to_numpy()]
        if te.empty or len(tr) < 500:
            continue
        feats = groups["ar"] + BASE + Y_ANOM + groups["climate"]
        Q = np.column_stack([
            lgb.LGBMRegressor(objective="quantile", alpha=a, **PARAMS, random_state=42, n_jobs=1, verbose=-1)
            .fit(tr[feats], tr.y, categorical_feature=["region_code"]).predict(te[feats]) for a in LEVELS])
        Q = np.sort(Q, axis=1)
        out = te[["region", "origin", "target_date"]].assign(horizon=h, test_year=test_year)
        for i, a in enumerate(LEVELS):
            out[f"q{int(a * 100):02d}"] = np.clip(np.expm1(Q[:, i]), 0, None)
        rows.append(out)
    return pd.concat(rows) if rows else pd.DataFrame()


def main(names=None):
    cfg = yaml.safe_load(open(ROOT / "configs/benchmark.yaml"))
    c = cfg["common"]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, d in cfg["datasets"].items():
        if (names and name not in names) or (OUT / f"{name}.parquet").exists():
            continue
        t0 = time.time()
        df = complete_grid(pd.read_csv(ROOT / "data/processed" / f"{name}_power_monthly.csv", parse_dates=["date"]))
        res = []
        for target in d["targets"]:
            frames, groups = {}, None
            for h in c["horizons"]:
                frames[h], groups = build_horizon_frame(df, target, h, c["ar_lags"], POWER_VARS, [], c["exo_lags"])
            panel = df[["region", "date"]].assign(ylog=np.log1p(df[target]))
            parts = Parallel(n_jobs=4)(delayed(fold)(frames, groups, panel, y) for y in d["test_years"])
            res.append(pd.concat(parts).assign(target=target))
        pd.concat(res).to_parquet(OUT / f"{name}.parquet", index=False)
        print(f"[{name}] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
