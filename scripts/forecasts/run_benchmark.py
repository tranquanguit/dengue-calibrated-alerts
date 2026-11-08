"""Run the v2 (STARS) benchmark."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cv2 import run_target_fold  # noqa: E402
from src.features import build_horizon_frame, complete_grid  # noqa: E402
from src.power import POWER_VARS  # noqa: E402


def run_dataset(name, dcfg, cfg, n_jobs, years=None, out_root=None):
    out = Path(out_root or ROOT / "forecasts/benchmark") / name
    out.mkdir(parents=True, exist_ok=True)
    df = complete_grid(pd.read_csv(ROOT / "data/processed" / f"{name}_power_monthly.csv", parse_dates=["date"]))
    parts = out / "_parts"
    parts.mkdir(exist_ok=True)
    for target in dcfg["targets"]:
        if (parts / f"{target}_predictions.parquet").exists() and not years:
            print(f"[{name}] {target} cached", flush=True)
            continue
        preds, logs, cost, stab = [], [], [], []
        t0 = time.time()
        cross = [v for v in dcfg["cross_vars"] if v != target]
        frames, groups = {}, None
        for h in cfg["horizons"]:
            frames[h], groups = build_horizon_frame(df, target, h, cfg["ar_lags"], POWER_VARS, cross, cfg["exo_lags"])
        panel = df[["region", "date", *POWER_VARS]].copy()
        panel["ylog"] = np.log1p(df[target])
        for v in cross:
            panel[f"x_{v}"] = np.log1p(df[v])
        anom_vars = ["ylog", *POWER_VARS, *[f"x_{v}" for v in cross]]
        res = Parallel(n_jobs=n_jobs)(
            delayed(run_target_fold)(frames, groups, panel, anom_vars, y, cfg)
            for y in (years or dcfg["test_years"]))
        for p, lg, c, s in res:
            preds += [x.assign(target=target) for x in p]
            logs += [dict(x, target=target) for x in lg]
            cost += [dict(x, target=target) for x in c]
            if s is not None:
                stab.append(s.assign(target=target))
        pd.concat(preds).to_parquet(parts / f"{target}_predictions.parquet", index=False)
        pd.DataFrame(logs).to_csv(parts / f"{target}_selection_log.csv", index=False)
        pd.DataFrame(cost).to_csv(parts / f"{target}_cost.csv", index=False)
        pd.concat(stab).to_csv(parts / f"{target}_stability_tables.csv.gz", index=False)
        print(f"[{name}] {target} done in {time.time() - t0:.0f}s", flush=True)
    # one checkpoint per target (survives restarts); merged files for the summaries
    tg = dcfg["targets"]
    pd.concat([pd.read_parquet(parts / f"{t}_predictions.parquet") for t in tg]).to_parquet(out / "predictions.parquet", index=False)
    for kind in ("selection_log.csv", "cost.csv", "stability_tables.csv.gz"):
        pd.concat([pd.read_csv(parts / f"{t}_{kind}") for t in tg]).to_csv(out / kind, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--years", nargs="*", type=int, help="override test years (smoke tests)")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--out", help="output root (default: results/benchmark)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs/benchmark.yaml"))
    for name, d in cfg["datasets"].items():
        if a.datasets is None or name in a.datasets:
            run_dataset(name, d, cfg["common"], a.n_jobs, a.years, a.out)
