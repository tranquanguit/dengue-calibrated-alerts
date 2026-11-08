"""Pilot: zero-shot Chronos-2 on the v2 benchmark forecast origins (no training, no fine-tuning).

Weights: s3://autogluon/chronos-2 (Apache-2.0), downloaded to --model-dir.
For every forecast origin t used by the rolling-origin benchmark, the model receives only data
observed up to t (log1p cases; optionally past climate as covariates) and forecasts 6 months.
Variants
  Chronos2           : univariate, each region separately
  Chronos2-cross     : univariate with cross_learning=True (regions of the same origin share
                       information inside the batch -> zero-shot "pooled" model)
  Chronos2+climate   : past-only climate covariates (13 NASA POWER variables)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.features import complete_grid  # noqa: E402
from src.power import POWER_VARS  # noqa: E402

H = 6
QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def origins_needed(test_years):
    t = []
    for Y in test_years:
        for m in pd.date_range(f"{Y}-01-01", f"{Y}-12-01", freq="MS"):
            t += [m - pd.DateOffset(months=h) for h in range(1, H + 1)]
    return sorted(set(t))


def run(pipe, name, dcfg, variants, out_dir):
    df = complete_grid(pd.read_csv(ROOT / "data/processed" / f"{name}_power_monthly.csv", parse_dates=["date"]))
    df = df.sort_values(["region", "date"])
    regions = df.region.unique()
    rows = []
    for target in dcfg["targets"]:
        piv = df.pivot(index="date", columns="region", values=target)
        ylog = np.log1p(piv)
        clim = {v: df.pivot(index="date", columns="region", values=v) for v in POWER_VARS}
        for t in origins_needed(dcfg["test_years"]):
            if t not in ylog.index:
                continue
            hist = ylog.loc[:t]
            items, keep = [], []
            for r in regions:
                s = hist[r].to_numpy(float)
                first = np.flatnonzero(~np.isnan(s))
                if len(first) == 0 or len(s) - first[0] < 24:
                    continue
                s = s[first[0]:]
                keep.append(r)
                items.append((r, s, {v: clim[v].loc[:t, r].to_numpy(float)[first[0]:] for v in POWER_VARS}))
            if not items:
                continue
            for var in variants:
                if var == "Chronos2+climate":
                    inp = [{"target": s, "past_covariates": {k: np.nan_to_num(v, nan=np.nanmean(v)) for k, v in cov.items()}}
                           for _, s, cov in items]
                else:
                    inp = [s for _, s, _ in items]
                q, _ = pipe.predict_quantiles(inp, prediction_length=H, quantile_levels=QUANTILES,
                                              cross_learning=(var == "Chronos2-cross"), batch_size=512)
                for (r, _, _), qq in zip(items, q):
                    qq = qq[0].numpy()  # (H, Q)
                    for h in range(1, H + 1):
                        td = t + pd.DateOffset(months=h)
                        if td.year not in dcfg["test_years"]:
                            continue
                        rows.append({"target": target, "region": r, "origin": t, "target_date": td,
                                     "horizon": h, "test_year": td.year, "model": var,
                                     "y_pred": float(np.clip(np.expm1(qq[h - 1, QUANTILES.index(0.5)]), 0, None)),
                                     **{f"q{int(a * 100):02d}": float(np.expm1(qq[h - 1, i])) for i, a in enumerate(QUANTILES)}})
        print(f"[{name}] {target} done", flush=True)
    out = pd.DataFrame(rows)
    truth = df.melt(id_vars=["region", "date"], value_vars=dcfg["targets"], var_name="target", value_name="y_true")
    out = out.merge(truth.rename(columns={"date": "target_date"}), on=["target", "region", "target_date"], how="left")
    out.to_parquet(out_dir / f"{name}.parquet", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/chronos-2")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--years", nargs="*", type=int, help="override test years")
    ap.add_argument("--out", help="output dir (default: results/chronos2)")
    ap.add_argument("--variants", nargs="*", default=["Chronos2", "Chronos2-cross", "Chronos2+climate"])
    a = ap.parse_args()
    from chronos import BaseChronosPipeline

    torch.set_num_threads(4)
    pipe = BaseChronosPipeline.from_pretrained(a.model_dir, device_map="cpu", torch_dtype=torch.float32)
    cfg = yaml.safe_load(open(ROOT / "configs/benchmark.yaml"))
    out_dir = Path(a.out) if a.out else ROOT / "forecasts/chronos2"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, d in cfg["datasets"].items():
        if a.datasets and name not in a.datasets:
            continue
        if (out_dir / f"{name}.parquet").exists():
            print(f"[{name}] cached")
            continue
        t0 = time.time()
        run(pipe, name, {**d, "test_years": a.years or d["test_years"]}, a.variants, out_dir)
        print(f"[{name}] {time.time() - t0:.0f}s", flush=True)
