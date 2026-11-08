"""Pilot 4: outbreak alerts from predictive quantiles (LightGBM quantile models, Chronos-2,
and their quantile average), with online, leak-free calibration.

P(event) is read off the predictive distribution: the quantile function (7 levels, linear
interpolation, flat tails at 0.01 / 0.99) is inverted at the endemic threshold.

Calibration variants (all use only cells whose target month is already observed):
  raw        : alert if P(event) >= 0.5 (no calibration)
  fdr-task   : cut-off on P(event) chosen per task so that (false+1)/(alerts+1) <= q on past cells
  fdr-pooled : same, but past cells of ALL dengue countries are pooled (more feedback)
  iso-pooled : P(event) first recalibrated by online isotonic regression on pooled past cells,
               then the pooled FDR cut-off
Discrimination (independent of any cut-off): average precision (PR-AUC) of P(event).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conformal_intervals import thresholds  # noqa: E402

OUT = ROOT / "results"
LEVELS = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
QCOLS = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]
DATASETS = ["thailand", "indonesia", "cambodia", "laos", "malaysia", "philippines", "vietnam", "srilanka", "brazil"]
KEY = ["target", "region", "origin", "target_date", "horizon", "test_year"]
Q = 0.3
MIN_CASES = 5
MIN_PAST = 200
TAU = np.round(np.arange(0.02, 1.0001, 0.02), 3)


def exceed_prob(qlog: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """P(log1p Y > thr) from sorted log-quantiles (rows) by inverting the piecewise-linear QF."""
    lv = np.concatenate([[0.01], LEVELS, [0.99]])
    out = np.empty(len(thr))
    for i in range(len(thr)):
        qs = np.concatenate([[qlog[i, 0]], qlog[i], [qlog[i, -1]]])
        out[i] = 1.0 - np.interp(thr[i], qs, lv, left=0.0, right=1.0)
    return out


def load(name: str) -> pd.DataFrame:
    lq = pd.read_parquet(OUT / "lgbm_quantiles" / f"{name}.parquet")
    ch = pd.read_parquet(ROOT / "forecasts/chronos2" / f"{name}.parquet")  # produced by paper 1
    ch = ch[ch.model == "Chronos2-cross"][KEY + QCOLS + ["y_true"]]
    d = lq.merge(ch, on=KEY, suffixes=("_lgb", "_chr"))
    d = d[d.y_true.notna() & (d.horizon <= 3)]
    thr = thresholds(name, d)
    d = d.assign(m=d.target_date.dt.month).merge(thr, on=["target", "region", "test_year", "m"], how="inner")
    d["dataset"] = name
    for src in ("lgb", "chr"):
        ql = np.sort(np.log1p(d[[f"{c}_{src}" for c in QCOLS]].to_numpy(float)), axis=1)
        d[f"p_{src}"] = exceed_prob(ql, d.thr.to_numpy())
        d[f"med_{src}"] = np.expm1(ql[:, 3])
    qa = np.sort((np.log1p(d[[f"{c}_lgb" for c in QCOLS]].to_numpy(float)) +
                  np.log1p(d[[f"{c}_chr" for c in QCOLS]].to_numpy(float))) / 2, axis=1)
    d["p_avg"] = exceed_prob(qa, d.thr.to_numpy())
    d["med_avg"] = np.expm1(qa[:, 3])
    return d


def cutoff(p, e, q):
    for t in TAU:
        a = p >= t
        if (np.sum(a & ~e) + 1) / (np.sum(a) + 1) <= q:
            return t
    return 1.01


def online(d: pd.DataFrame, pcol: str, pooled: bool, iso: bool) -> np.ndarray:
    """Return alerts; calibration at each origin uses only realised cells (target_date <= origin)."""
    alert = np.zeros(len(d), bool)
    groups = [d] if pooled else [g for _, g in d.groupby("task")]
    for g in groups:
        for h, gh in g.groupby("horizon"):
            td, org = gh.target_date.to_numpy(), gh.origin.to_numpy()
            p, e = gh[pcol].to_numpy(), gh.event.to_numpy()
            idx_all = gh.index.to_numpy()
            for t in np.unique(org):
                cur = org == t
                past = td <= t
                if past.sum() < MIN_PAST:
                    continue
                pp, pe, pc = p[past], e[past], p[cur]
                if iso:
                    f = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pp, pe.astype(float))
                    pp, pc = f.predict(pp), f.predict(pc)
                alert[idx_all[cur]] = pc >= cutoff(pp, pe, Q)
    return alert


def main():
    d = pd.concat([load(n) for n in DATASETS], ignore_index=True)
    d["task"] = d.dataset + ":" + d.target
    d["event"] = (np.log1p(d.y_true) > d.thr) & (d.y_true >= MIN_CASES)
    pols = {}
    for src in ("lgb", "chr", "avg"):
        pols[(src, "point")] = ((np.log1p(d[f"med_{src}"]) > d.thr) & (d[f"med_{src}"] >= MIN_CASES)).to_numpy()
        pols[(src, "raw>=0.5")] = (d[f"p_{src}"] >= 0.5).to_numpy()
        pols[(src, "fdr-task")] = online(d, f"p_{src}", pooled=False, iso=False)
        pols[(src, "fdr-pooled")] = online(d, f"p_{src}", pooled=True, iso=False)
        pols[(src, "iso-pooled")] = online(d, f"p_{src}", pooled=True, iso=True)
        print("done", src, flush=True)
    rows = []
    for (src, pol), a in pols.items():
        x = d.assign(alert=a)
        for task, y in pd.concat([x.assign(task="ALL"), x]).groupby("task"):
            na = int(y.alert.sum())
            rows.append({"source": src, "policy": pol, "task": task, "alerts": na, "events": int(y.event.sum()),
                         "fdr": (y.alert & ~y.event).sum() / na if na else np.nan,
                         "recall": (y.alert & y.event).sum() / max(y.event.sum(), 1)})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "quantile_alerts.csv", index=False)
    ap = pd.DataFrame([{"task": t, **{s: average_precision_score(g.event, g[f"p_{s}"]) for s in ("lgb", "chr", "avg")},
                        "base_rate": g.event.mean()} for t, g in pd.concat([d.assign(task="ALL"), d]).groupby("task")])
    ap.to_csv(OUT / "quantile_alerts_prauc.csv", index=False)
    pd.set_option("display.width", 220)
    print(ap.round(3).to_string(index=False))
    allr = r[r.task == "ALL"].pivot(index="policy", columns="source", values=["fdr", "recall", "alerts"])
    print(allr.round(3).to_string())
    per = r[(r.task != "ALL") & (r.source == "avg")].pivot(index="task", columns="policy", values=["fdr", "recall"])
    print(per.round(2).to_string())


if __name__ == "__main__":
    main()
