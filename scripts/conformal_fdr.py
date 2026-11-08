"""Pilot 2: outbreak alerts with an online, conformal-risk-control bound on the false discovery
rate (share of alerts that turn out to be false), on top of conformal predictive probabilities.

At each forecast origin t (per task x model x horizon):
  1. P(event) for every region = share of realised scaled residuals z (target_date <= t) with
     log1p(yhat) + z * s_region > threshold            (conformal predictive distribution)
  2. tau_t = smallest probability cut-off such that, on all realised past cells,
     (false alerts + 1) / (alerts + 1) <= q            (CRC-style finite-sample correction)
  3. alert if P(event) >= tau_t
Baselines: point forecast above threshold; fixed P(event) >= 0.5.
Event: log1p(y) above the training-years region x month mean + 2 sd AND y >= MIN_CASES.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conformal_intervals import DATASETS, load_predictions, thresholds  # noqa: E402

OUT = ROOT / "results"
MODELS = ["AR", "Ens(AR,Raw-all)"]
Q_TARGETS = [0.2, 0.3, 0.5]
MIN_CAL = 200
MIN_CASES = 5
TAU_GRID = np.round(np.arange(0.05, 1.0001, 0.025), 3)


def probs_for_group(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("origin").reset_index(drop=True)
    lp = np.log1p(g.y_pred.to_numpy())
    r = np.log1p(g.y_true.to_numpy()) - lp
    td, org, reg = g.target_date.to_numpy(), g.origin.to_numpy(), g.region.to_numpy()
    thr = g.thr.to_numpy()
    prob = np.full(len(g), np.nan)
    for t in np.unique(org):
        cal = td <= t
        if cal.sum() < MIN_CAL:
            continue
        idx = np.flatnonzero(org == t)
        rc = r[cal]
        sc_tab = pd.Series(np.abs(rc)).groupby(reg[cal]).agg(["mean", "count"])
        scale_all = np.abs(rc).mean()
        sc_map = sc_tab["mean"].where(sc_tab["count"] >= 6, scale_all)
        s_cal = pd.Series(reg[cal]).map(sc_map).to_numpy()
        z = np.sort(rc / np.maximum(s_cal, 1e-6))
        s_t = pd.Series(reg[idx]).map(sc_map).fillna(scale_all).to_numpy()
        need = (thr[idx] - lp[idx]) / np.maximum(s_t, 1e-6)   # event iff z > need
        prob[idx] = 1.0 - np.searchsorted(z, need, side="right") / len(z)
    g["prob"] = prob
    return g


def online_policy(g: pd.DataFrame, q: float) -> np.ndarray:
    """Alert decisions with tau chosen at each origin from already-realised cells only."""
    g = g.sort_values("origin")
    alert = np.zeros(len(g), bool)
    pos = {i: k for k, i in enumerate(g.index)}
    td, org, pr, ev = g.target_date.to_numpy(), g.origin.to_numpy(), g.prob.to_numpy(), g.event.to_numpy()
    for t in np.unique(org):
        idx = np.flatnonzero(org == t)
        past = (td <= t) & ~np.isnan(pr)
        tau = 1.01
        if past.sum() >= MIN_CAL:
            p, e = pr[past], ev[past]
            for cand in TAU_GRID:
                a = p >= cand
                if (np.sum(a & ~e) + 1) / (np.sum(a) + 1) <= q:
                    tau = cand
                    break
        alert[idx] = pr[idx] >= tau
    out = pd.Series(alert, index=g.index)
    return out.reindex(sorted(pos)).to_numpy()


def main():
    frames = []
    for name in DATASETS:
        p = load_predictions(name)
        p = p[p.model.isin(MODELS) & (p.horizon <= 3)]
        thr = thresholds(name, p)
        p = p.assign(m=p.target_date.dt.month).merge(thr, on=["target", "region", "test_year", "m"], how="left")
        p = p[p.thr.notna()]
        frames += [probs_for_group(g) for _, g in p.groupby(["target", "model", "horizon"])]
        print("probs", name, flush=True)
    d = pd.concat(frames, ignore_index=True)
    d = d[d.prob.notna()].reset_index(drop=True)
    d["event"] = (np.log1p(d.y_true) > d.thr) & (d.y_true >= MIN_CASES)
    d["task"] = d.dataset + ":" + d.target

    pols = {"point": (np.log1p(d.y_pred) > d.thr) & (d.y_pred >= MIN_CASES), "prob>=0.5": d.prob >= 0.5}
    for q in Q_TARGETS:
        a = np.zeros(len(d), bool)
        for _, g in d.groupby(["task", "model", "horizon"]):
            a[g.index] = online_policy(g, q)
        pols[f"CRC-FDR<={q}"] = a
    rows = []
    for name, alert in pols.items():
        x = d.assign(alert=np.asarray(alert))
        for (task, mdl), y in x.groupby(["task", "model"]):
            na = y.alert.sum()
            rows.append({"task": task, "model": mdl, "policy": name, "events": int(y.event.sum()),
                         "alerts": int(na), "fdr": float((y.alert & ~y.event).sum() / na) if na else np.nan,
                         "recall": float((y.alert & y.event).sum() / max(y.event.sum(), 1))})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "alert_fdr.csv", index=False)
    pd.set_option("display.width", 220)
    piv = r[r.model == "Ens(AR,Raw-all)"].pivot(index="task", columns="policy", values=["fdr", "recall"]).round(2)
    print(piv.to_string())
    agg = r.groupby(["model", "policy"]).agg(median_fdr=("fdr", "median"), max_fdr=("fdr", "max"),
                                             median_recall=("recall", "median"))
    print(agg.round(3))


if __name__ == "__main__":
    main()
