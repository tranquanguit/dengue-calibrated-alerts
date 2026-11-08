"""Pilot 3: adaptive FDR control for outbreak alerts under non-stationarity.

Variants of the online cut-off rule of conformal_fdr.py:
  window  : estimate the FDR of a cut-off only on the last W months of realised cells
  adaptive: target q_t updated by realised feedback, q_{t+1} = q_t + gamma (q* - FDP_t)
            (an ACI-style correction, Gibbs & Candes 2021, applied to the alert FDR)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conformal_intervals import DATASETS, load_predictions, thresholds  # noqa: E402
from conformal_fdr import MIN_CAL, MIN_CASES, TAU_GRID, probs_for_group  # noqa: E402

OUT = ROOT / "results"
Q = 0.3
GAMMA = 0.05


def policy(g, q_star, window=None, adaptive=False):
    g = g.sort_values("origin")
    td, org = g.target_date.to_numpy(), g.origin.to_numpy()
    pr, ev = g.prob.to_numpy(), g.event.to_numpy()
    alert = np.zeros(len(g), bool)
    q = q_star
    for t in np.unique(org):
        idx = np.flatnonzero(org == t)
        past = (td <= t) & ~np.isnan(pr)
        if window:
            past &= td > (t - np.timedelta64(window * 31, "D"))
        tau = 1.01
        if past.sum() >= MIN_CAL // (4 if window else 1):
            p, e = pr[past], ev[past]
            for cand in TAU_GRID:
                a = p >= cand
                if (np.sum(a & ~e) + 1) / (np.sum(a) + 1) <= max(q, 0.01):
                    tau = cand
                    break
        alert[idx] = pr[idx] >= tau
        if adaptive:  # feedback from alerts whose target month is realised now
            done = (td == t) & alert
            if done.any():
                fdp = np.mean(~ev[done])
                q = float(np.clip(q + GAMMA * (q_star - fdp), 0.01, 0.99))
    return pd.Series(alert, index=g.index).sort_index().to_numpy()


frames = []
for name in DATASETS:
    p = load_predictions(name)
    p = p[(p.model == "Ens(AR,Raw-all)") & (p.horizon <= 3)]
    thr = thresholds(name, p)
    p = p.assign(m=p.target_date.dt.month).merge(thr, on=["target", "region", "test_year", "m"], how="left")
    frames += [probs_for_group(g) for _, g in p[p.thr.notna()].groupby(["target", "model", "horizon"])]
d = pd.concat(frames, ignore_index=True)
d = d[d.prob.notna()].reset_index(drop=True)
d["event"] = (np.log1p(d.y_true) > d.thr) & (d.y_true >= MIN_CASES)
d["task"] = d.dataset + ":" + d.target
d = d[~d.dataset.isin(["japan"])]  # too few events for alerting in this pilot
pols = {"point": ((np.log1p(d.y_pred) > d.thr) & (d.y_pred >= MIN_CASES)).to_numpy()}
for lab, kw in {"static": {}, "window24": {"window": 24}, "adaptive": {"adaptive": True},
                "window24+adaptive": {"window": 24, "adaptive": True}}.items():
    a = np.zeros(len(d), bool)
    for _, g in d.groupby(["task", "horizon"]):
        a[g.index] = policy(g, Q, **kw)
    pols[lab] = a
rows = []
for lab, a in pols.items():
    x = d.assign(alert=a)
    for task, y in x.groupby("task"):
        na = y.alert.sum()
        rows.append({"task": task, "policy": lab, "alerts": int(na), "events": int(y.event.sum()),
                     "fdr": (y.alert & ~y.event).sum() / na if na else np.nan,
                     "recall": (y.alert & y.event).sum() / max(y.event.sum(), 1)})
r = pd.DataFrame(rows)
r.to_csv(OUT / "alert_fdr_adaptive.csv", index=False)
pd.set_option("display.width", 220)
print(r.pivot(index="task", columns="policy", values=["fdr", "recall"]).round(2).to_string())
pooled = pd.DataFrame({lab: {"fdr": float((~d.event[a]).mean()), "recall": float(a[d.event].mean()),
                             "alerts": int(a.sum())} for lab, a in pols.items()}).T
print(pooled.round(3))
