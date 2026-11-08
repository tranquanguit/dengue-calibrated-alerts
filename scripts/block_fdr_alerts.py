"""Year-block calibration of the alert cut-off (exploratory).

Hypothesis: FDR control fails because false alarms cluster by season (non-epidemic years produce mostly
false alerts), violating exchangeability across region-months. Here each past calendar year of a task
is ONE calibration unit: for a candidate cut-off the false-discovery proportion is computed per year,
and the cut-off is the smallest one whose yearly FDPs satisfy the rule
  block-mean : mean over years of FDP_y (years with >= 1 alert; years without alerts count 0) <= q
  block-q75  : 75th percentile of FDP_y over years <= q
  pooled     : pooled (false+1)/(alerts+1) <= q        (the earlier rule, for reference)
All cut-offs use only target months already observed at the forecast origin.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import quantile_alerts as qa  # noqa: E402

RULES = ["pooled", "block-mean", "block-q75"]


def cutoff(p, e, yr, q, rule):
    for t in qa.TAU:
        a = p >= t
        if rule == "pooled":
            ok = (np.sum(a & ~e) + 1) / (np.sum(a) + 1) <= q
        else:
            fdp = []
            for y in np.unique(yr):
                m = yr == y
                na = a[m].sum()
                fdp.append(((a & ~e)[m].sum() + (1 if na else 0)) / (na + 1) if na else 0.0)
            stat = np.mean(fdp) if rule == "block-mean" else np.quantile(fdp, 0.75)
            ok = stat <= q
        if ok:
            return t
    return 1.01


def run(d, pcol, q, rule, gate=None):
    alert = np.zeros(len(d), bool)
    for _, g in d.groupby(["task", "horizon"]):
        td, org = g.target_date.to_numpy(), g.origin.to_numpy()
        p, e, yr = g[pcol].to_numpy(), g.event.to_numpy(), g.target_date.dt.year.to_numpy()
        if gate is not None:  # same eligibility rule as the point policy: predicted median >= MIN_CASES
            p = np.where(g[gate].to_numpy() >= qa.MIN_CASES, p, 0.0)
        idx = g.index.to_numpy()
        for t in np.unique(org):
            cur, past = org == t, td <= t
            if past.sum() < qa.MIN_PAST or len(np.unique(yr[past])) < 2:
                continue
            alert[idx[cur]] = p[cur] >= cutoff(p[past], e[past], yr[past], q, rule)
    return alert


if __name__ == "__main__":
    d = pd.concat([qa.load(n) for n in qa.DATASETS], ignore_index=True)
    d["task"] = d.dataset + ":" + d.target
    d["event"] = (np.log1p(d.y_true) > d.thr) & (d.y_true >= qa.MIN_CASES)
    for src in ("lgb", "chr", "avg"):  # margin score: log1p(median) - threshold, squashed to (0, 1)
        d[f"m_{src}"] = 1 / (1 + np.exp(-2 * (np.log1p(d[f"med_{src}"]) - d.thr)))
    ev = d.event.to_numpy()
    pt = ((np.log1p(d.med_avg) > d.thr) & (d.med_avg >= qa.MIN_CASES)).to_numpy()
    print("point(avg): fdr", round((pt & ~ev).sum() / pt.sum(), 3), "recall", round((pt & ev).sum() / ev.sum(), 3), "alerts", pt.sum())
    rows = []
    for q in [0.2, 0.3, 0.4, 0.5]:
        for src in ("lgb", "chr", "avg"):
            for rule in [r + "+gate" for r in RULES] + [r + "+margin" for r in RULES]:
                base = rule.split("+")[0]
                score = f"m_{src}" if rule.endswith("+margin") else f"p_{src}"
                a = run(d, score, q, base, gate=f"med_{src}")
                per = [(a & ~ev)[g.index].sum() / a[g.index].sum() for _, g in d.groupby("task") if a[g.index].sum() >= 20]
                yfdp = [(a & ~ev)[g.index].sum() / a[g.index].sum() for _, g in d.groupby(["task", d.target_date.dt.year]) if a[g.index].sum() >= 5]
                rows.append({"target_fdr": q, "source": src, "rule": rule, "alerts": int(a.sum()),
                             "fdr": (a & ~ev).sum() / a.sum() if a.sum() else np.nan,
                             "recall": (a & ev).sum() / ev.sum(),
                             "worst_task_fdr": max(per) if per else np.nan,
                             "share_task_years_fdr_over_q": float(np.mean(np.array(yfdp) > q)) if yfdp else np.nan})
        print("q", q, flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(qa.OUT / "block_fdr_alerts.csv", index=False)
    pd.set_option("display.width", 220)
    print(r.round(3).to_string(index=False))
