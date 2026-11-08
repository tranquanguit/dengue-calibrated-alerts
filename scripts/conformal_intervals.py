"""Pilot: online conformal intervals and conformal outbreak alerts on the v2 benchmark predictions.

Calibration never looks ahead: at forecast origin t only residuals whose target month
has already been observed (target_date <= t) are used. Those residuals are genuine
out-of-sample errors of earlier rolling-origin folds (and of earlier months of the
current fold), so no additional model fitting is needed.

Residual score: r = log1p(y) - log1p(yhat) (signed, so intervals can be asymmetric).
Methods
  pooled   : quantiles of all realised residuals of the dataset (all regions)
  scaled   : residuals divided by a region scale (mean |r| of that region so far)
  aci      : pooled + adaptive conformal inference (Gibbs & Candes 2021), alpha_t updated
             monthly from realised panel coverage
Alerts (event = target exceeds the region x calendar-month endemic threshold
mean + 2 sd of log1p cases, estimated on training years only)
  point    : alert if point forecast > threshold
  conf(a)  : alert if the one-sided conformal lower bound at level a exceeds the threshold
             -> P(alert and no event) <= a per region-month (marginal guarantee)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
RES = ROOT / "forecasts/benchmark"
OUT = ROOT / "results"
DATASETS = ["thailand", "indonesia", "cambodia", "laos", "malaysia", "philippines", "vietnam",
            "srilanka", "brazil", "japan"]
MODELS = ["AR", "Ens(AR,Raw-all)", "SeasonalNaive"]
ALPHA = 0.1          # 90 % two-sided intervals
GAMMA = 0.02         # ACI step size
ALERT_LEVELS = [0.05, 0.10, 0.20]
MIN_CAL = 200        # minimum realised residuals before intervals are issued


def load_predictions(name: str) -> pd.DataFrame:
    p = pd.read_parquet(RES / name / "predictions.parquet")
    key = ["target", "region", "origin", "target_date", "horizon", "test_year"]
    w = p.pivot_table(index=key, columns="model", values="y_pred")
    w["Ens(AR,Raw-all)"] = w[["AR", "AR+Raw(all)"]].mean(axis=1)
    truth = p[p.model == "AR"].set_index(key).y_true
    long = w[MODELS].stack().rename("y_pred").reset_index().rename(columns={"level_6": "model"})
    long = long.merge(truth.rename("y_true").reset_index(), on=key)
    long["dataset"] = name
    return long


def thresholds(name: str, preds: pd.DataFrame) -> pd.Series:
    """Endemic threshold per (target, region, test_year, calendar month) from training years."""
    df = pd.read_csv(ROOT / "data/processed" / f"{name}_power_monthly.csv", parse_dates=["date"])
    rows = []
    for tgt in preds.target.unique():
        d = df[["region", "date", tgt]].dropna().rename(columns={tgt: "y"})
        d["ly"], d["m"], d["yr"] = np.log1p(d.y), d.date.dt.month, d.date.dt.year
        for Y in preds.test_year.unique():
            h = d[d.yr < Y].groupby(["region", "m"]).ly.agg(["mean", "std", "count"])
            h = h[h["count"] >= 3]
            rows.append(pd.DataFrame({"target": tgt, "region": h.index.get_level_values(0),
                                      "test_year": Y, "m": h.index.get_level_values(1),
                                      "thr": (h["mean"] + 2 * h["std"].fillna(0)).to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def run_group(g: pd.DataFrame) -> pd.DataFrame:
    """Online conformal for one (dataset, target, model, horizon) panel, ordered by origin."""
    g = g.sort_values("origin").reset_index(drop=True)
    g["r"] = np.log1p(g.y_true) - np.log1p(g.y_pred)
    g["lp"] = np.log1p(g.y_pred)
    out = {k: np.full(len(g), np.nan) for k in
           ["lo_pooled", "hi_pooled", "lo_scaled", "hi_scaled", "lo_aci", "hi_aci",
            *[f"lb_{a}" for a in ALERT_LEVELS]]}
    td = g.target_date.to_numpy()
    r = g.r.to_numpy()
    alpha_t = ALPHA
    for t in np.sort(g.origin.unique()):
        cal = td <= t                     # residuals already observed at origin t
        idx = np.flatnonzero(g.origin.to_numpy() == t)
        if cal.sum() >= MIN_CAL:
            rc = r[cal]
            q = lambda a: np.quantile(rc, min(max(a, 0.0), 1.0), method="higher")  # noqa: E731
            lo = np.quantile(rc, ALPHA / 2, method="lower")
            hi = np.quantile(rc, 1 - ALPHA / 2, method="higher")
            out["lo_pooled"][idx], out["hi_pooled"][idx] = g.lp[idx] + lo, g.lp[idx] + hi
            # region-scaled residuals
            reg_cal = g[cal].assign(ar=np.abs(rc)).groupby("region").ar.agg(["mean", "count"])
            scale_all = np.abs(rc).mean()
            sc = g.region[cal].map(reg_cal["mean"].where(reg_cal["count"] >= 6, scale_all)).to_numpy()
            z = rc / np.maximum(sc, 1e-6)
            s_t = g.region[idx].map(reg_cal["mean"].where(reg_cal["count"] >= 6, scale_all)).fillna(scale_all).to_numpy()
            zl, zh = np.quantile(z, ALPHA / 2, method="lower"), np.quantile(z, 1 - ALPHA / 2, method="higher")
            out["lo_scaled"][idx], out["hi_scaled"][idx] = g.lp[idx] + zl * s_t, g.lp[idx] + zh * s_t
            # ACI
            out["lo_aci"][idx] = g.lp[idx] + q(alpha_t / 2)
            out["hi_aci"][idx] = g.lp[idx] + q(1 - alpha_t / 2)
            for a in ALERT_LEVELS:  # one-sided lower bound, scaled residuals
                out[f"lb_{a}"][idx] = g.lp[idx] + np.quantile(z, a, method="lower") * s_t
        # ACI update with the coverage of targets realised at month t
        done = np.flatnonzero((td == t) & ~np.isnan(out["lo_aci"]))
        if len(done):
            ly = np.log1p(g.y_true.to_numpy()[done])
            miss = np.mean((ly < out["lo_aci"][done]) | (ly > out["hi_aci"][done]))
            alpha_t = alpha_t + GAMMA * (ALPHA - miss)
    for k, v in out.items():
        g[k] = v
    return g


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for name in DATASETS:
        p = load_predictions(name)
        thr = thresholds(name, p)
        p = p.assign(m=p.target_date.dt.month).merge(thr, on=["target", "region", "test_year", "m"], how="left")
        res = [run_group(g) for _, g in p.groupby(["target", "model", "horizon"])]
        frames.append(pd.concat(res))
        print("done", name, flush=True)
    d = pd.concat(frames, ignore_index=True)
    d = d[d.lo_pooled.notna()]
    d.to_parquet(OUT / "conformal_predictions.parquet", index=False)

    ly = np.log1p(d.y_true)
    rows = []
    for meth in ["pooled", "scaled", "aci"]:
        cov = (ly >= d[f"lo_{meth}"]) & (ly <= d[f"hi_{meth}"])
        width = np.expm1(d[f"hi_{meth}"]) - np.expm1(d[f"lo_{meth}"])
        tmp = d.assign(cov=cov, width=width, rel_width=width / (d.y_true.groupby([d.dataset, d.target]).transform("mean") + 1))
        # coverage when incidence is in the top decile of the panel (outbreak months)
        top = d.y_true >= d.groupby(["dataset", "target"]).y_true.transform(lambda s: s.quantile(0.9))
        for (ds, tg, mdl), x in tmp.groupby(["dataset", "target", "model"]):
            rows.append({"dataset": ds, "target": tg, "model": mdl, "method": meth,
                         "coverage": x["cov"].mean(), "coverage_top10": x["cov"][top[x.index]].mean(),
                         "rel_width": x.rel_width.median()})
    cov = pd.DataFrame(rows)
    cov.to_csv(OUT / "interval_coverage.csv", index=False)

    a = d[d.thr.notna() & (d.horizon <= 3)].copy()
    a["event"] = np.log1p(a.y_true) > a.thr
    pol = {"point": np.log1p(a.y_pred) > a.thr}
    for lv in ALERT_LEVELS:
        pol[f"conformal_{int(lv * 100)}"] = a[f"lb_{lv}"] > a.thr
    rows = []
    for name, alert in pol.items():
        x = a.assign(alert=alert.to_numpy())
        for (mdl, ds), y in pd.concat([x.assign(dataset="ALL"), x]).groupby(["model", "dataset"]):
            ne = ~y.event
            rows.append({"model": mdl, "dataset": ds, "policy": name, "events": int(y.event.sum()),
                         "alerts": int(y.alert.sum()),
                         "false_alarm_per_cell": float((y.alert & ne).mean()),
                         "false_alarm_rate_nonevent": float((y.alert & ne).sum() / max(ne.sum(), 1)),
                         "precision": float((y.alert & y.event).sum() / max(y.alert.sum(), 1)),
                         "recall": float((y.alert & y.event).sum() / max(y.event.sum(), 1))})
    al = pd.DataFrame(rows)
    al.to_csv(OUT / "alert_performance.csv", index=False)

    pd.set_option("display.width", 220)
    print(cov.groupby(["model", "method"])[["coverage", "coverage_top10", "rel_width"]].median().round(3))
    print(al[al.dataset == "ALL"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
