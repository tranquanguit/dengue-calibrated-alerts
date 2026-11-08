"""Simulation: when do per-cell guarantees hold for outbreak alerts?

Synthetic panel: R provinces x Y years x 12 months. Latent log-incidence
    x[r,y,m] = a_r + seas_m + u_y + v[r,y,m],   u_y ~ N(0, sigma_u)  (year-level epidemic shock)
Event: x above the province-month 90th percentile of the no-shock distribution.
Forecast score: s = x + e with e ~ N(0, sigma_e) (forecast error); the score has no knowledge of the event.
For each origin (one per month of the last `n_test` years) the provinces of that month form one bulletin;
calibration = all cells of earlier years. Procedures:
  CS  : conformal selection (BH on conformal p-values vs past null cells), level q
  CRC : smallest cut-off with (false+1)/(alerts+1) <= q on past cells (pilot rule)
  PT  : point rule s > 0 (no calibration), the paper's main comparator
Metrics: bulletin FDR (mean over bulletins of V/max(R,1)) and recall, averaged over replicates.
sigma_u = 0 -> cells exchangeable (theory applies); sigma_u > 0 -> clustered by year.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alerts_core as ac  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/simulation"


def one(seed, sigma_u, q, R=40, Y=14, n_test=4, sigma_e=0.6, sigma_w=0.0, rho=0.0, shock_in_score=True):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, R)[:, None, None]
    seas = np.sin(2 * np.pi * np.arange(12) / 12)[None, None, :]
    u = rng.normal(0, sigma_u, Y)[None, :, None]
    v = rng.normal(0, 1, (R, Y, 12))
    x = a + seas + u + v
    thr = a + seas + 1.2816  # 90th percentile of the no-shock distribution
    ev = x > thr
    # forecast error: idiosyncratic + province-correlated within month (rho) + shared bulletin error w_{y,m}
    common = rng.normal(0, 1, (1, Y, 12))
    e = sigma_e * (np.sqrt(1 - rho) * rng.normal(0, 1, x.shape) + np.sqrt(rho) * common)
    w = rng.normal(0, sigma_w, (1, Y, 12))
    s = x + e + w - thr - (0 if shock_in_score else u)  # margin score (optionally blind to the year shock)
    res = {"cs": [], "crc": [], "pt": []}
    rec = {"cs": [0, 0], "crc": [0, 0], "pt": [0, 0]}
    tot = {"cs": [0, 0], "crc": [0, 0], "pt": [0, 0]}  # pooled V, R
    pos = {"cs": [], "crc": [], "pt": []}
    for y in range(Y - n_test, Y):
        cal_s, cal_e = s[:, :y, :].ravel(), ev[:, :y, :].ravel()
        grid = np.quantile(cal_s, np.linspace(0.5, 0.995, 100))
        cut = np.inf
        for c in grid:
            al = cal_s >= c
            if (np.sum(al & ~cal_e) + 1) / (np.sum(al) + 1) <= q:
                cut = c
                break
        for m in range(12):
            st, et = s[:, y, m], ev[:, y, m]
            p = ac.conformal_pvalues(st, cal_s, ~cal_e)
            for name, al in (("cs", ac.bh(p, q)), ("crc", st >= cut), ("pt", st > 0)):
                fdp = (al & ~et).sum() / max(al.sum(), 1)
                res[name].append(fdp)
                if al.sum():
                    pos[name].append(fdp)
                tot[name][0] += (al & ~et).sum()
                tot[name][1] += al.sum()
                rec[name][0] += (al & et).sum()
                rec[name][1] += et.sum()
    out = {f"fdr_{k}": np.mean(v) for k, v in res.items()}
    out |= {f"fdrpos_{k}": (np.mean(v) if v else np.nan) for k, v in pos.items()}
    out |= {f"fdrpool_{k}": tot[k][0] / max(tot[k][1], 1) for k in tot}
    out |= {f"recall_{k}": rec[k][0] / max(rec[k][1], 1) for k in rec}
    return out


SCENARIOS = {
    # name: kwargs  (sigma_u = year shock in incidence; shock_in_score=False -> forecasts miss the year shock;
    #                sigma_w = shared bulletin-level forecast error; rho = within-month correlation of errors)
    "iid": dict(sigma_u=0.0),
    "year shock, forecast knows it": dict(sigma_u=1.0),
    "year shock, forecast misses it": dict(sigma_u=1.0, shock_in_score=False),
    "shared bulletin error 0.25": dict(sigma_w=0.25),
    "shared bulletin error 0.5": dict(sigma_w=0.5),
    "shared bulletin error 1.0": dict(sigma_w=1.0),
    "within-month correlation 0.5": dict(rho=0.5),
    "missed year shock + shared error 0.5": dict(sigma_u=1.0, shock_in_score=False, sigma_w=0.5),
    # calibrated to the confirmation forecasts: between-bulletin share of log error variance 0.25 (mean) and
    # 0.5 (Sri Lanka); sigma_w = sigma_e * sqrt(share / (1 - share))
    "empirical share 0.25 (sigma_w 0.35)": dict(sigma_w=0.346),
    "empirical share 0.5 (sigma_w 0.6)": dict(sigma_w=0.6),
}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, kw in SCENARIOS.items():
        for q in (0.1, 0.2, 0.3):
            kw2 = {"sigma_u": 0.0, **kw}
            r = pd.DataFrame(Parallel(n_jobs=4)(delayed(one)(s_, q=q, **kw2) for s_ in range(300)))
            m, se = r.mean(), r.std() / np.sqrt(len(r))
            rows.append({"scenario": name, "q": q, **{k: m[k] for k in r}, **{f"se_{k}": se[k] for k in r}})
            print(name, q, {k: round(m[k], 3) for k in r}, flush=True)
    t = pd.DataFrame(rows)
    t.round(4).to_csv(OUT / "simulation.csv", index=False)
    print(t[["scenario", "q", "fdr_cs", "fdrpos_cs", "fdrpool_cs", "recall_cs", "fdr_crc", "fdrpool_crc", "recall_crc"]].round(3).to_string(index=False))
