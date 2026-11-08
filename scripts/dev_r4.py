"""Development-set rerun with the final procedure set.

- CS variants calibrated on eligible past cells only.
- Paired gaps: gap_pooled = pooled FDR - bulletin FDR, gap_pos = FDR|R>0 - bulletin FDR (paired bootstrap CIs).
- Frozen point-rule family: margin > c and zmargin > c, c in {0, 0.1, 0.2, 0.3, 0.5}.
- q sweep for CS and Mondrian (q in {0.1, 0.15, 0.2, 0.25, 0.3}); paired Mondrian - CS differences.
- Mondrian with the epidemic state lagged one month (label-delay sensitivity).
- Projected CI half-width for the confirmation set from the dev bootstrap (scaled by sqrt(n_dev / n_conf)).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alerts_core as ac  # noqa: E402
import dev_alerts as da  # noqa: E402

QS = (0.1, 0.15, 0.2, 0.25, 0.3)
CS_ = (0.0, 0.1, 0.2, 0.3, 0.5)
N_BOOT = 500


def build():
    base = da.prepare()
    x = base.copy()
    for c in da.Q:
        x[c] = x[f"{c}_avg"]
    x = ac.scores(x, da.Q).reset_index(drop=True)
    sh = x.groupby(["country", "origin"]).persist.mean().rename("share_above")
    x = x.join(sh, on=["country", "origin"])
    x["state"] = (x.share_above >= da.STATE_CUT).astype(int)
    lag = sh.reset_index().sort_values("origin")
    lag["share_lag1"] = lag.groupby("country").share_above.shift(1)
    x = x.merge(lag[["country", "origin", "share_lag1"]], on=["country", "origin"], how="left")
    x["state_lag1"] = (x.share_lag1.fillna(0) >= da.STATE_CUT).astype(int)
    return x


if __name__ == "__main__":
    out = da.OUT
    x = build()
    procs = [(f"point-margin>{c}", "point-margin", c) for c in CS_] + [(f"point-z>{c}", "point-z", c) for c in CS_]
    procs += [(f"cs@{q}", "cs", q) for q in QS] + [(f"cs-mondrian@{q}", "cs-mondrian", q) for q in QS]
    procs += [(f"cs-mondrian-lag1@{q}", "cs-mondrian-lag1", q) for q in (0.2, 0.3)]
    procs += [(f"crc@{q}", "crc", q) for q in (0.2, 0.3)] + [("topk", "topk", None)]
    res, leads, alerts = [], [], {}
    for name, proc, q in procs:
        a_all = np.zeros(len(x), bool)
        for h in (1, 2, 3):
            m = (x.horizon == h).to_numpy()
            g = x[m].reset_index(drop=True)
            kw = {"q": q} if q is not None else {}
            a = ac.run_procedure(g, proc, score="s_zmargin", **kw)
            a_all[np.flatnonzero(m)] = a
            res.append({"procedure": name, "family": proc, "level": q, "h": h,
                        **ac.evaluate(g, a, q if proc.startswith(("cs", "crc")) else 0.2, n_boot=N_BOOT)})
        alerts[name] = a_all
        lt = ac.lead_time(x, a_all)
        leads.append({"procedure": name, "onsets": len(lt), "detected_any_h": float((lt.lead > 0).mean()),
                      "median_lead_detected": float(lt.lead[lt.lead > 0].median()) if (lt.lead > 0).any() else np.nan})
        print(name, flush=True)
    r = pd.DataFrame(res)
    r.round(4).to_csv(out / "r4_procedures.csv", index=False)
    pd.DataFrame(leads).round(4).to_csv(out / "r4_lead_time.csv", index=False)
    # paired Mondrian - CS at each q, and lag-1 Mondrian - CS
    pr = []
    pairs = [(f"cs@{q}", f"cs-mondrian@{q}") for q in QS] + [(f"cs@{q}", f"cs-mondrian-lag1@{q}") for q in (0.2, 0.3)]
    for a1, a2 in pairs:
        for h in (1, 2, 3):
            m = (x.horizon == h).to_numpy()
            g = x[m].reset_index(drop=True)
            pr.append({"base": a1, "alt": a2, "h": h, **ac.evaluate_paired(g, alerts[a1][m], alerts[a2][m], n_boot=N_BOOT)})
    pd.DataFrame(pr).round(4).to_csv(out / "r4_paired_mondrian.csv", index=False)
    # projected CI half-width of the primary gap for the confirmation set
    n_dev = x.assign(yr=x.target_date.dt.year).groupby(["country", "yr"]).ngroups
    proj = []
    for n_conf in (12, 16, 20, 24):
        for _, row in r[r.procedure == "cs@0.2"].iterrows():
            hw = (row.gap_pooled_hi - row.gap_pooled_lo) / 2
            proj.append({"h": row.h, "n_dev_country_years": n_dev, "n_conf_country_years": n_conf,
                         "gap_pooled": row.gap_pooled, "dev_halfwidth": hw,
                         "projected_halfwidth": hw * np.sqrt(n_dev / n_conf)})
    pd.DataFrame(proj).round(4).to_csv(out / "r4_ci_projection.csv", index=False)
    print(r[["procedure", "h", "alerts", "fdr_bulletin", "fdr_given_alert", "fdr_pooled", "recall", "onset_recall",
             "gap_pooled", "gap_pooled_lo", "gap_pooled_hi"]].round(3).to_string(index=False))
