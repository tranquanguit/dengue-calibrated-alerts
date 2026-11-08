"""Monitored-precision label on the development set (Paper 2).

Each bulletin carries the trailing 24-month verified precision of its procedure (same country and horizon) and a
one-sided 90 % Clopper-Pearson lower bound. Checks: (1) coverage — share of country-years whose realised pooled
precision is >= the bound issued at the year's first bulletin; (2) realised precision of alerts by label tier.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alerts_core as ac  # noqa: E402
import dev_alerts as da  # noqa: E402

PROCS = [("point", {}), ("cs@0.2", {"proc": "cs", "q": 0.2}), ("cs-mondrian@0.2", {"proc": "cs-mondrian", "q": 0.2}),
         ("cs@0.3", {"proc": "cs", "q": 0.3})]

if __name__ == "__main__":
    base = da.prepare()
    x = base.copy()
    for c in da.Q:
        x[c] = x[f"{c}_avg"]
    x = ac.scores(x, da.Q).reset_index(drop=True)
    sh = x.groupby(["country", "origin"]).persist.mean().rename("share_above")
    x = x.join(sh, on=["country", "origin"])
    x["state"] = (x.share_above >= da.STATE_CUT).astype(int)
    cov, tiers = [], []
    for name, kw in PROCS:
        kw = dict(kw)
        proc = kw.pop("proc", name)
        for h in (1, 2, 3):
            g = x[x.horizon == h].reset_index(drop=True)
            a = ac.run_procedure(g, proc, score="s_zmargin", **kw)
            for delay in (0, 2):
                b = ac.monitored_precision(ac.bulletin_table(g, a), delay=delay)
                b["yr"] = b.origin.dt.year
                for (c, y), gy in b.groupby(["country", "yr"]):
                    first = gy.sort_values("origin").iloc[0]
                    if gy.R.sum() == 0 or np.isnan(first.prec_lb):
                        continue
                    cov.append({"procedure": name, "h": h, "delay": delay, "country": c, "yr": y,
                                "bound": first.prec_lb, "realised": gy.TP.sum() / gy.R.sum(),
                                "covered": gy.TP.sum() / gy.R.sum() >= first.prec_lb})
                if delay == 0:
                    bb = b[b.R > 0].dropna(subset=["prec_lb"])
                    bb["tier"] = pd.cut(bb.prec_lb, [-0.01, 0.5, 0.7, 1.0], labels=["low <0.5", "mid 0.5-0.7", "high >=0.7"])
                    for t, gt in bb.groupby("tier", observed=True):
                        tiers.append({"procedure": name, "h": h, "tier": t, "alerts": int(gt.R.sum()),
                                      "realised_precision": gt.TP.sum() / gt.R.sum()})
    cov, tiers = pd.DataFrame(cov), pd.DataFrame(tiers)
    out = da.OUT
    s = cov.groupby(["procedure", "h", "delay"]).agg(country_years=("covered", "size"), coverage=("covered", "mean"),
                                                     mean_bound=("bound", "mean"), mean_realised=("realised", "mean")).reset_index()
    s.round(4).to_csv(out / "monitored_precision_coverage.csv", index=False)
    tiers.round(4).to_csv(out / "monitored_precision_tiers.csv", index=False)
    print(s.round(3).to_string(index=False))
    print(tiers.round(3).to_string(index=False))
