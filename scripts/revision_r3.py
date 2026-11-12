"""Additional analyses on the confirmation set (set 3), in response to the internal review. Not pre-registered.

R11 fixed-alert regrouping: the alerts actually issued by CS@0.2 are kept fixed and only their bulletin membership is
    shuffled (within country x year, bulletin sizes kept); BH is NOT re-run. The pooled FDR is unchanged by
    construction, so the change in the gap measures the month-level grouping alone (20 permutations).
R12 Mondrian CS vs CS: paired country-year block bootstrap (the pre-registered resampling) of the gap Delta,
    bulletin FDR, pooled FDR and recall, on all origins and on the burn-in-excluded origins.
R13 burn-in-excluded origins (where CS could run): block-bootstrap CIs for CS@0.2, Mondrian CS@0.2 and the
    point-margin>0 rule, and the paired CS - point-margin>0 differences.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alerts_core as ac  # noqa: E402
import confirm  # noqa: E402
from revision_r1 import Q, decomp  # noqa: E402

OUT = confirm.OUT / "revision_r3"


def regroup_fixed(g, alert, rng):
    """Shuffle bulletin membership of cells within country x year (sizes kept), keeping each cell's alert flag."""
    g = g.assign(alert=alert, yr=g.target_date.dt.year).copy()
    parts = []
    for _, gy in g.groupby(["country", "yr"]):
        perm = rng.permutation(len(gy))
        parts.append(gy.assign(origin=gy.origin.to_numpy()[perm]))
    d = pd.concat(parts).reset_index(drop=True)
    return d, d.alert.to_numpy()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    x = confirm.build("ENS")
    rng = np.random.default_rng(11)
    reg, mon, bi = [], [], []
    for h in (1, 2, 3):
        g = x[x.horizon == h].reset_index(drop=True)
        pv = ac.run_procedure(g, "cs-p")
        ran = g.assign(r=~np.isnan(pv)).groupby(["country", "origin"]).r.transform("any").to_numpy()
        gg = g[ran].reset_index(drop=True)
        a_cs = ac.run_procedure(g, "cs", q=Q)[ran]
        b_cs = ac.bulletin_table(gg, a_cs)
        base = {**decomp(b_cs), **ac.metrics_from_bulletins(b_cs)}
        reg.append({"h": h, "what": "observed (burn-in excluded)", **base})
        reps = []
        for _ in range(20):
            d_, a_ = regroup_fixed(gg, a_cs, rng)
            b = ac.bulletin_table(d_, a_)
            reps.append({**decomp(b), **ac.metrics_from_bulletins(b)})
        reps = pd.DataFrame(reps)
        reg.append({"h": h, "what": "fixed alerts, random regrouping (mean of 20)", **reps.mean().to_dict()})
        reg.append({"h": h, "what": "fixed alerts, random regrouping (sd of 20)", **reps.std().to_dict()})
        a_all = {"cs@0.2": ac.run_procedure(g, "cs", q=Q), "cs-mondrian@0.2": ac.run_procedure(g, "cs-mondrian", q=Q),
                 "point-margin>0.0": ac.run_procedure(g, "point-margin", q=0.0)}
        for version, mask in (("all origins", np.ones(len(g), bool)), ("burn-in excluded", ran)):
            d = g[mask].reset_index(drop=True)
            for name, a in a_all.items():
                m = ac.evaluate(d, a[mask], Q, n_boot=1000)
                b = ac.bulletin_table(d, a[mask])
                bi.append({"version": version, "h": h, "procedure": name, "n_bulletins": len(b), **m, **decomp(b)})
            for base, alt in (("cs@0.2", "cs-mondrian@0.2"), ("point-margin>0.0", "cs@0.2")):
                pr = ac.evaluate_paired(d, a_all[base][mask], a_all[alt][mask], n_boot=1000)
                mon.append({"version": version, "h": h, "base": base, "alt": alt, **pr})
        print("h", h, flush=True)
    pd.DataFrame(reg).round(4).to_csv(OUT / "fixed_alert_regrouping.csv", index=False)
    pd.DataFrame(mon).round(4).to_csv(OUT / "paired_differences.csv", index=False)
    pd.DataFrame(bi).round(4).to_csv(OUT / "procedures_block_bootstrap.csv", index=False)
    print("done")
