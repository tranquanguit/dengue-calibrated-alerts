"""Additional analyses on the confirmation set (set 1). Not pre-registered unless noted.

R2  burn-in origins (CS could not run: < 200 eligible calibration cells) excluded for every procedure
R3  excess pooled FDR over the announced q; share of country-years with pooled FDP > q
R4  BH over eligible cells only (m = m_el)
R5  decomposition  pooled = FDR|R>0 + Cov(R, FDP | R>0) / E[R | R>0]; corr(R, FDP)
R6  exchangeable reference: CS p-values regrouped into random pseudo-bulletins (same sizes) within
    country x horizon x year, BH per pseudo-bulletin; empirical between-bulletin share of forecast-error variance
R10 country-stratified bootstrap (country-years resampled within country) and leave-one-country-out estimates
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alerts_core as ac  # noqa: E402
import confirm  # noqa: E402

OUT = confirm.OUT / "revision_r1"
Q = 0.2
N_BOOT = 1000


def decomp(b):
    pos = b[b.R > 0]
    fdp = pos.V / pos.R
    m = ac.metrics_from_bulletins(b)
    return {"delta": m["gap_pooled"], "delta_empty": m["gap_pos"], "cov_part": m["fdr_pooled"] - m["fdr_given_alert"],
            "corr_R_FDP": float(np.corrcoef(pos.R, fdp)[0, 1]) if len(pos) > 2 else np.nan}


def strat_boot(b1, b2=None, n=N_BOOT, seed=0):
    """Country-stratified bootstrap of (metric of b1) or (b2 - b1); returns dict of percentile CIs."""
    rng = np.random.default_rng(seed)
    key = b1.country + ":" + b1.yr.astype(str)
    units = {c: g[["country", "yr"]].drop_duplicates().to_numpy() for c, g in b1.groupby("country")}
    reps = []
    for _ in range(n):
        pick = np.concatenate([u[rng.integers(0, len(u), len(u))] for u in units.values()])
        w = key.map(pd.Series([f"{a}:{c}" for a, c in pick]).value_counts()).fillna(0).to_numpy()
        m1 = ac.metrics_from_bulletins(b1, w)
        if b2 is None:
            reps.append(m1)
        else:
            m2 = ac.metrics_from_bulletins(b2, w)
            reps.append({k: m2[k] - m1[k] for k in m1})
    reps = pd.DataFrame(reps)
    return {f"{k}_{s}": float(reps[k].quantile(a)) for k in ("fdr_bulletin", "fdr_pooled", "gap_pooled", "recall")
            for s, a in (("lo", 0.025), ("hi", 0.975))}


def pseudo_bulletins(g, pv, q, rng):
    """BH on CS p-values regrouped at random into pseudo-bulletins of the original sizes (within country x year)."""
    g = g.assign(pv=pv, yr=g.target_date.dt.year).copy()
    g = g[g.pv.notna()]
    out = []
    for (_, _), gy in g.groupby(["country", "yr"]):
        perm = rng.permutation(len(gy))
        gy = gy.assign(origin=gy.origin.to_numpy()[perm])  # shuffle bulletin membership, keep sizes
        a = np.zeros(len(gy), bool)
        for o, gb in gy.groupby("origin"):
            pos = np.flatnonzero((gy.origin == o).to_numpy())
            a[pos] = ac.bh(gb.pv.to_numpy(), q)
        out.append(gy.assign(alert=a))
    d = pd.concat(out)
    return d, d.alert.to_numpy()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    x = confirm.build("ENS")
    fam = [(f"point-{s}>{c}", f"point-{s}", c) for s in ("margin", "z") for c in confirm.CS_]
    rows, dec, perm, loco, strat = [], [], [], [], []
    rng = np.random.default_rng(7)
    for h in (1, 2, 3):
        g = x[x.horizon == h].reset_index(drop=True)
        pv = ac.run_procedure(g, "cs-p")
        ran = g.assign(r=~np.isnan(pv)).groupby(["country", "origin"]).r.transform("any").to_numpy()
        al = {"cs@0.2": ac.run_procedure(g, "cs", q=Q), "cs-el@0.2": ac.run_procedure(g, "cs-el", q=Q),
              "cs-mondrian@0.2": ac.run_procedure(g, "cs-mondrian", q=Q)}
        for name, proc, c in fam:
            al[name] = ac.run_procedure(g, proc, q=c)
        for version, mask in (("all origins (pre-registered analysis)", np.ones(len(g), bool)), ("burn-in excluded", ran)):
            gg = g[mask].reset_index(drop=True)
            for name, a in al.items():
                b = ac.bulletin_table(gg, a[mask])
                m = ac.evaluate(gg, a[mask], Q, n_boot=0)
                rows.append({"version": version, "h": h, "procedure": name, "n_bulletins": len(b), **m,
                             "excess_pooled_over_q": m["fdr_pooled"] - Q, **decomp(b)})
            for name in ("cs@0.2", "cs-el@0.2", "point-margin>0.0"):
                b = ac.bulletin_table(gg, al[name][mask])
                strat.append({"version": version, "h": h, "what": name, **strat_boot(b)})
            b1, b2 = ac.bulletin_table(gg, al["point-margin>0.0"][mask]), ac.bulletin_table(gg, al["cs@0.2"][mask])
            strat.append({"version": version, "h": h, "what": "cs@0.2 - point-margin>0.0", **strat_boot(b1, b2)})
            pr = ac.evaluate_paired(gg, al["point-margin>0.0"][mask], al["cs@0.2"][mask], n_boot=N_BOOT)
            strat.append({"version": version, "h": h, "what": "paired unstratified cs@0.2 - point-margin>0.0",
                          **{k.replace("d_", ""): v for k, v in pr.items() if k.startswith("d_") and any(
                              k.startswith(f"d_{m}") for m in ("fdr_bulletin", "fdr_pooled", "gap_pooled", "recall"))}})
        # R6 exchangeable reference (20 permutations)
        for i in range(20):
            d_, a_ = pseudo_bulletins(g, pv, Q, rng)
            b = ac.bulletin_table(d_.reset_index(drop=True), a_)
            perm.append({"h": h, "rep": i, **decomp(b), **ac.metrics_from_bulletins(b)})
        # R10 leave-one-country-out
        for c in confirm.CONF:
            mm = (g.country != c).to_numpy()
            gg = g[mm].reset_index(drop=True)
            m_cs = ac.metrics_from_bulletins(ac.bulletin_table(gg, al["cs@0.2"][mm]))
            m_pt = ac.metrics_from_bulletins(ac.bulletin_table(gg, al["point-margin>0.0"][mm]))
            loco.append({"h": h, "dropped": c, "delta_cs": m_cs["gap_pooled"], "delta_point": m_pt["gap_pooled"],
                         "h2b": m_cs["gap_pooled"] - m_pt["gap_pooled"], "fdr_bulletin_cs": m_cs["fdr_bulletin"],
                         "pooled_cs": m_cs["fdr_pooled"], "pooled_point": m_pt["fdr_pooled"],
                         "recall_cs": m_cs["recall"], "recall_point": m_pt["recall"]})
        # empirical between-bulletin share of log forecast-error variance
        e = np.log1p(g.q50) - np.log1p(g.y_true)
        gg = g.assign(e=e)
        for c, gc in gg.groupby("country"):
            mu = gc.groupby("origin").e.transform("mean")
            dec.append({"h": h, "country": c, "var_total": gc.e.var(), "share_between_bulletins": mu.var() / gc.e.var()})
        print("h", h, flush=True)
    pd.DataFrame(rows).round(4).to_csv(OUT / "procedures_versions.csv", index=False)
    pd.DataFrame(strat).round(4).to_csv(OUT / "stratified_bootstrap.csv", index=False)
    pd.DataFrame(perm).groupby("h").agg(["mean", "std"]).round(4).to_csv(OUT / "exchangeable_reference.csv")
    pd.DataFrame(loco).round(4).to_csv(OUT / "leave_one_country_out.csv", index=False)
    pd.DataFrame(dec).round(4).to_csv(OUT / "error_variance_share.csv", index=False)
    print("done")
