"""Confirmatory analysis for Paper 2 (pre-registration v3 + v3.1 addendum; branches prereg-p2-v3, prereg-p2-v3.1).

Confirmation countries: Philippines, Viet Nam (Block A, 2006-2010), Sri Lanka (Block B, 2016-2024),
Brazil (Block B, 2017-2024). Forecasts: ENS = per-level log1p average of Paper 1 Pooled-tuned and Chronos2-cross
quantiles (secondary: Pooled-tuned alone). Horizons 1-3. N_BOOT = 1000. Outputs in results/confirmation/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import alerts_core as ac  # noqa: E402

P1 = ROOT / "forecasts"
OUT = ROOT / "results/confirmation"
Q = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]
CONF = {"philippines": ("A", P1 / "chronos2/philippines.parquet", range(2006, 2011)),
        "vietnam": ("A", P1 / "chronos2/vietnam.parquet", range(2006, 2011)),
        "srilanka": ("B", P1 / "chronos2_grid/blockB/srilanka.parquet", range(2016, 2025)),
        "brazil": ("B", P1 / "chronos2/brazil.parquet", range(2017, 2025))}
STATE_CUT = 0.2
QS = (0.1, 0.15, 0.2, 0.25, 0.3)
CS_ = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5)
N_BOOT = 1000
KEY = ["region", "origin", "target_date", "horizon"]


def load(c, source="ENS"):
    blk, chp, years = CONF[c]
    g = pd.read_parquet(P1 / f"unified_grid/block{blk}.parquet")
    g = g[(g.country == c) & (g.design == "Pooled-tuned") & (g.horizon <= 3)].copy()
    g["region"] = g.region.str.split(":", n=1).str[1]
    g = g[g.target_date.dt.year.isin(list(years)) & g.y_true.notna()]
    if source == "ENS":
        ch = pd.read_parquet(chp)
        ch = ch[(ch.model == "Chronos2-cross") & (ch.horizon <= 3)][KEY + Q]
        d = g[KEY + ["y_true"] + Q].merge(ch, on=KEY, suffixes=("_p", "_c"))
        for q in Q:
            d[q] = np.expm1((np.log1p(d[f"{q}_p"].clip(lower=0)) + np.log1p(d[f"{q}_c"].clip(lower=0))) / 2)
        d = d[KEY + ["y_true"] + Q]
    elif source == "ENS-arith":  # Paper 1's ENS definition (arithmetic mean per level); sensitivity only
        ch = pd.read_parquet(chp)
        ch = ch[(ch.model == "Chronos2-cross") & (ch.horizon <= 3)][KEY + Q]
        d = g[KEY + ["y_true"] + Q].merge(ch, on=KEY, suffixes=("_p", "_c"))
        for q in Q:
            d[q] = (d[f"{q}_p"].clip(lower=0) + d[f"{q}_c"].clip(lower=0)) / 2
        d = d[KEY + ["y_true"] + Q]
    else:
        d = g[KEY + ["y_true"] + Q]
    panel = pd.read_csv(ROOT / "data/processed" / f"{c}_power_monthly.csv", parse_dates=["date"])
    panel = panel[["region", "date", "dengue"]].rename(columns={"dengue": "cases"}).dropna()
    thr = ac.endemic_channel(panel, 5, 2.0, True)
    flags = ac.onset_flags(panel, thr, 5)
    d = ac.add_events(d.assign(country=c), thr, 5)
    d = d.merge(flags[["region", "date", "onset"]].rename(columns={"date": "target_date"}), on=["region", "target_date"], how="left")
    pers = flags[["region", "date", "event"]].rename(columns={"date": "origin", "event": "persist"})
    return d.merge(pers, on=["region", "origin"], how="left")


def build(source="ENS"):
    x = pd.concat([load(c, source) for c in CONF], ignore_index=True)
    x = ac.scores(x, Q).reset_index(drop=True)
    sh = x.groupby(["country", "origin"]).persist.mean().rename("share_above")
    x = x.join(sh, on=["country", "origin"])
    x["state"] = (x.share_above >= STATE_CUT).astype(int)
    lag = sh.reset_index().sort_values("origin")
    lag["share_lag1"] = lag.groupby("country").share_above.shift(1)
    x = x.merge(lag[["country", "origin", "share_lag1"]], on=["country", "origin"], how="left")
    x["state_lag1"] = (x.share_lag1.fillna(0) >= STATE_CUT).astype(int)
    return x


def procedures():
    p = [(f"point-margin>{c}", "point-margin", c) for c in CS_] + [(f"point-z>{c}", "point-z", c) for c in CS_]
    p += [(f"cs@{q}", "cs", q) for q in QS] + [(f"cs-mondrian@{q}", "cs-mondrian", q) for q in QS]
    p += [(f"cs-mondrian-lag1@{q}", "cs-mondrian-lag1", q) for q in (0.2, 0.3)]
    p += [(f"crc@{q}", "crc", q) for q in (0.2, 0.3)] + [("topk", "topk", None)]
    return p


def run(x, procs, n_boot, per_country=True):
    res, leads, alerts = [], [], {}
    for name, proc, q in procs:
        a_all = np.zeros(len(x), bool)
        for h in (1, 2, 3):
            m = (x.horizon == h).to_numpy()
            g = x[m].reset_index(drop=True)
            a = ac.run_procedure(g, proc, score="s_zmargin", **({"q": q} if q is not None else {}))
            a_all[np.flatnonzero(m)] = a
            qq = q if proc.startswith(("cs", "crc")) else 0.2
            res.append({"procedure": name, "family": proc, "level": q, "h": h, "country": "pooled",
                        **ac.evaluate(g, a, qq, n_boot=n_boot)})
            if per_country:
                for c in CONF:
                    mc = (g.country == c).to_numpy()
                    res.append({"procedure": name, "family": proc, "level": q, "h": h, "country": c,
                                **ac.evaluate(g[mc].reset_index(drop=True), a[mc], qq, n_boot=200, seed=1)})
        alerts[name] = a_all
        lt = ac.lead_time(x, a_all)
        leads.append({"procedure": name, "onsets": len(lt), "detected_any_h": float((lt.lead > 0).mean()),
                      "median_lead_detected": float(lt.lead[lt.lead > 0].median()) if (lt.lead > 0).any() else np.nan})
        print(name, flush=True)
    return pd.DataFrame(res), pd.DataFrame(leads), alerts


def paired(x, alerts, pairs, n_boot):
    rows = []
    for a1, a2 in pairs:
        for h in (1, 2, 3):
            m = (x.horizon == h).to_numpy()
            g = x[m].reset_index(drop=True)
            rows.append({"base": a1, "alt": a2, "h": h, **ac.evaluate_paired(g, alerts[a1][m], alerts[a2][m], n_boot=n_boot)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    x = build("ENS")
    x.groupby("country").agg(rows=("event", "size"), event_rate=("event", "mean"),
                             years=("target_date", lambda s: s.dt.year.nunique())).to_csv(OUT / "data_summary.csv")
    r, lt, al = run(x, procedures(), N_BOOT)
    r.round(4).to_csv(OUT / "procedures.csv", index=False)
    lt.round(4).to_csv(OUT / "lead_time.csv", index=False)
    pairs = [("point-margin>0.0", "cs@0.2"), ("point-margin>0.0", "cs@0.3")]
    pairs += [(f"cs@{q}", f"cs-mondrian@{q}") for q in QS] + [(f"cs@{q}", f"cs-mondrian-lag1@{q}") for q in (0.2, 0.3)]
    # H4: CS@0.2 vs every point-family member (the dominating one is picked in evaluate_hypotheses.py)
    pairs += [(f"point-{s}>{c}", "cs@0.2") for s in ("margin", "z") for c in CS_ if not (s == "margin" and c == 0.0)]
    paired(x, al, pairs, N_BOOT).round(4).to_csv(OUT / "paired.csv", index=False)
    # H5 monitored precision
    cov = []
    for name in ("point-margin>0.0", "cs@0.2", "cs-mondrian@0.2", "cs@0.3"):
        for h in (1, 2, 3):
            m = (x.horizon == h).to_numpy()
            g = x[m].reset_index(drop=True)
            b = ac.monitored_precision(ac.bulletin_table(g, al[name][m]))
            b["yr"] = b.origin.dt.year
            for (c, y), gy in b.groupby(["country", "yr"]):
                first = gy.sort_values("origin").iloc[0]
                if gy.R.sum() == 0 or np.isnan(first.prec_lb):
                    continue
                cov.append({"procedure": name, "h": h, "country": c, "yr": y, "bound": first.prec_lb,
                            "realised": gy.TP.sum() / gy.R.sum()})
    cov = pd.DataFrame(cov)
    cov["covered"] = cov.realised >= cov.bound
    cov.round(4).to_csv(OUT / "monitored_precision.csv", index=False)
    # secondary forecast source: Pooled-tuned alone (primary contrasts only)
    xs = build("Pooled-tuned")
    rs, _, als = run(xs, [("point-margin>0.0", "point-margin", 0.0), ("cs@0.2", "cs", 0.2), ("cs@0.3", "cs", 0.3)],
                     N_BOOT, per_country=False)
    rs.round(4).to_csv(OUT / "procedures_pooled_tuned_source.csv", index=False)
    paired(xs, als, [("point-margin>0.0", "cs@0.2")], N_BOOT).round(4).to_csv(OUT / "paired_pooled_tuned_source.csv", index=False)
    print("done")
