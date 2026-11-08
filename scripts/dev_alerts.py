"""Development-set study for Paper 2 (dev countries only: TH, ID, KH, LA, MY).

Forecast sources (exploratory, from the pilot): LightGBM local quantiles, Chronos-2-cross quantiles,
and their quantile average. Confirmation countries (PH, VN, LK, BR) are NOT evaluated here.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import alerts_core as ac  # noqa: E402

DEV = ["thailand", "indonesia", "cambodia", "laos", "malaysia"]
Q = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]
KEY = ["target", "region", "origin", "target_date", "horizon", "test_year"]
OUT = ROOT / "results/dev"
STATE_CUT = 0.2  # origin month counts as "epidemic state" when >= 20 % of provinces are above their channel


def load(c):
    lq = pd.read_parquet(ROOT / "results/lgbm_quantiles" / f"{c}.parquet")
    ch = pd.read_parquet(ROOT / "forecasts/chronos2" / f"{c}.parquet")
    ch = ch[ch.model == "Chronos2-cross"][KEY + Q + ["y_true"]]
    d = lq.merge(ch, on=KEY, suffixes=("_lgb", "_chr"))
    d = d[d.y_true.notna() & (d.horizon <= 3)].copy()
    for c_ in Q:
        d[f"{c_}_avg"] = np.expm1((np.log1p(d[f"{c_}_lgb"]) + np.log1p(d[f"{c_}_chr"])) / 2)
    panel = pd.read_csv(ROOT / "data/processed" / f"{c}_power_monthly.csv", parse_dates=["date"])
    panel = panel[["region", "date", "dengue"]].rename(columns={"dengue": "cases"}).dropna()
    return d.assign(country=c), panel


def prepare(window=5, k=2.0, exclude_max=True, floor=5):
    frames = []
    for c in DEV:
        d, panel = load(c)
        thr = ac.endemic_channel(panel, window, k, exclude_max)
        flags = ac.onset_flags(panel, thr, floor)
        d = ac.add_events(d, thr, floor)
        d = d.merge(flags[["region", "date", "onset"]].rename(columns={"date": "target_date"}),
                    on=["region", "target_date"], how="left")
        pers = flags[["region", "date", "event"]].rename(columns={"date": "origin", "event": "persist"})
        d = d.merge(pers, on=["region", "origin"], how="left")
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    base = prepare()
    print("dev rows", len(base), "event rate", round(base.event.mean(), 3), "onset share of events",
          round(base.onset.sum() / base.event.sum(), 3))
    # 1) score choice by PR-AUC (per horizon, per source)
    rows = []
    for src in ("lgb", "chr", "avg"):
        d = ac.scores(base, [f"{c}_{src}" for c in Q]) if False else None
        x = base.copy()
        for c in Q:
            x[c] = x[f"{c}_{src}"]
        x = ac.scores(x, Q)
        for h in (1, 2, 3):
            g = x[x.horizon == h]
            rows.append({"source": src, "h": h, "base_rate": g.event.mean(),
                         **{s: average_precision_score(g.event, g[s]) for s in ("s_margin", "s_zmargin", "s_lognorm")}})
    sc = pd.DataFrame(rows)
    sc.round(4).to_csv(OUT / "score_prauc.csv", index=False)
    print(sc.round(3).to_string(index=False))
    # 2) procedures on the chosen source/score (avg, zmargin unless the table says otherwise)
    best = sc.groupby("source")[["s_margin", "s_zmargin", "s_lognorm"]].mean()
    src, score = np.unravel_index(np.argmax(best.to_numpy()), best.shape)
    src, score = best.index[src], best.columns[score]
    print("chosen source/score:", src, score)
    x = base.copy()
    for c in Q:
        x[c] = x[f"{c}_{src}"]
    x = ac.scores(x, Q).reset_index(drop=True)
    # epidemic state of the country at each origin (known at the origin): share of provinces above channel
    sh = x.groupby(["country", "origin"]).persist.mean().rename("share_above")
    x = x.join(sh, on=["country", "origin"])
    x["state"] = (x.share_above >= STATE_CUT).astype(int)
    res, leads = [], []
    procs = [("point", {}), ("persistence", {}), ("topk", {"score": score})]
    for q in (0.2, 0.3):
        procs += [(f"crc@{q}", {"proc": "crc", "q": q, "score": score}),
                  (f"cs@{q}", {"proc": "cs", "q": q, "score": score}),
                  (f"cs-ltt@{q}", {"proc": "cs-ltt", "q": q, "score": score}),
                  (f"cs-mondrian@{q}", {"proc": "cs-mondrian", "q": q, "score": score}),
                  (f"cs-aci@{q}", {"proc": "cs-aci", "q": q, "score": score, "gamma": 0.5}),
                  (f"ltt-mfdr@{q}", {"proc": "ltt-mfdr", "q": q, "score": score})]
    for name, kw in procs:
        kw = dict(kw)
        proc = kw.pop("proc", name.split("@")[0])
        alerts_all = np.zeros(len(x), bool)
        for h in (1, 2, 3):
            m = (x.horizon == h).to_numpy()
            g = x[m].reset_index(drop=True)
            a = ac.run_procedure(g, proc, **kw)
            alerts_all[np.flatnonzero(m)] = a
            ev = ac.evaluate(g, a, kw.get("q"), n_boot=500)
            res.append({"procedure": name, "h": h, **ev})
        lt = ac.lead_time(x, alerts_all)
        leads.append({"procedure": name, "onsets": len(lt), "detected_any_h": float((lt.lead > 0).mean()),
                      "median_lead_detected": float(lt.lead[lt.lead > 0].median()) if (lt.lead > 0).any() else np.nan,
                      "mean_lead_all": float(lt.lead.mean())})
        print(name, "done", flush=True)
    r = pd.DataFrame(res)
    r.round(4).to_csv(OUT / "procedures_dev.csv", index=False)
    pd.DataFrame(leads).round(4).to_csv(OUT / "lead_time_dev.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["procedure", "h", "alerts", "fdr_bulletin", "fdr_given_alert", "fdr_given_alert_lo", "fdr_given_alert_hi",
            "fdr_pooled", "fdr_pooled_lo", "fdr_pooled_hi", "recall", "recall_lo", "recall_hi", "onset_recall",
            "share_bulletins_alerting"]
    print(r[cols].round(3).to_string(index=False))
    print(pd.DataFrame(leads).round(3).to_string(index=False))
    # event-definition sensitivity and per-country event rates
    sens = []
    for k in (1.5, 2.0, 2.5, 3.0):
        b = prepare(k=k)
        for c, g in b.groupby("country"):
            sens.append({"k": k, "country": c, "event_rate": g.event.mean(), "onset_share": g.onset.sum() / max(g.event.sum(), 1)})
    pd.DataFrame(sens).round(4).to_csv(OUT / "event_rates_by_k.csv", index=False)
    print(pd.DataFrame(sens).pivot(index="country", columns="k", values="event_rate").round(3))
