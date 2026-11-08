"""v2 benchmark: rolling-origin CV comparing STARS with feature-selection baselines and ablations.

Everything that is estimated from data (climatologies, seasonal profile, residuals,
screening, gate) is recomputed inside each fold from training rows only.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .features import add_seasonal_profile
from .screening import score_candidates, select_top_k
from .stars import (Timer, anomaly_windows, fit_lgbm, gate_choice, oof_residuals, residual_booster,
                    stability_select)

BASE = ["tm_sin", "tm_cos", "region_code", "seas_mean"]
Y_ANOM = ["ylog_anom_w0_0", "ylog_anom_w1_1", "ylog_anom_w0_2", "ylog_anom_w3_5"]
CTX = ["seas_mean", "tm_sin", "tm_cos", "region_code", "y_lag0"]


def run_target_fold(frames: dict[int, pd.DataFrame], groups: dict, panel: pd.DataFrame, anom_vars: list[str],
                    test_year: int, cfg: dict) -> tuple[list, list, list, pd.DataFrame | None]:
    params, seed, k = cfg["lgbm"], cfg["random_state"], cfg["top_k"]
    st = cfg["stars"]
    timer = Timer()

    # ---- fold-specific anomaly windows (train-only climatology) ------------------------
    A = timer.run("library", anomaly_windows, panel, anom_vars, test_year)
    lib_anom = [c for c in A.columns if "_anom_w" in c and not c.startswith("ylog_")]
    raw = groups["climate"] + groups["cross"]
    library = raw + lib_anom

    tr, te = {}, {}
    for h, fr in frames.items():
        f = fr.merge(A, on=["region", "origin"], how="left")
        mask = (f.target_date.dt.year < test_year).to_numpy()
        f = add_seasonal_profile(f, mask)
        tr[h], te[h] = f[mask].reset_index(drop=True), f[(f.target_date.dt.year == test_year).to_numpy()].reset_index(drop=True)
    if any(len(v) == 0 for v in te.values()) or any(len(v) < 500 for v in tr.values()):
        return [], [], [], None
    ar = groups["ar"] + BASE + Y_ANOM

    # ---- residual targets ---------------------------------------------------------------
    t0 = time.perf_counter()
    oof = {h: oof_residuals(tr[h], ar, params, seed) for h in frames}
    timer.add("oof_residuals", time.perf_counter() - t0)
    seas_res = {h: (tr[h].y - tr[h].seas_mean).to_numpy() for h in frames}

    # ---- screening (fold-level, shared by all horizons) ---------------------------------
    sel = {}
    kw = dict(n_boot=st["n_boot"], top_q=st["top_q"], pi=st["pi"], k_max=k, seed=seed)
    sel["STARS"], stab_table = timer.run("sel_STARS", stability_select, tr, oof, library, **kw)
    sel["STARS-noStab"], _ = timer.run("sel_STARS-noStab", stability_select, tr, oof, library,
                                       **{**kw, "top_q": k}, bootstrap=False)
    sel["STARS-noResid"], _ = timer.run("sel_STARS-noResid", stability_select, tr, seas_res, library, **kw)
    sel["STARS-noAnom"], _ = timer.run("sel_STARS-noAnom", stability_select, tr, oof, raw, **kw)

    preds, logs, cost = [], [], []
    for h in frames:
        trh, teh = tr[h], te[h]
        y_true = np.expm1(teh.y.to_numpy())
        meta = teh[["region", "origin", "target_date"]].assign(horizon=h, test_year=test_year)

        def add(name, yhat_log, n_feat, fit_s, sel_s=0.0):
            yhat = np.clip(np.expm1(yhat_log), 0, None)
            preds.append(meta.assign(model=name, y_true=y_true, y_pred=yhat, abs_err=np.abs(yhat - y_true)))
            cost.append({"model": name, "horizon": h, "test_year": test_year, "n_features": n_feat,
                         "fit_seconds": fit_s, "select_seconds": sel_s})

        def fit_eval(name, feats, sel_s=0.0):
            t0 = time.perf_counter()
            m = fit_lgbm(trh, feats, trh.y, params, seed)
            p = m.predict(teh[feats])
            add(name, p, len(feats), time.perf_counter() - t0, sel_s)
            return m, p

        add("SeasonalNaive", teh.y_snaive.fillna(teh.y_lag0).to_numpy(), 0, 0.0)
        _, p_ar = fit_eval("AR", ar)
        fit_eval("AR+Raw(all)", ar + raw)
        m_lib, _ = fit_eval("AR+Lib(all)", ar + library)

        # embedded selection: LightGBM gain importance of the all-library model
        t0 = time.perf_counter()
        imp = pd.Series(m_lib.booster_.feature_importance("gain"), index=ar + library)[library]
        gain_sel = imp.nlargest(k).index.tolist()
        fit_eval("AR+Gain(top-k)", ar + gain_sel, sel_s=time.perf_counter() - t0)
        logs.append({"model": "AR+Gain(top-k)", "horizon": h, "test_year": test_year, "features": "|".join(gain_sel)})

        # v1 method: seasonal-residual NL screening on raw lags, per horizon
        t0 = time.perf_counter()
        s = score_candidates(trh, raw, random_state=seed)
        v1_sel = select_top_k(s, trh, k, by="nl_score")
        fit_eval("AR+NL-v1(top-k)", ar + v1_sel, sel_s=time.perf_counter() - t0)
        logs.append({"model": "AR+NL-v1(top-k)", "horizon": h, "test_year": test_year, "features": "|".join(v1_sel)})

        n_h = len(frames)
        for name in ("STARS", "STARS-noStab", "STARS-noResid", "STARS-noAnom"):
            sel_s = (timer.t[f"sel_{name}"] + timer.t["oof_residuals"] * (name != "STARS-noResid")) / n_h
            label = "AR+STARS(joint)" if name == "STARS" else f"AR+{name}"
            _, p_joint = fit_eval(label, ar + sel[name], sel_s=sel_s + timer.t["library"] / n_h)
            logs.append({"model": label, "horizon": h, "test_year": test_year, "features": "|".join(sel[name])})
            if name == "STARS":
                p_stars_joint = p_joint

        # STARS-G: gate between AR / joint / residual correction
        t0 = time.perf_counter()
        choice, val_mae = gate_choice(trh, ar, sel["STARS"], CTX, params, seed)
        if choice == "AR":
            p_g = p_ar
        elif choice == "joint":
            p_g = p_stars_joint
        else:
            rb = residual_booster(trh, oof[h], sel["STARS"], CTX, seed)
            p_g = p_ar + rb.predict(teh[sel["STARS"] + CTX])
        n_g = len(ar) + (len(sel["STARS"]) if choice != "AR" else 0)
        sel_s = (timer.t["sel_STARS"] + timer.t["oof_residuals"] + timer.t["library"]) / n_h
        add("STARS-G", p_g, n_g, time.perf_counter() - t0, sel_s)
        logs.append({"model": "STARS-G", "horizon": h, "test_year": test_year, "features": "|".join(sel["STARS"]),
                     "gate": choice, **{f"val_mae_{a}": b for a, b in val_mae.items()}})

    stab_table = stab_table.assign(test_year=test_year).reset_index(names="feature")
    return preds, logs, cost, stab_table
