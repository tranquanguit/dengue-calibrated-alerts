"""Paper 1 confirmatory runs (docs/PREREGISTRATION_P1_confirmatory.md): every design on identical cells.

Block A: SEA historical (KH, LA, MY, PH, TH, VN [+ ID rows in training]), test years 2006-2010.
Block B: recent (TH, ID, LK, BR), test years 2015-2026 (per country where data exist); Block C is
the Sri Lanka 2025-01..2026-04 subset of Block B and is split out in the analysis.

Per (horizon, test year) and design: LightGBM point model (log1p, L2) with hyper-parameters chosen on the
last two training years from a 3x3 grid, plus quantile models with the same hyper-parameters.
Designs: Local-tuned (per country), Pooled-tuned (all block countries), LOCO (other countries only),
climate variants (exploratory), negative-binomial GLM baseline, seasonal naive.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cv2 import Y_ANOM  # noqa: E402
from src.features import add_seasonal_profile, build_horizon_frame, complete_grid  # noqa: E402
from src.power import POWER_VARS  # noqa: E402
from src.stars import anomaly_windows  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "forecasts/unified_grid"
BLOCKS = {
    "A": {"countries": ["cambodia", "laos", "malaysia", "philippines", "thailand", "vietnam"],
          "extra_train": ["indonesia"], "years": [2006, 2007, 2008, 2009, 2010], "start": "1998-01-01", "end": "2010-12-01"},
    "B": {"countries": ["thailand", "indonesia", "srilanka", "brazil"], "extra_train": [],
          "years": list(range(2015, 2027)), "start": "2000-01-01", "end": "2026-12-01",
          # pre-registered test years per country (Sri Lanka 2025-2026 = Block C)
          "country_years": {"thailand": range(2015, 2023), "indonesia": range(2015, 2023),
                            "srilanka": range(2016, 2027), "brazil": range(2017, 2025)}},
    # ablation: Block B without Brazil (Pooled-tuned and LOCO only; Block C excluded)
    "Bnb": {"countries": ["thailand", "indonesia", "srilanka"], "extra_train": [],
            "years": list(range(2015, 2025)), "start": "2000-01-01", "end": "2024-12-01", "lite": True,
            "country_years": {"thailand": range(2015, 2023), "indonesia": range(2015, 2023),
                              "srilanka": range(2016, 2025)}},
}
H = [1, 2, 3, 4, 5, 6]
LEVELS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
GRID = [(nl, mc) for nl in (7, 15, 31) for mc in (20, 50, 100)]
BASEP = dict(n_estimators=300, learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
             reg_lambda=1.0, max_bin=63, force_col_wise=True, n_jobs=1, verbose=-1, random_state=42)
BASE = ["tm_sin", "tm_cos", "seas_mean"]


def load_block(spec):
    frames, panels, groups = {h: [] for h in H}, [], None
    for c in spec["countries"] + spec["extra_train"]:
        df = pd.read_csv(ROOT / "data/processed" / f"{c}_power_monthly.csv", parse_dates=["date"])
        df = complete_grid(df[(df.date >= spec["start"]) & (df.date <= spec["end"])])
        for h in H:
            f, groups = build_horizon_frame(df, "dengue", h, 12, POWER_VARS, [], [0, 1, 2, 3, 4, 5, 6])
            f["country"] = c
            f["region"] = c + ":" + f.region
            frames[h].append(f)
        p = df[["region", "date"]].assign(ylog=np.log1p(df.dengue), region=lambda x, c=c: c + ":" + x.region)
        panels.append(p.assign(country=c))
    idx = pd.read_csv(ROOT / "data/interim/climate_indices.csv", parse_dates=["date"])
    enso = {}
    for lag in (0, 1, 2, 3):
        e = idx.assign(origin=idx.date + pd.DateOffset(months=lag))[["origin", "oni", "dmi"]]
        enso[lag] = e.rename(columns={"oni": f"oni_lag{lag}", "dmi": f"dmi_lag{lag}"})
    out = {}
    for h in H:
        f = pd.concat(frames[h], ignore_index=True)
        for lag in enso:
            f = f.merge(enso[lag], on="origin", how="left")
        f["country_code"] = f.country.astype("category").cat.codes
        out[h] = f
    return out, pd.concat(panels, ignore_index=True), groups


def fit(tr, feats, params, objective="regression", alpha=None):
    kw = dict(BASEP, num_leaves=params[0], min_child_samples=params[1], objective=objective)
    if alpha is not None:
        kw["alpha"] = alpha
    return lgb.LGBMRegressor(**kw).fit(tr[feats], tr.y)


def tune(tr, feats):
    ty = tr.target_date.dt.year
    cut = ty.max() - 1
    inner, val = tr[ty < cut], tr[ty >= cut]
    if len(inner) < 300 or len(val) < 50:
        return (15, 50)
    scores = {g: np.mean(np.abs(fit(inner, feats, g).predict(val[feats]) - val.y)) for g in GRID}
    return min(scores, key=scores.get)


def nb_glm(tr, te):
    """Negative-binomial GLM: region FE + target-month dummies + log1p lags 0-2 (own region).

    Fitted with a small ridge penalty (regions with all-zero training counts otherwise diverge);
    dispersion alpha by moments from a Poisson fit.
    """
    def design(d, cols=None):
        X = pd.concat([pd.get_dummies(d.region, prefix="r", dtype=float),
                       pd.get_dummies(d.target_month, prefix="m", dtype=float),
                       d[["y_lag0", "y_lag1", "y_lag2"]].fillna(0)], axis=1)
        X.insert(0, "const", 1.0)
        return X if cols is None else X.reindex(columns=cols, fill_value=0.0)
    Xtr = design(tr)
    ytr = np.expm1(tr.y).round().clip(0).to_numpy()
    pois = sm.GLM(ytr, Xtr, family=sm.families.Poisson()).fit_regularized(alpha=1e-4, L1_wt=0.0, maxiter=300)
    mu_tr = np.exp(np.clip(Xtr.to_numpy() @ pois.params.to_numpy(), -30, 30))
    alpha = float(np.clip(np.mean(((ytr - mu_tr) ** 2 - mu_tr) / np.maximum(mu_tr, 1e-6) ** 2), 1e-3, 50))
    nb = sm.GLM(ytr, Xtr, family=sm.families.NegativeBinomial(alpha=alpha)).fit_regularized(
        alpha=1e-4, L1_wt=0.0, maxiter=300)
    mu = np.exp(np.clip(design(te, Xtr.columns).to_numpy() @ nb.params.to_numpy(), -30, 16))
    n, p = 1.0 / alpha, 1.0 / (1.0 + alpha * mu)
    q = np.column_stack([stats.nbinom.ppf(a, n, p) for a in LEVELS])
    return mu, q


LOYO = False  # --loyo: leave-one-year-out seasonal profile and anomaly climatology for training rows (robustness check)


def seasonal_profile_loyo(f, Y):
    """Region x target-month mean of y over training years; a training row's own target year is left out."""
    ty = f.target_date.dt.year
    tr = f[ty < Y]
    g = tr.groupby(["region", "target_month", ty[ty < Y].rename("yr")]).y.agg(["sum", "count"])
    tot = g.groupby(level=[0, 1]).sum()
    key = pd.MultiIndex.from_arrays([f.region, f.target_month])
    s_tot = tot["sum"].reindex(key).to_numpy()
    n_tot = tot["count"].reindex(key).to_numpy()
    own = g.reindex(pd.MultiIndex.from_arrays([f.region, f.target_month, ty]))
    own_s = np.where(ty < Y, own["sum"].fillna(0).to_numpy(), 0.0)
    own_n = np.where(ty < Y, own["count"].fillna(0).to_numpy(), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = (s_tot - own_s) / (n_tot - own_n)
    glob = tr.groupby("target_month").y.mean()
    f = f.drop(columns="seas_mean", errors="ignore").copy()
    f["seas_mean"] = np.where(np.isfinite(prof), prof, f.target_month.map(glob).to_numpy())
    return f


def anomaly_windows_loyo(panel, Y):
    """Anomaly windows of ylog with a train-only climatology that leaves out each value's own year."""
    from src.stars import WINDOWS
    p = panel[["region", "date", "ylog"]].copy()
    yr, mo = p.date.dt.year, p.date.dt.month
    tr = yr < Y
    g = p[tr].groupby([p.region[tr], mo[tr], yr[tr]]).ylog.agg(["sum", "count"])
    tot = g.groupby(level=[0, 1]).sum()
    key = pd.MultiIndex.from_arrays([p.region, mo])
    s_tot, n_tot = tot["sum"].reindex(key).to_numpy(), tot["count"].reindex(key).to_numpy()
    own = g.reindex(pd.MultiIndex.from_arrays([p.region, mo, yr]))
    own_s = np.where(tr, own["sum"].fillna(0).to_numpy(), 0.0)
    own_n = np.where(tr, own["count"].fillna(0).to_numpy(), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        base = (s_tot - own_s) / (n_tot - own_n)
    a = pd.Series(p.ylog.to_numpy() - base, index=p.index).groupby(p.region, sort=False)
    out = {"region": p.region, "origin": p.date}
    for l1, l2 in WINDOWS:
        w = l2 - l1 + 1
        out[f"ylog_anom_w{l1}_{l2}"] = a.transform(lambda x, w=w, s0=l1: x.rolling(w, min_periods=w).mean().shift(s0))
    return pd.DataFrame(out)


def run_cell(frames, panel, groups, spec, h, Y):
    if LOYO:
        A = anomaly_windows_loyo(panel, Y)
        f = frames[h].merge(A, on=["region", "origin"], how="left")
        f = seasonal_profile_loyo(f, Y)
    else:
        A = anomaly_windows(panel[["region", "date", "ylog"]], ["ylog"], Y)
        f = frames[h].merge(A, on=["region", "origin"], how="left")
        f = add_seasonal_profile(f, (f.target_date.dt.year < Y).to_numpy())
    train = f[f.target_date.dt.year < Y]
    cy = spec.get("country_years", {})
    allowed = [c for c in spec["countries"] if Y in cy.get(c, spec["years"])]
    test = f[(f.target_date.dt.year == Y) & f.country.isin(allowed)]
    if test.empty:
        return pd.DataFrame()
    ar = groups["ar"] + BASE + Y_ANOM
    clim = groups["climate"]
    enso = [c for c in f.columns if c.startswith(("oni_lag", "dmi_lag"))]
    rows = []

    def emit(design, te, point, q=None, extra=None):
        r = te[["region", "country", "origin", "target_date"]].assign(
            horizon=h, test_year=Y, design=design, y_true=np.expm1(te.y.to_numpy()),
            y_pred=np.clip(point, 0, None))
        if q is not None:
            q = np.sort(np.clip(q, 0, None), axis=1)
            for i, a in enumerate(LEVELS):
                r[f"q{int(a * 100):02d}"] = q[:, i]
        if extra:
            r = r.assign(**extra)
        rows.append(r)

    def lgb_design(design, tr, te, feats, params=None, quantiles=True):
        t0 = time.perf_counter()
        params = params or tune(tr, feats)
        m = fit(tr, feats, params)
        point = np.expm1(m.predict(te[feats]))
        q = None
        if quantiles:
            q = np.column_stack([np.expm1(fit(tr, feats, params, "quantile", a).predict(te[feats])) for a in LEVELS])
        emit(design, te, point, q, {"n_train": len(tr), "params": f"{params[0]}/{params[1]}",
                                    "fit_seconds": time.perf_counter() - t0})
        return params

    if LOYO:  # leave-one-year-out re-run: the designs of the three primary comparisons (with quantiles)
        for c in allowed:
            te_c = test[test.country == c]
            if te_c.empty:
                continue
            tr_c = train[train.country == c]
            if len(tr_c) >= 300:
                lgb_design("Local-tuned", tr_c, te_c, ar)
            lgb_design("LOCO", train[train.country != c], te_c, ar)
        lgb_design("Pooled-tuned", train, test, ar)
        return pd.concat(rows, ignore_index=True)
    if spec.get("lite"):
        for c in allowed:
            te_c = test[test.country == c]
            if len(te_c):
                lgb_design("LOCO", train[train.country != c], te_c, ar)
        lgb_design("Pooled-tuned", train, test, ar)
        return pd.concat(rows, ignore_index=True)
    for c in allowed:
        te_c = test[test.country == c]
        if te_c.empty:
            continue
        emit("SeasonalNaive", te_c, np.expm1(te_c.y_snaive.fillna(te_c.y_lag0).to_numpy()))
        tr_c = train[train.country == c]
        if len(tr_c) >= 300:
            p_loc = lgb_design("Local-tuned", tr_c, te_c, ar)
            lgb_design("Local+clim", tr_c, te_c, ar + clim, p_loc, quantiles=False)
            try:
                mu, q = nb_glm(tr_c, te_c)
                emit("NegBinGLM", te_c, mu, q)
            except Exception as ex:  # noqa: BLE001
                print("nb failed", c, h, Y, ex, flush=True)
        tr_o = train[train.country != c]
        p_loco = lgb_design("LOCO", tr_o, te_c, ar)
        lgb_design("LOCO+clim", tr_o, te_c, ar + clim, p_loco, quantiles=False)
    p_pool = lgb_design("Pooled-tuned", train, test, ar)
    lgb_design("Pooled+clim", train, test, ar + clim, p_pool, quantiles=False)
    lgb_design("Pooled+clim+country", train, test, ar + clim + ["country_code"], p_pool, quantiles=False)
    lgb_design("Pooled+ENSO", train, test, ar + enso, p_pool, quantiles=False)
    lgb_design("Pooled+clim+ENSO", train, test, ar + clim + enso, p_pool, quantiles=False)
    # sensitivity: one-month reporting delay (lag-0 cases unknown at the origin)
    dly = [c for c in ar if c not in ("y_lag0", "ylog_anom_w0_0", "ylog_anom_w0_2")]
    lgb_design("Pooled-delay1", train, test, dly, quantiles=False)
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="*", default=["A", "B"])
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--loyo", action="store_true")
    a = ap.parse_args()
    LOYO = a.loyo
    sfx = "_loyo" if LOYO else ""
    OUT.mkdir(parents=True, exist_ok=True)
    for b in a.blocks:
        spec = BLOCKS[b]
        t0 = time.time()
        frames, panel, groups = load_block(spec)
        parts = OUT / f"_parts_{b}{sfx}"
        parts.mkdir(exist_ok=True)
        todo = [(h, Y) for h in H for Y in spec["years"] if not (parts / f"h{h}_{Y}.parquet").exists()]

        def job(h, Y):
            r = run_cell(frames, panel, groups, spec, h, Y)
            r.to_parquet(parts / f"h{h}_{Y}.parquet", index=False)
            return len(r)

        Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(job)(h, Y) for h, Y in todo)
        allp = [pd.read_parquet(p) for p in sorted(parts.glob("*.parquet"))]
        pd.concat([p for p in allp if len(p)], ignore_index=True).to_parquet(OUT / f"block{b}{sfx}.parquet", index=False)
        print(f"[block {b}] done in {time.time() - t0:.0f}s", flush=True)
