"""STARS: Stability-selected, Anomaly-window, Residual-conditional Screening (+ validation Gate).

Pipeline for one (target, test fold), all steps fitted on training rows only:

1. Candidate library   raw lags x(t-l) and *anomaly windows*: the mean, over lags l1..l2,
                       of x minus its region x calendar-month climatology (DLNM-style
                       distributed-lag aggregation, made seasonality-free).
2. Residual relevance  each candidate is scored against the out-of-fold residual of the
                       autoregressive model (information *beyond* the target's own history),
                       using 0.5 dCor + 0.5 |Spearman|.
3. Stability           scores are recomputed on B temporal block bootstraps (calendar years
                       resampled with replacement) and pooled over all horizons; a candidate
                       is kept if it is in the top-q of >= pi of the bootstraps, followed by a
                       redundancy filter. One feature set serves every horizon.
4. Gate                an inner validation on the last training years chooses between AR,
                       AR + selected features (joint model) and AR + residual-correction
                       booster; exogenous information is only used if it helps.
"""
from __future__ import annotations

import time

import dcor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata

WINDOWS = [(l, l) for l in range(7)] + [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (0, 5)]


# ----------------------------------------------------------------------------- library
def anomaly_windows(panel: pd.DataFrame, variables: list[str], train_end_year: int) -> pd.DataFrame:
    """Anomaly-window features at each (region, date) using train-only climatology.

    `panel` is the gap-free region-month panel; cross-disease variables are expected
    already on log1p scale.
    """
    p = panel[["region", "date", *variables]].copy()
    month = p.date.dt.month
    train = p.date.dt.year < train_end_year
    out = {"region": p.region, "origin": p.date}
    for v in variables:
        clim = p[train].groupby([p.region[train], month[train]])[v].mean()
        base = pd.Series(clim.reindex(pd.MultiIndex.from_arrays([p.region, month])).to_numpy(), index=p.index)
        a = (p[v] - base).groupby(p.region, sort=False)
        for l1, l2 in WINDOWS:
            width = l2 - l1 + 1
            s = a.transform(lambda x, w=width, s0=l1: x.rolling(w, min_periods=w).mean().shift(s0))
            out[f"{v}_anom_w{l1}_{l2}"] = s
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- relevance
def relevance_matrix(X: np.ndarray, e: np.ndarray) -> np.ndarray:
    """0.5 dCor + 0.5 |Spearman| of every column of X with e (complete cases assumed)."""
    rx = rankdata(X, axis=0)
    re = rankdata(e)
    rx = (rx - rx.mean(0)) / (rx.std(0) + 1e-12)
    re = (re - re.mean()) / (re.std() + 1e-12)
    rho = np.abs(rx.T @ re) / len(e)
    dc = np.array([dcor.distance_correlation(X[:, j], e, method="mergesort") if X[:, j].std() > 1e-9 else 0.0
                   for j in range(X.shape[1])])
    return 0.5 * dc + 0.5 * rho


def oof_residuals(train: pd.DataFrame, feats: list[str], params: dict, seed: int, n_blocks: int = 3) -> np.ndarray:
    """Out-of-fold AR residuals using contiguous blocks of target years."""
    years = np.sort(train.target_date.dt.year.unique())
    blocks = np.array_split(years, n_blocks)
    pred = np.full(len(train), np.nan)
    ty = train.target_date.dt.year.to_numpy()
    for b in blocks:
        hold = np.isin(ty, b)
        m = fit_lgbm(train[~hold], feats, train.y[~hold], params, seed)
        pred[hold] = m.predict(train.loc[hold, feats])
    return train.y.to_numpy() - pred


def stability_select(
    frames: dict[int, pd.DataFrame],
    residuals: dict[int, np.ndarray],
    candidates: list[str],
    n_boot: int = 20,
    top_q: int = 20,
    pi: float = 0.5,
    k_max: int = 10,
    n_sub: int = 1000,
    seed: int = 42,
    bootstrap: bool = True,
) -> tuple[list[str], pd.DataFrame]:
    """Horizon-pooled stability selection with a temporal (calendar-year) block bootstrap.

    Rows of all horizons are stacked (candidate at origin t, residual at t+h), so one
    relevance evaluation per bootstrap covers the whole forecast window.
    """
    rng = np.random.default_rng(seed)
    X_parts, e_parts, y_parts = [], [], []
    for h, fr in frames.items():
        X_parts.append(fr[candidates].to_numpy(float))
        e_parts.append(residuals[h])
        y_parts.append(fr.target_date.dt.year.to_numpy())
    X, e, ty = np.vstack(X_parts), np.concatenate(e_parts), np.concatenate(y_parts)
    ok = ~(np.isnan(X).any(1) | np.isnan(e))
    X, e, ty = X[ok], e[ok], ty[ok]
    years = np.unique(ty)
    by_year = {y: np.flatnonzero(ty == y) for y in years}
    n_rounds = n_boot if bootstrap else 1
    freq = np.zeros(len(candidates))
    mean_score = np.zeros(len(candidates))
    for _ in range(n_rounds):
        if bootstrap:
            idx = np.concatenate([by_year[y] for y in rng.choice(years, len(years), replace=True)])
        else:
            idx = np.arange(len(e))
        if len(idx) > n_sub:
            idx = rng.choice(idx, n_sub, replace=False)
        score = relevance_matrix(X[idx], e[idx])
        mean_score += score / n_rounds
        freq[np.argsort(-score)[:top_q]] += 1.0 / n_rounds
    table = pd.DataFrame({"freq": freq, "score": mean_score}, index=candidates)
    pool = table[table.freq >= pi] if bootstrap else table.nlargest(top_q, "score")
    ranked = pool.sort_values(["freq", "score"], ascending=False).index.tolist()
    return _redundancy_filter(ranked, next(iter(frames.values())), k_max), table


def _redundancy_filter(ranked: list[str], frame: pd.DataFrame, k_max: int, rho_max: float = 0.9,
                       per_var: int = 2) -> list[str]:
    chosen: list[str] = []
    count: dict[str, int] = {}
    for f in ranked:
        var = f.split("_anom_w")[0].rsplit("_lag", 1)[0]
        if count.get(var, 0) >= per_var:
            continue
        if any(abs(frame[f].corr(frame[c], method="spearman")) > rho_max for c in chosen):
            continue
        chosen.append(f)
        count[var] = count.get(var, 0) + 1
        if len(chosen) == k_max:
            break
    return chosen


# ----------------------------------------------------------------------------- models
def fit_lgbm(X: pd.DataFrame, feats: list[str], y, params: dict, seed: int) -> lgb.LGBMRegressor:
    m = lgb.LGBMRegressor(**params, random_state=seed, n_jobs=1, verbose=-1)
    cat = ["region_code"] if "region_code" in feats else "auto"
    m.fit(X[feats], y, categorical_feature=cat)
    return m


RESID_PARAMS = dict(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0)


def residual_booster(train: pd.DataFrame, resid: np.ndarray, exo: list[str], ctx: list[str], seed: int):
    ok = ~np.isnan(resid)
    return fit_lgbm(train[ok], exo + ctx, resid[ok], RESID_PARAMS, seed)


def gate_choice(train: pd.DataFrame, ar_feats: list[str], exo: list[str], ctx: list[str], params: dict,
                seed: int, val_years: int = 2, margin: float = 0.005) -> tuple[str, dict]:
    """Inner validation on the last `val_years` training years: AR vs joint vs residual."""
    ty = train.target_date.dt.year
    cut = ty.max() - val_years + 1
    inner, val = train[ty < cut], train[ty >= cut]
    if not exo or len(val) < 50 or len(inner) < 300:
        return "AR", {}
    ar = fit_lgbm(inner, ar_feats, inner.y, params, seed)
    p_ar = ar.predict(val[ar_feats])
    joint = fit_lgbm(inner, ar_feats + exo, inner.y, params, seed)
    p_joint = joint.predict(val[ar_feats + exo])
    r_in = oof_residuals(inner, ar_feats, params, seed, n_blocks=3)
    rb = residual_booster(inner, r_in, exo, ctx, seed)
    p_res = p_ar + rb.predict(val[exo + ctx])
    mae = {k: float(np.mean(np.abs(v - val.y.to_numpy()))) for k, v in
           {"AR": p_ar, "joint": p_joint, "residual": p_res}.items()}
    best = min(("joint", "residual"), key=mae.get)
    return (best if mae[best] < mae["AR"] * (1 - margin) else "AR"), mae


class Timer:
    def __init__(self):
        self.t = {}

    def add(self, key, seconds):
        self.t[key] = self.t.get(key, 0.0) + seconds

    def run(self, key, fn, *a, **k):
        t0 = time.perf_counter()
        out = fn(*a, **k)
        self.add(key, time.perf_counter() - t0)
        return out
