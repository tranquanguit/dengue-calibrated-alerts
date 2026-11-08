"""Nonlinear dependence screening of lagged exogenous candidates (fit on training rows only).

Each candidate x(t-l) is scored against the forecast target y(t+h) after removing
the region x calendar-month seasonal profile from both, so that shared
seasonality is not mistaken for predictive dependence (the "month_province"
control of the original study).
"""
from __future__ import annotations

import dcor
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression


def _deseasonalize(values: pd.Series, region: pd.Series, month: pd.Series) -> pd.Series:
    return values - values.groupby([region, month]).transform("mean")


def score_candidates(
    frame: pd.DataFrame,
    candidates: list[str],
    max_samples: int = 4000,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    e = _deseasonalize(frame.y, frame.region, frame.target_month)
    origin_month = frame.origin.dt.month
    rows = []
    for c in candidates:
        x = _deseasonalize(frame[c], frame.region, origin_month)
        ok = x.notna() & e.notna()
        if ok.sum() < 200 or x[ok].std() < 1e-9:
            continue
        xv, ev = x[ok].to_numpy(float), e[ok].to_numpy(float)
        if len(xv) > max_samples:
            idx = rng.choice(len(xv), max_samples, replace=False)
            xv, ev = xv[idx], ev[idx]
        rows.append({
            "feature": c,
            "n": len(xv),
            "pearson": abs(pearsonr(xv, ev)[0]),
            "spearman": abs(spearmanr(xv, ev)[0]),
            "dcor": dcor.distance_correlation(xv, ev, method="mergesort"),
            "mi": mutual_info_regression(xv.reshape(-1, 1), ev, n_neighbors=3, random_state=random_state)[0],
        })
    s = pd.DataFrame(rows)
    if s.empty:
        return s
    mi_max = s.mi.max()
    s["mi_norm"] = s.mi / mi_max if mi_max > 0 else 0.0
    s["nl_score"] = 0.40 * s.dcor + 0.25 * s.mi_norm + 0.35 * s.spearman
    return s


def select_top_k(
    scores: pd.DataFrame,
    frame: pd.DataFrame,
    k: int,
    by: str = "nl_score",
    redundancy: float = 0.9,
    max_lags_per_var: int = 2,
) -> list[str]:
    """Greedy top-k with a redundancy filter (|Spearman| between selected features)."""
    if scores.empty:
        return []
    chosen: list[str] = []
    per_var: dict[str, int] = {}
    for f in scores.sort_values(by, ascending=False).feature:
        var = f.rsplit("_lag", 1)[0]
        if per_var.get(var, 0) >= max_lags_per_var:
            continue
        if any(abs(frame[f].corr(frame[c], method="spearman")) > redundancy for c in chosen):
            continue
        chosen.append(f)
        per_var[var] = per_var.get(var, 0) + 1
        if len(chosen) == k:
            break
    return chosen
