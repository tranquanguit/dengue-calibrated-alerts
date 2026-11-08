"""Lagged feature construction for direct multi-horizon forecasting on a region-month panel."""
from __future__ import annotations

import numpy as np
import pandas as pd


def complete_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex every region onto a gap-free monthly calendar."""
    dates = pd.date_range(df.date.min(), df.date.max(), freq="MS")
    idx = pd.MultiIndex.from_product([sorted(df.region.unique()), dates], names=["region", "date"])
    return df.set_index(["region", "date"]).reindex(idx).reset_index()


def build_horizon_frame(
    df: pd.DataFrame,
    target: str,
    horizon: int,
    ar_lags: int,
    climate_vars: list[str],
    cross_vars: list[str],
    exo_lags: list[int],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """One row per (region, origin month t); label is log1p(target) at t + horizon.

    Every feature is observed at or before the origin t, so nothing from the
    forecast window enters the inputs.
    """
    df = df.sort_values(["region", "date"]).reset_index(drop=True)
    g = df.groupby("region", sort=False)
    y = np.log1p(df[target])
    cols: dict[str, pd.Series] = {}
    groups: dict[str, list[str]] = {"ar": [], "climate": [], "cross": []}

    ylog = y.groupby(df.region, sort=False)
    for k in range(ar_lags):
        name = f"y_lag{k}"
        cols[name] = ylog.shift(k)
        groups["ar"].append(name)
    for v in climate_vars:
        for lag in exo_lags:
            name = f"{v}_lag{lag}"
            cols[name] = g[v].shift(lag)
            groups["climate"].append(name)
    for v in cross_vars:
        zlog = np.log1p(df[v]).groupby(df.region, sort=False)
        for lag in exo_lags:
            name = f"x_{v}_lag{lag}"
            cols[name] = zlog.shift(lag)
            groups["cross"].append(name)

    out = pd.DataFrame(cols)
    out.insert(0, "region", df.region)
    out.insert(1, "origin", df.date)
    out["target_date"] = df.date + pd.DateOffset(months=horizon)
    out["target_month"] = out.target_date.dt.month
    out["tm_sin"] = np.sin(2 * np.pi * out.target_month / 12)
    out["tm_cos"] = np.cos(2 * np.pi * out.target_month / 12)
    out["region_code"] = out.region.astype("category").cat.codes
    out["y"] = ylog.shift(-horizon)
    # seasonal-naive reference: same calendar month one year before the target
    out["y_snaive"] = ylog.shift(12 - horizon)
    out = out[out.y.notna() & out.y_lag0.notna()].reset_index(drop=True)
    return out, groups


def add_seasonal_profile(frame: pd.DataFrame, train_mask: np.ndarray) -> pd.DataFrame:
    """Region x calendar-month mean of the log target, estimated on training rows only."""
    prof = frame[train_mask].groupby(["region", "target_month"]).y.mean().rename("seas_mean")
    frame = frame.drop(columns="seas_mean", errors="ignore").join(prof, on=["region", "target_month"])
    glob = frame[train_mask].groupby("target_month").y.mean()
    frame["seas_mean"] = frame.seas_mean.fillna(frame.target_month.map(glob))
    return frame
