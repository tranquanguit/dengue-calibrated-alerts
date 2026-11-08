"""Point-forecast metrics and the Diebold-Mariano test with the Harvey-Leybourne-Newbold correction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae_log": float(np.mean(np.abs(np.log1p(y_pred) - np.log1p(y_true)))),
    }


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int) -> tuple[float, float]:
    """H0: equal expected loss. Negative statistic => model A has lower loss."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return np.nan, np.nan
    d_mean = d.mean()
    gamma = [np.sum((d[k:] - d_mean) * (d[: n - k] - d_mean)) / n for k in range(horizon)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    dm = d_mean / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return float(stat), float(p)


def panel_dm(preds: pd.DataFrame, model_a: str, model_b: str, horizon: int) -> tuple[float, float]:
    """DM on the time series of cross-region mean absolute error (one loss per target month)."""
    p = preds[preds.horizon == horizon]
    la = p[p.model == model_a].groupby("target_date").abs_err.mean()
    lb = p[p.model == model_b].groupby("target_date").abs_err.mean()
    common = la.index.intersection(lb.index)
    return dm_test(la[common].to_numpy(), lb[common].to_numpy(), horizon)
