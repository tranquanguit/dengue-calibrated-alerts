"""NOAA GSOD daily station records -> monthly climate variables per region."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CLIMATE_VARS = [
    "temp_mean", "tmax_mean", "tmin_mean", "tmax_abs", "tmin_abs",
    "dewpoint_mean", "rh_mean", "rh_min",
    "precip_total", "precip_max_daily", "rain_days", "wind_mean",
]
MIN_VALID_DAYS = 15


def _f_to_c(x):
    return (x - 32.0) * 5.0 / 9.0


def _rel_humidity(t_c, td_c):
    # Magnus formula
    a, b = 17.625, 243.04
    return 100.0 * np.exp(a * td_c / (b + td_c)) / np.exp(a * t_c / (b + t_c))


def read_station_daily(files: list[Path]) -> pd.DataFrame:
    frames = []
    for f in files:
        d = pd.read_csv(f, usecols=["DATE", "TEMP", "DEWP", "MAX", "MIN", "PRCP", "PRCP_ATTRIBUTES", "WDSP"],
                        dtype={"PRCP_ATTRIBUTES": str})
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["date"] = pd.to_datetime(d.DATE)
    for col, miss in [("TEMP", 9999.9), ("DEWP", 9999.9), ("MAX", 9999.9), ("MIN", 9999.9), ("WDSP", 999.9)]:
        d[col] = d[col].where(d[col] < miss - 1)
    prcp = d.PRCP.where(d.PRCP < 99.9)
    # Flag 'I' = station did not report precipitation that day (value is not a real zero)
    prcp = prcp.where(d.PRCP_ATTRIBUTES.str.strip() != "I")
    out = pd.DataFrame({
        "date": d.date,
        "t": _f_to_c(d.TEMP), "td": _f_to_c(d.DEWP),
        "tmax": _f_to_c(d.MAX), "tmin": _f_to_c(d.MIN),
        "prcp": prcp * 25.4, "wind": d.WDSP * 0.514444,
    })
    out["rh"] = _rel_humidity(out.t, out.td).clip(upper=100)
    return out


def station_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    g = daily.groupby(daily.date.dt.to_period("M"))
    n_t = g.t.count()
    n_p = g.prcp.count()
    days = g.date.apply(lambda s: s.iloc[0].days_in_month)
    m = pd.DataFrame({
        "temp_mean": g.t.mean(), "tmax_mean": g.tmax.mean(), "tmin_mean": g.tmin.mean(),
        "tmax_abs": g.tmax.max(), "tmin_abs": g.tmin.min(),
        "dewpoint_mean": g.td.mean(), "rh_mean": g.rh.mean(), "rh_min": g.rh.min(),
        # totals are rescaled to the full month from the valid reporting days
        "precip_total": g.prcp.mean() * days, "precip_max_daily": g.prcp.max(),
        "rain_days": g.prcp.apply(lambda s: (s >= 1.0).mean()) * days,
        "wind_mean": g.wind.mean(),
    })
    temp_cols = [c for c in CLIMATE_VARS if c.startswith(("t", "dew", "rh"))]
    m.loc[n_t < MIN_VALID_DAYS, temp_cols] = np.nan
    m.loc[n_p < MIN_VALID_DAYS, ["precip_total", "precip_max_daily", "rain_days"]] = np.nan
    m.index = m.index.to_timestamp()
    return m


def build_region_climate(raw_dir: Path, country: str) -> pd.DataFrame:
    smap = pd.read_csv(raw_dir / "station_map.csv", dtype={"station": str})
    smap = smap[smap.country == country]
    cache = {}
    for st in smap.station.unique():
        files = sorted((raw_dir / "gsod").glob(f"*/{st}.csv"))
        if files:
            cache[st] = station_monthly(read_station_daily(files))
    rows = []
    for region, grp in smap.groupby("region"):
        parts, weights = [], []
        for _, r in grp.iterrows():
            if r.station in cache:
                parts.append(cache[r.station])
                weights.append(1.0 / (r.dist_km + 10.0))  # inverse-distance weights
        if not parts:  # no station data for this region: climate left missing
            continue
        num = sum(p.fillna(0) * w for p, w in zip(parts, weights))
        den = sum(p.notna().astype(float) * w for p, w in zip(parts, weights))
        reg = (num / den.replace(0, np.nan))[CLIMATE_VARS]
        reg["region"] = region
        rows.append(reg.rename_axis("date").reset_index())
    return pd.concat(rows, ignore_index=True)
