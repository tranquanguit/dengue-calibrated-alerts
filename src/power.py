"""NASA POWER (MERRA-2 based, ~0.5 x 0.625 deg) daily grids -> monthly climate per region.

Read anonymously from the AWS Open Data zarr store; only the chunks covering the
requested grid cells are downloaded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

POWER_ZARR = "https://nasa-power.s3.amazonaws.com/merra2/temporal/power_merra2_daily_temporal_lst.zarr"
DAILY = ["T2M", "T2M_MAX", "T2M_MIN", "T2MDEW", "RH2M", "PRECTOTCORR", "WS2M", "GWETROOT"]
POWER_VARS = [
    "temp_mean", "tmax_mean", "tmin_mean", "tmax_abs", "tmin_abs",
    "dewpoint_mean", "rh_mean", "rh_min",
    "precip_total", "precip_max_daily", "rain_days", "wind_mean",
    "soil_wetness",  # root-zone soil wetness (0-1): proxy for standing water / breeding sites
]


def extract_daily(coords: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """coords: columns region, lat, lon. Returns long daily frame (date, region, <DAILY>)."""
    ds = xr.open_zarr(POWER_ZARR, consolidated=True)[DAILY].sel(time=slice(start, end))
    lat = xr.DataArray(coords.lat.to_numpy(), dims="region")
    lon = xr.DataArray(coords.lon.to_numpy(), dims="region")
    pts = ds.sel(lat=lat, lon=lon, method="nearest").load()
    pts = pts.assign_coords(region=coords.region.to_numpy())
    df = pts.to_dataframe().reset_index().rename(columns={"time": "date"})
    df = df[["date", "region", *DAILY]]
    return df.replace(-999.0, np.nan)


def to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["date"] = d.date.dt.to_period("M").dt.to_timestamp()
    g = d.groupby(["region", "date"])
    m = pd.DataFrame({
        "temp_mean": g.T2M.mean(), "tmax_mean": g.T2M_MAX.mean(), "tmin_mean": g.T2M_MIN.mean(),
        "tmax_abs": g.T2M_MAX.max(), "tmin_abs": g.T2M_MIN.min(),
        "dewpoint_mean": g.T2MDEW.mean(), "rh_mean": g.RH2M.mean(), "rh_min": g.RH2M.min(),
        "precip_total": g.PRECTOTCORR.sum(min_count=20), "precip_max_daily": g.PRECTOTCORR.max(),
        "rain_days": g.PRECTOTCORR.apply(lambda s: float((s >= 1.0).sum()) if s.notna().sum() >= 20 else np.nan),
        "wind_mean": g.WS2M.mean(), "soil_wetness": g.GWETROOT.mean(),
    })
    return m.reset_index()
