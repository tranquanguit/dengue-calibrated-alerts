"""Large-scale climate indices from NASA POWER skin temperature (TS, ~SST over the ocean).

  nino34 : Niño-3.4 box (5N-5S, 170W-120W) mean TS anomaly
  oni    : trailing 3-month mean of nino34 (past months only, usable at the forecast origin)
  dmi    : Dipole Mode Index = west box (10S-10N, 50E-70E) minus east box (10S-0, 90E-110E) anomaly
Anomalies are relative to a fixed 1981-2000 monthly climatology, i.e. before every test year (no look-ahead).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.power import POWER_ZARR  # noqa: E402

BOXES = {"nino34": (-5, 5, -170, -120), "dmi_w": (-10, 10, 50, 70), "dmi_e": (-10, 0, 90, 110)}

if __name__ == "__main__":
    ds = xr.open_zarr(POWER_ZARR, consolidated=True)["TS"].sel(time=slice("1981-01-01", "2026-08-31"))
    series = {}
    for name, (la0, la1, lo0, lo1) in BOXES.items():
        lats, lons = np.arange(la0, la1 + 0.1, 2.5), np.arange(lo0, lo1 + 0.1, 2.5)
        pts = ds.sel(lat=xr.DataArray(np.repeat(lats, len(lons)), dims="p"),
                     lon=xr.DataArray(np.tile(lons, len(lats)), dims="p"), method="nearest").load()
        v = pts.where(pts > -900).mean("p").to_series()
        series[name] = v.resample("MS").mean()
        print(name, len(series[name]), flush=True)
    m = pd.DataFrame(series)
    base = m.loc["1981-01-01":"2000-12-01"]
    clim = base.groupby(base.index.month).mean()
    anom = m - clim.reindex(m.index.month).to_numpy()
    out = pd.DataFrame({"nino34": anom.nino34, "oni": anom.nino34.rolling(3).mean(),
                        "dmi": anom.dmi_w - anom.dmi_e}).rename_axis("date").reset_index()
    out.round(4).to_csv(ROOT / "data/interim/climate_indices.csv", index=False)
    print(out.tail(3))
    # sanity: strong El Niño peaks (1997-98, 2015-16) and La Niña (2010-11)
    o = out.set_index("date").oni
    print("1997-12", round(o["1997-12-01"], 2), "2015-12", round(o["2015-12-01"], 2), "2010-12", round(o["2010-12-01"], 2))
