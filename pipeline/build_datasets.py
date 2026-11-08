"""Build the processed region-month modelling tables (disease + climate)."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.climate import build_region_climate, read_station_daily, station_monthly  # noqa: E402
from src.data_sources import LOADERS  # noqa: E402

RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"



def export_station_monthly() -> None:
    """Station x month climate table (compact, committed copy of the GSOD input)."""
    smap = pd.read_csv(RAW / "station_map.csv", dtype={"station": str})
    frames = []
    for st in smap.station.unique():
        files = sorted((RAW / "gsod").glob(f"*/{st}.csv"))
        if files:
            m = station_monthly(read_station_daily(files)).round(3)
            frames.append(m.rename_axis("date").reset_index().assign(station=st))
    pd.concat(frames).to_csv(ROOT / "data" / "interim" / "gsod_station_monthly.csv.gz", index=False)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "interim").mkdir(parents=True, exist_ok=True)
    export_station_monthly()
    for country, loader in LOADERS.items():
        dis = loader(RAW / "disease")
        smap = pd.read_csv(RAW / "station_map.csv")
        if country in set(smap.country):  # GSOD station climate (legacy v1 tables)
            clim = build_region_climate(RAW, country)
            df = dis.merge(clim, on=["region", "date"], how="left").sort_values(["region", "date"])
            df.to_csv(OUT / f"{country}_monthly.csv", index=False)
        power = ROOT / "data" / "interim" / "power_region_monthly.csv.gz"
        if power.exists():
            pw = pd.read_csv(power, parse_dates=["date"])
            pw = pw[pw.country == country].drop(columns="country")
            dp = dis.merge(pw, on=["region", "date"], how="left").sort_values(["region", "date"])
            dp.to_csv(OUT / f"{country}_power_monthly.csv", index=False)
        print(f"== {country}: {dis.region.nunique()} regions, {dis.date.min():%Y-%m}..{dis.date.max():%Y-%m}, rows={len(dis)}")
