"""Download all raw data used by this pipeline.

Disease data come from public GitHub repositories (pinned commits):
  - Sri Lanka : thiyangt/denguedatahub  (weekly dengue by district, Epidemiology Unit WER)
  - Japan     : TomonoriHoshi/jpinfect   (weekly notifiable diseases by prefecture, JIHS/NIID IDWR)
  - Brazil    : Mosqlimate-project/sprint-template (weekly dengue by state, InfoDengue/SINAN)
Climate data come from NOAA Global Summary of the Day (GSOD) on the AWS Open Data registry.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.opendengue import build_coords, extract_opendengue  # noqa: E402
from src.regions import all_countries  # noqa: E402

RAW = ROOT / "data" / "raw"
SOURCES = {
    "denguedatahub": ("https://github.com/thiyangt/denguedatahub.git", "86d8070966644ee0b4e5b937c3792d5719390b38"),
    "jpinfect": ("https://github.com/TomonoriHoshi/jpinfect.git", "d4044d6b49b2196657ea7434a13abe40ccc88bf7"),
    "sprint-template": ("https://github.com/Mosqlimate-project/sprint-template.git", "c3e07614e2cfbbd92c076237f555703d07d5c858"),
    "opendengue": ("https://github.com/OpenDengue/master-repo.git", "f2232b8f1c82bfc38006d8ecebd93677b9ac75d0"),
}
NE_ADMIN1 = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/10m_cultural/ne_10m_admin_1_states_provinces.dbf"
DISEASE_FILES = [
    "denguedatahub/data/srilanka_weekly_data.rda",
    "jpinfect/data/sex_prefecture.rda",
    "sprint-template/scores/data/dengue_uf.csv.gz",
]
ISD_HISTORY = "https://noaa-isd-pds.s3.amazonaws.com/isd-history.csv"
GSOD = "https://noaa-gsod-pds.s3.amazonaws.com/{year}/{station}.csv"
YEARS = {"srilanka": (2006, 2026), "japan": (2006, 2025), "brazil": (2009, 2025), "thailand": (1999, 2022),
         "indonesia": (2006, 2022), "cambodia": (1997, 2010), "laos": (1997, 2010), "malaysia": (1997, 2010),
         "philippines": (1999, 2010), "vietnam": (1997, 2010)}


def clone_sources() -> None:
    dest = RAW / "sources"
    dest.mkdir(parents=True, exist_ok=True)
    for name, (url, sha) in SOURCES.items():
        path = dest / name
        if not path.exists():
            subprocess.run(["git", "clone", "-q", url, str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "-q", sha], check=True)
        print(f"[source] {name} @ {sha[:10]}")
    # keep only the files the pipeline reads (committed under data/raw/disease/)
    disease = RAW / "disease"
    disease.mkdir(parents=True, exist_ok=True)
    for rel in DISEASE_FILES:
        shutil.copy2(dest / rel, disease / Path(rel).name)
    od_csv = disease / "opendengue_sea_month.csv"
    if not od_csv.exists():
        with zipfile.ZipFile(dest / "opendengue/data/releases/V1.3/Spatial_extract_V1_3.zip") as z:
            z.extractall(dest / "opendengue_extract")
        extract_opendengue(dest / "opendengue_extract/Spatial_extract_V1_3.csv", od_csv)
    geo = RAW / "geo"
    geo.mkdir(parents=True, exist_ok=True)
    if not (geo / "ne_admin1.dbf").exists():
        (geo / "ne_admin1.dbf").write_bytes(requests.get(NE_ADMIN1, timeout=120).content)
    build_coords(od_csv, geo / "ne_admin1.dbf").to_csv(geo / "opendengue_region_coords.csv", index=False)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def select_stations(n_per_region: int = 3, max_km: float = 120.0) -> pd.DataFrame:
    """Attach up to `n_per_region` long-running GSOD stations to each region seat."""
    hist_path = RAW / "isd-history.csv"
    if not hist_path.exists():
        hist_path.write_bytes(requests.get(ISD_HISTORY, timeout=60).content)
    hist = pd.read_csv(hist_path, dtype=str)
    hist = hist.dropna(subset=["LAT", "LON"])
    hist["LAT"], hist["LON"] = hist["LAT"].astype(float), hist["LON"].astype(float)
    hist["station"] = hist["USAF"] + hist["WBAN"]
    rows = []
    for country, (fips, regions) in all_countries(RAW / "geo" / "opendengue_region_coords.csv").items():
        y0, y1 = YEARS[country]
        cand = hist[(hist.CTRY == fips) & (hist.BEGIN <= f"{y0 + 1}0101") & (hist.END >= f"{y1}0101")]
        cand = cand[cand.USAF != "999999"]
        for region, (lat, lon) in regions.items():
            d = haversine_km(lat, lon, cand.LAT.values, cand.LON.values)
            order = np.argsort(d)
            picked = [i for i in order[:n_per_region] if d[i] <= max_km] or [order[0]]
            for i in picked:
                rows.append({"country": country, "region": region, "station": cand.station.iloc[i],
                             "station_name": cand["STATION NAME"].iloc[i], "dist_km": round(float(d[i]), 1)})
    out = pd.DataFrame(rows)
    out.to_csv(RAW / "station_map.csv", index=False)
    return out


def _download(task):
    year, station, path = task
    if path.exists():
        return True
    r = requests.get(GSOD.format(year=year, station=station), timeout=60)
    if r.status_code != 200:
        return False
    path.write_bytes(r.content)
    return True


def fetch_gsod(stations: pd.DataFrame, workers: int = 16) -> None:
    tasks = []
    for country, grp in stations.groupby("country"):
        y0, y1 = YEARS[country]
        for st in grp.station.unique():
            for y in range(y0, y1 + 1):
                p = RAW / "gsod" / str(y) / f"{st}.csv"
                p.parent.mkdir(parents=True, exist_ok=True)
                tasks.append((y, st, p))
    with ThreadPoolExecutor(workers) as ex:
        ok = list(ex.map(_download, tasks))
    print(f"[gsod] {sum(ok)}/{len(ok)} station-year files available")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-climate", action="store_true")
    args = ap.parse_args()
    clone_sources()
    if not args.skip_climate:
        st = select_stations()
        print(st.groupby("country").agg(regions=("region", "nunique"), stations=("station", "nunique")))
        fetch_gsod(st)
