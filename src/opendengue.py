"""OpenDengue (V1.3, CC BY 4.0) sub-national monthly dengue panels for South-East Asia."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .geo import load_natural_earth, match_regions

# country key -> (OpenDengue adm_0_name, spatial level, first year, last year)
OD_COUNTRIES = {
    "thailand": ("THAILAND", "Admin1", 2000, 2022),
    "indonesia": ("INDONESIA", "Admin1", 2007, 2022),
    "cambodia": ("CAMBODIA", "Admin1", 1998, 2010),
    "laos": ("LAO PEOPLE'S DEMOCRATIC REPUBLIC", "Admin1", 1998, 2010),
    "malaysia": ("MALAYSIA", "Admin1", 1998, 2010),
    "philippines": ("PHILIPPINES", "Admin2", 2000, 2010),
    "vietnam": ("VIET NAM", "Admin1", 1998, 2010),
}
# Provinces split or merged during the study period are summed back to stable units
# (Viet Nam 2004 splits and the 2008 Ha Tay -> Ha Noi merger).
MERGE = {"HA TAY": "HA NOI CITY", "DAK NONG": "DAK LAK", "HAU GIANG": "CAN THO CITY", "DIEN BIEN": "LAI CHAU"}
# spellings the fuzzy matcher gets wrong; None = province created during the study
# period (no stable boundary over the series) and dropped
MANUAL = {
    "TRAD": "Trat", "D.I YOGYA": "Yogyakarta", "PREAH SIHANOUK": "Krong Preah Sihanouk",
    "VIENTIANE CAPITAL": "Vientiane [prefecture]", "VIENTIANE CAPITAL [PREFECTURE]": "Vientiane [prefecture]",
    "PAPUA BARAT DAYA": None, "PAPUA PEGUNUNGAN": None, "PAPUA SELATAN": None,
    "PAPUA TENGAH": None, "KALIMANTAN UTARA": None, "XAISOMBOUN": None,
}
# provinces that Natural Earth 10m lumps into regional polygons: seat coordinates
MANUAL_COORDS = {"BAC KAN": (22.147, 105.834), "DONG NAI": (10.957, 106.843), "HUNG YEN": (20.646, 106.051)}
MIN_SCORE = 0.85


def extract_opendengue(spatial_csv: Path, out_csv: Path) -> None:
    """Keep the monthly SE-Asian sub-national records (a few MB) from the 500 MB extract."""
    d = pd.read_csv(spatial_csv, low_memory=False,
                    usecols=["adm_0_name", "adm_1_name", "adm_2_name", "full_name", "calendar_start_date",
                             "calendar_end_date", "dengue_total", "S_res", "T_res", "UUID"])
    keep = pd.Series(False, index=d.index)
    for adm0, level, _, _ in OD_COUNTRIES.values():
        keep |= (d.adm_0_name == adm0) & (d.S_res == level) & (d.T_res == "Month")
    d[keep].to_csv(out_csv, index=False)


def build_coords(od_csv: Path, ne_dbf: Path) -> pd.DataFrame:
    d = pd.read_csv(od_csv)
    ne = load_natural_earth(ne_dbf)
    rows = []
    for key, (adm0, level, _, _) in OD_COUNTRIES.items():
        col = "adm_2_name" if level == "Admin2" else "adm_1_name"
        names = sorted(set(d.loc[d.adm_0_name == adm0, col].replace(MERGE).unique()))
        manual = {k: v for k, v in MANUAL.items() if v}
        auto = [n for n in names if MANUAL.get(n, "") is not None and n not in MANUAL_COORDS]
        m = match_regions(auto, adm0, ne, manual)
        fixed = pd.DataFrame([{"region": n, "ne_name": n.title(), "lat": la, "lon": lo, "score": 1.0}
                              for n, (la, lo) in MANUAL_COORDS.items() if n in names])
        m = pd.concat([m, fixed], ignore_index=True)
        m = m[m.score >= MIN_SCORE].assign(country=key)
        rows.append(m)
    return pd.concat(rows, ignore_index=True)


def load_od_country(od_csv: Path, coords: pd.DataFrame, country: str) -> pd.DataFrame:
    adm0, level, y0, y1 = OD_COUNTRIES[country]
    col = "adm_2_name" if level == "Admin2" else "adm_1_name"
    d = pd.read_csv(od_csv)
    d = d[d.adm_0_name == adm0]
    canon = coords[coords.country == country].set_index("region").ne_name
    d["date"] = pd.to_datetime(d.calendar_start_date).dt.to_period("M").dt.to_timestamp()
    # boundary changes: sum the parts (min_count keeps months where no part reported missing)
    d["unit"] = d[col].replace(MERGE)
    d = d.groupby(["unit", "date"], as_index=False).dengue_total.sum(min_count=1)
    d["region"] = d.unit.map(canon)
    d = d.dropna(subset=["region"])
    # spelling variants of the same province map to one region; if two variants
    # report the same month keep the larger value rather than double counting
    out = d.groupby(["region", "date"], as_index=False).dengue_total.max().rename(columns={"dengue_total": "dengue"})
    return out[(out.date.dt.year >= y0) & (out.date.dt.year <= y1)]
