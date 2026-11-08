"""Disease surveillance loaders -> tidy monthly panels (region, date, <disease counts>)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyreadr

# Japanese notifiable diseases kept for the multi-disease study, with the
# column names used in different IDWR reporting periods.
JAPAN_DISEASES = {
    "ehec": ["Enterohemorrhagic E. coli infection (total)", "Enterohemorrhagic Escherichia coli infection (total)"],
    "legionellosis": ["Legionellosis (total)"],
    "scrub_typhus": ["Scrub typhus (total)", "Scrub typhus (Tsutsugamushi disease) (total)"],
    "japanese_spotted_fever": ["Japanese spotted fever (total)"],
    "amebiasis": ["Amebiasis (total)"],
    "hepatitis_a": ["Hepatitis A (total)"],
    "hepatitis_e": ["Hepatitis E (total)"],
    "shigellosis": ["Shigellosis (total)"],
}


def _to_month(df: pd.DataFrame, date_col: str, value_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df[date_col]).dt.to_period("M").dt.to_timestamp()
    out = df.groupby(["region", "date"], as_index=False)[value_cols].sum()
    return out


def load_srilanka(src: Path) -> pd.DataFrame:
    d = next(iter(pyreadr.read_r(src / "srilanka_weekly_data.rda").values()))
    d = d.rename(columns={"district": "region", "cases": "dengue"})
    # weeks are assigned to the month containing their mid-point (start + 3 days)
    d["mid"] = pd.to_datetime(d["start.date"], format="%m/%d/%Y") + pd.Timedelta(days=3)
    out = _to_month(d, "mid", ["dengue"])
    # keep complete months only (the weekly series currently ends mid-May 2026)
    last_full = (d["mid"].max() + pd.Timedelta(days=4)).to_period("M").to_timestamp() - pd.offsets.MonthBegin(1)
    return out[(out.date >= "2007-01-01") & (out.date <= last_full)]


def load_japan(src: Path) -> pd.DataFrame:
    d = pyreadr.read_r(src / "sex_prefecture.rda")["sex_prefecture"]
    d = d[d.prefecture != "Total"].rename(columns={"prefecture": "region"}).copy()
    for name, cols in JAPAN_DISEASES.items():
        d[name] = d[[c for c in cols if c in d.columns]].astype(float).sum(axis=1)
    # IDWR weeks end on Sunday ("date" = week end); use mid-week date
    d["mid"] = pd.to_datetime(d["date"]) - pd.Timedelta(days=3)
    out = _to_month(d, "mid", list(JAPAN_DISEASES))
    # 2006 is the first year in which every selected disease is reported under the current law
    return out[(out.date >= "2006-04-01") & (out.date <= "2024-11-01")]


def load_brazil(src: Path) -> pd.DataFrame:
    d = pd.read_csv(src / "dengue_uf.csv.gz")
    d = d.rename(columns={"uf": "region", "casos": "dengue"})
    d["mid"] = pd.to_datetime(d["date"]) + pd.Timedelta(days=3)
    out = _to_month(d, "mid", ["dengue"])
    last_full = out.date.max() - pd.offsets.MonthBegin(1)
    return out[(out.date >= "2010-01-01") & (out.date <= last_full)]


LOADERS = {"srilanka": load_srilanka, "japan": load_japan, "brazil": load_brazil}


def _od_loader(country):
    from .opendengue import load_od_country

    def load(src: Path) -> pd.DataFrame:
        coords = pd.read_csv(src.parent / "geo" / "opendengue_region_coords.csv")
        return load_od_country(src / "opendengue_sea_month.csv", coords, country)
    return load


for _c in ("thailand", "indonesia", "cambodia", "laos", "malaysia", "philippines", "vietnam"):
    LOADERS[_c] = _od_loader(_c)
