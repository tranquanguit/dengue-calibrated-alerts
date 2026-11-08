"""Match OpenDengue admin names to Natural Earth admin-1 label points (lat, lon)."""
from __future__ import annotations

import difflib
import re
import unicodedata
from pathlib import Path

import pandas as pd
import shapefile

NE_ADM0 = {  # OpenDengue adm_0_name -> Natural Earth adm0_a3
    "THAILAND": "THA", "INDONESIA": "IDN", "CAMBODIA": "KHM",
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC": "LAO", "MALAYSIA": "MYS", "PHILIPPINES": "PHL",
    "VIET NAM": "VNM",
}
_DROP = r"\b(PROVINCE|PROVINSI|CITY|STATE|NEGERI|KHET|CHANGWAT|DAERAH ISTIMEWA|DKI|KEPULAUAN|NAME)\b"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    s = re.sub(_DROP, " ", s)
    return re.sub(r"[^A-Z]", "", s)


def load_natural_earth(dbf: Path) -> pd.DataFrame:
    r = shapefile.Reader(dbf=open(dbf, "rb"))
    fields = [f[0] for f in r.fields[1:]]
    df = pd.DataFrame(r.records(), columns=fields)
    return df[["adm0_a3", "name", "name_alt", "woe_name", "gn_name", "gns_name", "name_en", "latitude", "longitude"]]


def match_regions(names: list[str], country: str, ne: pd.DataFrame, manual: dict | None = None) -> pd.DataFrame:
    sub = ne[ne.adm0_a3 == NE_ADM0[country]]
    keys = []
    for _, r in sub.iterrows():
        alts = {r["name"], r.woe_name, r.gn_name, r.gns_name, r.name_en, *str(r.name_alt).split("|")}
        for a in alts:
            if a and str(a) != "nan":
                keys.append((norm(a), r["name"], r.latitude, r.longitude))
    rows = []
    for n in names:
        if manual and n in manual:
            target = manual[n]
            hit = sub[sub["name"] == target].iloc[0]
            rows.append({"region": n, "ne_name": target, "lat": hit.latitude, "lon": hit.longitude, "score": 1.0})
            continue
        q = norm(n)
        best = max(keys, key=lambda k: difflib.SequenceMatcher(None, q, k[0]).ratio())
        rows.append({"region": n, "ne_name": best[1], "lat": best[2], "lon": best[3],
                     "score": round(difflib.SequenceMatcher(None, q, best[0]).ratio(), 3)})
    return pd.DataFrame(rows)
