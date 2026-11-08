"""Approximate coordinates (lat, lon) of the administrative seat of each region.

Used only to attach the nearest NOAA GSOD weather stations to each region,
so ~0.05 degree precision is sufficient.
"""

SRI_LANKA = {  # 25 districts + Kalmunai RDHS (as reported by the Epidemiology Unit)
    "Ampara": (7.30, 81.67), "Anuradhapura": (8.31, 80.40), "Badulla": (6.99, 81.06),
    "Batticaloa": (7.71, 81.69), "Colombo": (6.93, 79.86), "Galle": (6.05, 80.22),
    "Gampaha": (7.09, 79.99), "Hambanthota": (6.12, 81.12), "Jaffna": (9.66, 80.01),
    "Kalmune": (7.41, 81.83), "Kalutara": (6.58, 79.96), "Kandy": (7.29, 80.63),
    "Kegalle": (7.25, 80.35), "Kilinochchi": (9.39, 80.40), "Kurunegala": (7.49, 80.36),
    "Mannar": (8.98, 79.90), "Matale": (7.47, 80.62), "Matara": (5.95, 80.54),
    "Monaragala": (6.87, 81.35), "Mullaitivu": (9.27, 80.81), "NuwaraEliya": (6.97, 80.78),
    "Polonnaruwa": (7.94, 81.00), "Puttalam": (8.04, 79.84), "Ratnapura": (6.68, 80.40),
    "Trincomalee": (8.59, 81.21), "Vavuniya": (8.75, 80.50),
}

JAPAN = {  # prefectural capitals
    "Hokkaido": (43.06, 141.35), "Aomori": (40.82, 140.74), "Iwate": (39.70, 141.15),
    "Miyagi": (38.27, 140.87), "Akita": (39.72, 140.10), "Yamagata": (38.24, 140.36),
    "Fukushima": (37.75, 140.47), "Ibaraki": (36.34, 140.45), "Tochigi": (36.57, 139.88),
    "Gunma": (36.39, 139.06), "Saitama": (35.86, 139.65), "Chiba": (35.61, 140.12),
    "Tokyo": (35.69, 139.69), "Kanagawa": (35.45, 139.64), "Niigata": (37.90, 139.02),
    "Toyama": (36.70, 137.21), "Ishikawa": (36.59, 136.63), "Fukui": (36.07, 136.22),
    "Yamanashi": (35.66, 138.57), "Nagano": (36.65, 138.18), "Gifu": (35.39, 136.72),
    "Shizuoka": (34.98, 138.38), "Aichi": (35.18, 136.91), "Mie": (34.73, 136.51),
    "Shiga": (35.00, 135.87), "Kyoto": (35.02, 135.76), "Osaka": (34.69, 135.52),
    "Hyogo": (34.69, 135.18), "Nara": (34.69, 135.83), "Wakayama": (34.23, 135.17),
    "Tottori": (35.50, 134.24), "Shimane": (35.47, 133.05), "Okayama": (34.66, 133.93),
    "Hiroshima": (34.40, 132.46), "Yamaguchi": (34.19, 131.47), "Tokushima": (34.07, 134.56),
    "Kagawa": (34.34, 134.04), "Ehime": (33.84, 132.77), "Kochi": (33.56, 133.53),
    "Fukuoka": (33.61, 130.42), "Saga": (33.25, 130.30), "Nagasaki": (32.74, 129.87),
    "Kumamoto": (32.79, 130.74), "Oita": (33.24, 131.61), "Miyazaki": (31.91, 131.42),
    "Kagoshima": (31.56, 130.56), "Okinawa": (26.21, 127.68),
}

BRAZIL = {  # state capitals
    "AC": (-9.97, -67.81), "AL": (-9.67, -35.74), "AP": (0.03, -51.07), "AM": (-3.12, -60.02),
    "BA": (-12.97, -38.51), "CE": (-3.73, -38.53), "DF": (-15.79, -47.88), "ES": (-20.32, -40.34),
    "GO": (-16.69, -49.26), "MA": (-2.53, -44.30), "MT": (-15.60, -56.10), "MS": (-20.47, -54.62),
    "MG": (-19.92, -43.94), "PA": (-1.46, -48.49), "PB": (-7.12, -34.86), "PR": (-25.43, -49.27),
    "PE": (-8.05, -34.88), "PI": (-5.09, -42.80), "RJ": (-22.91, -43.17), "RN": (-5.79, -35.21),
    "RS": (-30.03, -51.23), "RO": (-8.76, -63.90), "RR": (2.82, -60.67), "SC": (-27.60, -48.55),
    "SP": (-23.55, -46.63), "SE": (-10.91, -37.07), "TO": (-10.18, -48.33),
}

# FIPS country codes used by the NOAA ISD station history
COUNTRY = {
    "srilanka": ("CE", SRI_LANKA),
    "japan": ("JA", JAPAN),
    "brazil": ("BR", BRAZIL),
}
# OpenDengue countries: coordinates come from Natural Earth admin-1 label points
# (data/raw/geo/opendengue_region_coords.csv, built by scripts/fetch_data.py)
OD_FIPS = {"thailand": "TH", "indonesia": "ID", "cambodia": "CB", "laos": "LA", "malaysia": "MY", "philippines": "RP",
           "vietnam": "VM"}


def all_countries(coords_csv) -> dict[str, tuple[str, dict]]:
    import pandas as pd

    out = dict(COUNTRY)
    c = pd.read_csv(coords_csv)
    for key, fips in OD_FIPS.items():
        sub = c[c.country == key].drop_duplicates("ne_name")
        out[key] = (fips, {r.ne_name: (r.lat, r.lon) for r in sub.itertuples()})
    return out
