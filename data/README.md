# Data

All inputs are public. Everything here is rebuilt by `pipeline/fetch_data.py` and
`pipeline/build_datasets.py`; the copies are committed for convenience.

| Path | Content | Origin | Licence / terms |
|---|---|---|---|
| `raw/disease/srilanka_weekly_data.rda` | Weekly dengue cases by RDHS district, 2006–2026 | Epidemiology Unit, Ministry of Health Sri Lanka (Weekly Epidemiological Report), compiled by [`denguedatahub`](https://github.com/thiyangt/denguedatahub) @ `86d8070` | package GPL-3; official public reports |
| `raw/disease/sex_prefecture.rda` | Weekly notifiable-disease cases by prefecture, 1999–2024 | JIHS / NIID Infectious Diseases Weekly Report (IDWR), compiled by [`jpinfect`](https://github.com/TomonoriHoshi/jpinfect) @ `d4044d6` | package GPL (≥ 3); official public reports |
| `raw/disease/dengue_uf.csv.gz` | Weekly probable dengue cases by state, 2010–2024 | SINAN via InfoDengue, from [`Mosqlimate-project/sprint-template`](https://github.com/Mosqlimate-project/sprint-template) @ `c3e0761` | GPL-3 |
| `raw/station_map.csv` | Region → GSOD stations (≤ 3 within 120 km) with distances | derived from NOAA ISD station history | — |
| `interim/gsod_station_monthly.csv.gz` | Monthly climate variables per station | [NOAA GSOD](https://registry.opendata.aws/noaa-gsod/) (AWS Open Data) | US Government public domain |
| `processed/{srilanka,japan,brazil}_monthly.csv` | Region × month modelling tables (disease counts + 12 climate variables) | this repository | — |

The daily GSOD files (`raw/gsod/`, ~270 MB, ~3.4k files) and the cloned source
repositories (`raw/sources/`) are not committed; `pipeline/fetch_data.py` downloads them.

Cite the original data producers (Epidemiology Unit Sri Lanka, JIHS/NIID, SINAN/InfoDengue,
NOAA NCEI) and the compiling packages in any publication.
