"""Extract NASA POWER monthly climate for every region of every country."""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.fetch_data import YEARS  # noqa: E402
from src.power import extract_daily, to_monthly  # noqa: E402
from src.regions import all_countries  # noqa: E402

if __name__ == "__main__":
    out = ROOT / "data" / "interim"
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for country, (_, regions) in all_countries(ROOT / "data/raw/geo/opendengue_region_coords.csv").items():
        t0 = time.time()
        y0, y1 = YEARS[country]
        coords = pd.DataFrame([(r, la, lo) for r, (la, lo) in regions.items()], columns=["region", "lat", "lon"])
        m = to_monthly(extract_daily(coords, f"{y0}-01-01", f"{y1}-12-31")).assign(country=country)
        frames.append(m)
        print(f"[power] {country}: {coords.shape[0]} regions, {len(m)} region-months, "
              f"missing={m.drop(columns=['region', 'date', 'country']).isna().mean().mean():.3f} ({time.time() - t0:.0f}s)",
              flush=True)
    pd.concat(frames).round(4).to_csv(out / "power_region_monthly.csv.gz", index=False)
