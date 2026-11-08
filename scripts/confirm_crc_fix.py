"""Re-run the descriptive CRC baseline on the confirmation set after the NaN-score fix (disclosed deviation:
Philippine cells without a 5-year channel have NaN scores, which made the CRC grid all-NaN -> zero alerts)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import confirm  # noqa: E402

if __name__ == "__main__":
    x = confirm.build("ENS")
    r, lt, _ = confirm.run(x, [("crc@0.2", "crc", 0.2), ("crc@0.3", "crc", 0.3)], confirm.N_BOOT, per_country=False)
    r.round(4).to_csv(confirm.OUT / "procedures_crc_fixed.csv", index=False)
    lt.round(4).to_csv(confirm.OUT / "lead_time_crc_fixed.csv", index=False)
    print(r[["procedure", "h", "alerts", "fdr_bulletin", "fdr_pooled", "recall", "onset_recall"]].round(3).to_string(index=False))
