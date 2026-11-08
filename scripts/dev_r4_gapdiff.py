"""Paired difference of the experienced-FDR gap between CS@q and the matched point rule (dev set)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alerts_core as ac  # noqa: E402
import dev_r4  # noqa: E402

if __name__ == "__main__":
    x = dev_r4.build()
    rows = []
    for h in (1, 2, 3):
        g = x[x.horizon == h].reset_index(drop=True)
        pt = ac.run_procedure(g, "point-margin", q=0.0)
        for q in (0.2, 0.3):
            cs = ac.run_procedure(g, "cs", q=q)
            rows.append({"base": "point-margin>0", "alt": f"cs@{q}", "h": h, **ac.evaluate_paired(g, pt, cs, n_boot=500)})
    r = pd.DataFrame(rows)
    r.round(4).to_csv(dev_r4.da.OUT / "r4_paired_cs_vs_point.csv", index=False)
    print(r[["alt", "h", "d_gap_pooled", "d_gap_pooled_lo", "d_gap_pooled_hi", "d_fdr_pooled", "d_fdr_pooled_lo",
             "d_fdr_pooled_hi", "d_recall", "d_recall_lo", "d_recall_hi"]].round(3).to_string(index=False))
