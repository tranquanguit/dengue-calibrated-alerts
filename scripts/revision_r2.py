"""Additional analysis (set 2): inference for the bulletin-plane reversal (exploratory, not pre-registered).

For each horizon the matched comparator is the point-family member whose bulletin FDR is closest to that of CS@0.2
(rule fixed here, before computing the CIs). Paired country-stratified bootstrap CIs of (CS - member) for recall and
bulletin FDR, with all origins and with burn-in origins excluded."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alerts_core as ac  # noqa: E402
import confirm  # noqa: E402
import revision_r1 as rv  # noqa: E402

if __name__ == "__main__":
    x = confirm.build("ENS")
    fam = [(f"point-{s}>{c}", f"point-{s}", c) for s in ("margin", "z") for c in confirm.CS_]
    rows = []
    for h in (1, 2, 3):
        g = x[x.horizon == h].reset_index(drop=True)
        pv = ac.run_procedure(g, "cs-p")
        ran = g.assign(r=~np.isnan(pv)).groupby(["country", "origin"]).r.transform("any").to_numpy()
        cs = ac.run_procedure(g, "cs", q=0.2)
        for version, mask in (("all origins", np.ones(len(g), bool)), ("burn-in excluded", ran)):
            gg = g[mask].reset_index(drop=True)
            bcs = ac.bulletin_table(gg, cs[mask])
            mcs = ac.metrics_from_bulletins(bcs)
            cand = {n: ac.run_procedure(g, p, q=c)[mask] for n, p, c in fam}
            fb = {n: ac.metrics_from_bulletins(ac.bulletin_table(gg, a))["fdr_bulletin"] for n, a in cand.items()}
            best = min(fb, key=lambda n: abs(fb[n] - mcs["fdr_bulletin"]))
            bpt = ac.bulletin_table(gg, cand[best])
            mpt = ac.metrics_from_bulletins(bpt)
            ci = rv.strat_boot(bpt, bcs)
            rows.append({"version": version, "h": h, "matched_member": best, "fdr_bulletin_cs": mcs["fdr_bulletin"],
                         "fdr_bulletin_pt": mpt["fdr_bulletin"], "recall_cs": mcs["recall"], "recall_pt": mpt["recall"],
                         "d_recall": mcs["recall"] - mpt["recall"], "d_recall_lo": ci["recall_lo"], "d_recall_hi": ci["recall_hi"],
                         "d_fdr_bulletin": mcs["fdr_bulletin"] - mpt["fdr_bulletin"],
                         "d_fdr_bulletin_lo": ci["fdr_bulletin_lo"], "d_fdr_bulletin_hi": ci["fdr_bulletin_hi"]})
        print("h", h, flush=True)
    r = pd.DataFrame(rows)
    r.round(4).to_csv(confirm.OUT / "revision_r1" / "reversal_matched_bulletin_fdr.csv", index=False)
    print(r.round(3).to_string(index=False))
