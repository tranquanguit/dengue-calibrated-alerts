"""Sensitivity (not pre-registered): primary Paper 2 contrasts with Paper 1's ENS definition
(arithmetic mean of quantiles per level) instead of the pre-registered log1p average."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import confirm  # noqa: E402

if __name__ == "__main__":
    x = confirm.build("ENS-arith")
    procs = [("point-margin>0.0", "point-margin", 0.0), ("cs@0.2", "cs", 0.2), ("cs@0.3", "cs", 0.3)]
    r, _, al = confirm.run(x, procs, confirm.N_BOOT, per_country=False)
    r.round(4).to_csv(confirm.OUT / "procedures_ens_arith.csv", index=False)
    confirm.paired(x, al, [("point-margin>0.0", "cs@0.2")], confirm.N_BOOT).round(4).to_csv(
        confirm.OUT / "paired_ens_arith.csv", index=False)
    print(r[["procedure", "h", "fdr_bulletin", "fdr_bulletin_hi", "fdr_pooled", "recall", "gap_pooled", "gap_pooled_lo"]].round(3).to_string(index=False))
