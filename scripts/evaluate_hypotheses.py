"""Apply the frozen decision rules (prereg v3 + v3.1) to results/confirmation/ and write hypotheses.md/.csv."""
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).resolve().parents[1] / "results/confirmation"
r = pd.read_csv(D / "procedures.csv")
p = r[r.country == "pooled"].set_index(["procedure", "h"])
pr = pd.read_csv(D / "paired.csv")
H = (1, 2, 3)
out, lines = [], []


def rec(name, h, value, verdict):
    out.append({"hypothesis": name, "h": h, "value": value, "verdict": verdict})


# H1 (A1 three-way)
cs = [p.loc[("cs@0.2", h)] for h in H]
pt = [c.fdr_bulletin for c in cs]
hi = [c.fdr_bulletin_hi for c in cs]
h1 = "holds" if all(x <= 0.2 for x in hi) else ("violated" if any(x > 0.2 for x in pt) else "not confirmed")
for h, c in zip(H, cs):
    rec("H1 bulletin FDR CS@0.2", h, f"{c.fdr_bulletin:.3f} [{c.fdr_bulletin_lo:.3f}, {c.fdr_bulletin_hi:.3f}]", "")
# H2 + A2
lo = [c.gap_pooled_lo for c in cs]
h2 = all(x > 0 for x in lo)
material = all(x > 0.05 for x in lo)
for h, c in zip(H, cs):
    rec("H2 gap pooled-bulletin CS@0.2", h, f"{c.gap_pooled:.3f} [{c.gap_pooled_lo:.3f}, {c.gap_pooled_hi:.3f}]", "")
    rec("H2 gap' FDR|R>0-bulletin CS@0.2", h, f"{c.gap_pos:.3f} [{c.gap_pos_lo:.3f}, {c.gap_pos_hi:.3f}]", "")
b = pr[(pr.base == "point-margin>0.0") & (pr.alt == "cs@0.2")].set_index("h")
h2b = all(b.loc[h, "d_gap_pooled_lo"] > 0 for h in H)
for h in H:
    rec("H2b gap(CS@0.2)-gap(point-margin>0)", h,
        f"{b.loc[h, 'd_gap_pooled']:.3f} [{b.loc[h, 'd_gap_pooled_lo']:.3f}, {b.loc[h, 'd_gap_pooled_hi']:.3f}]", "")
# H4 dominance + paired CI vs dominating member; onset criterion (A3)
fam = [i for i in p.index.get_level_values(0).unique() if i.startswith("point-")]
dom_h, onset_ok, comparable = 0, 0, 0
for h in H:
    c = p.loc[("cs@0.2", h)]
    doms = [f for f in fam if p.loc[(f, h)].fdr_pooled <= c.fdr_pooled and p.loc[(f, h)].recall >= c.recall
            and (p.loc[(f, h)].fdr_pooled < c.fdr_pooled or p.loc[(f, h)].recall > c.recall)]
    if doms:
        dom_h += 1
        best = min(doms, key=lambda f: p.loc[(f, h)].fdr_pooled)
        q = pr[(pr.base == best) & (pr.alt == "cs@0.2") & (pr.h == h)]
        q = q.iloc[0] if len(q) else pr[(pr.base == "point-margin>0.0") & (pr.alt == "cs@0.2") & (pr.h == h)].iloc[0]
        rec("H4 CS@0.2 dominated by", h, f"{best}: dFDRpool {q.d_fdr_pooled:.3f} [{q.d_fdr_pooled_lo:.3f}, {q.d_fdr_pooled_hi:.3f}], "
            f"dRecall {q.d_recall:.3f} [{q.d_recall_lo:.3f}, {q.d_recall_hi:.3f}] (CS - point)", "dominated")
    else:
        rec("H4 CS@0.2 dominated by", h, "none", "not dominated")
    near = [f for f in fam if abs(p.loc[(f, h)].fdr_pooled - c.fdr_pooled) <= 0.03]
    if near:
        comparable += 1
        m = min(near, key=lambda f: abs(p.loc[(f, h)].fdr_pooled - c.fdr_pooled))
        better = c.onset_recall > p.loc[(m, h)].onset_recall
        onset_ok += better
        rec("H4 onset criterion", h, f"CS {c.onset_recall:.3f} vs {m} {p.loc[(m, h)].onset_recall:.3f} "
            f"(pooled FDR {c.fdr_pooled:.3f} vs {p.loc[(m, h)].fdr_pooled:.3f})", "CS higher" if better else "CS not higher")
    else:
        rec("H4 onset criterion", h, "no point member within +-0.03", "not comparable")
h4 = dom_h >= 2
onset_met = comparable >= 2 and onset_ok >= 2
# H3
m3 = pr[(pr.base == "cs@0.2") & (pr.alt == "cs-mondrian@0.2")].set_index("h")
h3n = sum(m3.loc[h, "d_fdr_pooled_hi"] < 0 for h in H)
for h in H:
    rec("H3 Mondrian-CS pooled FDR @0.2", h,
        f"{m3.loc[h, 'd_fdr_pooled']:.3f} [{m3.loc[h, 'd_fdr_pooled_lo']:.3f}, {m3.loc[h, 'd_fdr_pooled_hi']:.3f}]", "")
# H5
mp = pd.read_csv(D / "monitored_precision.csv")
cov = mp.groupby(["procedure", "h"]).covered.agg(["mean", "size"])
for (proc, h), row in cov.iterrows():
    rec(f"H5 coverage {proc}", h, f"{row['mean']:.3f} (n={int(row['size'])})", "")
h5 = bool((cov.loc["cs@0.2"]["mean"] >= 0.85).all())

rows = {(True, True): 0, (True, False): 1, (False, True): 2, (False, False): 3}
h1ok = h1 in ("holds", "not confirmed")
row = rows[(h2, h4)] + (0 if h1ok else 4) + 1
summary = {"H1": h1, "H2": h2, "H2 material (A2)": material, "H2b": h2b, "H4 dominated (>=2/3 h)": h4,
           "onset recommendation met": onset_met, "H3 (upper CI<0 at >=2 h)": h3n >= 2, "H5 (cs@0.2 coverage>=0.85)": h5,
           "claim-table row": row}
pd.DataFrame(out).to_csv(D / "hypotheses.csv", index=False)
with open(D / "hypotheses.md", "w") as f:
    f.write("# Confirmatory decisions (prereg v3 + v3.1)\n\n")
    for k, v in summary.items():
        f.write(f"- **{k}**: {v}\n")
    f.write("\n" + pd.DataFrame(out).to_markdown(index=False) + "\n")
print(open(D / "hypotheses.md").read())
