"""Vector figures for the IEEE (ISCI 2027) version of Paper 2: frontier (full width) and simulation (one column)."""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(__file__).resolve().parents[1] / "results"
OUT = Path(__file__).resolve().parents[1] / "manuscript/ieee_isci2027/figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "lines.linewidth": 1.0,
                     "pdf.fonttype": 42})

r = pd.read_csv(R / "confirmation/procedures.csv")
r = r[r.country == "pooled"]
crc = pd.read_csv(R / "confirmation/procedures_crc_fixed.csv")
r = pd.concat([r[r.family != "crc"], crc])
r = r[r.procedure != "point-z>0.0"]
fam = {"point-margin": ("#4C78A8", "o", "Fixed threshold (margin $>c$)"), "point-z": ("#72B7B2", "s", "Fixed threshold (z-margin $>c$)"),
       "cs": ("#E45756", "^", "Conformal selection"), "cs-mondrian": ("#F58518", "v", "Mondrian CS"),
       "crc": ("#54A24B", "D", "CRC scan")}
fig, ax = plt.subplots(2, 3, figsize=(7.16, 3.45))
for j, h in enumerate((1, 2, 3)):
    g = r[r.h == h]
    for i, xc in enumerate(("fdr_pooled", "fdr_bulletin")):
        for f, (col, mk, lab) in fam.items():
            s = g[g.family == f].sort_values(xc)
            ax[i, j].plot(s[xc], s.recall, marker=mk, color=col, label=lab, ms=2.8, alpha=.9)
        lab_pos = {0: {"cs@0.2": ((5, -10), "left"), "point-margin>0.0": ((-5, 5), "right")},
                   1: {"cs@0.2": ((-6, 6), "right"), "point-margin>0.0": ((5, -11), "left")}}[i]
        for nm, txt, col in (("cs@0.2", "CS@0.2", "#B22222"), ("point-margin>0.0", "$c=0$", "#1f3b73")):
            off, ha = lab_pos[nm]
            c = g[g.procedure == nm].iloc[0]
            ax[i, j].plot(c[xc], c.recall, "o", mfc="white", mec=col, ms=5, mew=1.2, zorder=5)
            ax[i, j].annotate(txt, (c[xc], c.recall), textcoords="offset points", xytext=off, ha=ha, fontsize=6.5,
                              color=col, zorder=6, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))
        ax[i, j].axvline(0.2, color="k", ls=":", lw=0.7)
        ax[i, j].grid(alpha=.3, lw=0.5)
    ax[0, j].set_title(f"$h={h}$")
    ax[0, j].set_xlabel("Pooled FDR (experienced)")
    ax[1, j].set_xlabel("Bulletin FDR (guaranteed)")
ax[0, 0].set_ylabel("Recall")
ax[1, 0].set_ylabel("Recall")
hs, ls = ax[0, 0].get_legend_handles_labels()
fig.legend(hs, ls, loc="upper center", ncol=5, frameon=False, handlelength=1.8, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.6, rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "frontier.pdf")

s = pd.read_csv(R / "simulation/simulation.csv")
s = s[s.q == 0.2].set_index("scenario")
rows = [("iid", "i.i.d."), ("shared bulletin error 0.25", "Shared error $\\sigma_w$=0.25"),
        ("empirical share 0.25 (sigma_w 0.35)", "Empirical share 0.25"), ("empirical share 0.5 (sigma_w 0.6)", "Empirical share 0.50"),
        ("shared bulletin error 1.0", "Shared error $\\sigma_w$=1.0"), ("year shock, forecast misses it", "Missed year shock")]
fig, a = plt.subplots(figsize=(3.45, 2.2))
y = range(len(rows))
w = 0.26
for k, (col, c, lab) in enumerate((("fdr_cs", "#4C78A8", "CS: bulletin FDR"), ("fdrpool_cs", "#E45756", "CS: pooled FDR"),
                                   ("fdrpool_pt", "#999999", "Fixed threshold: pooled FDR"))):
    a.barh([i + (k - 1) * w for i in y], [s.loc[n, col] for n, _ in rows], w,
           xerr=[1.96 * s.loc[n, f"se_{col}"] for n, _ in rows], color=c, label=lab, error_kw=dict(lw=0.5, capsize=1))
a.axvline(0.2, color="k", ls="--", lw=0.7)
a.set_yticks(list(y))
a.set_yticklabels([lab for _, lab in rows])
a.invert_yaxis()
a.set_xlabel("FDR at $q=0.2$ (300 replicates)")
a.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=2, columnspacing=1.0, handlelength=1.2)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "simulation.pdf")
print("ok")
