"""Realised FDR vs target FDR (control curve) for quantile-based alert policies."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import quantile_alerts as qa  # noqa: E402

d = pd.concat([qa.load(n) for n in qa.DATASETS], ignore_index=True)
d["task"] = d.dataset + ":" + d.target
d["event"] = (np.log1p(d.y_true) > d.thr) & (d.y_true >= qa.MIN_CASES)
rows = []
for q in [0.2, 0.3, 0.4, 0.5, 0.6]:
    qa.Q = q
    for src in ("lgb", "chr", "avg"):
        for lab, kw in {"fdr-task": dict(pooled=False, iso=False), "iso-task": dict(pooled=False, iso=True),
                        "fdr-pooled": dict(pooled=True, iso=False), "iso-pooled": dict(pooled=True, iso=True)}.items():
            a = qa.online(d, f"p_{src}", **kw)
            ev = d.event.to_numpy()
            # pooled over tasks + per-task worst case (tasks with >= 20 alerts)
            per = [((a & ~ev)[g.index].sum() / a[g.index].sum()) for _, g in d.groupby("task") if a[g.index].sum() >= 20]
            rows.append({"target_fdr": q, "source": src, "policy": lab, "alerts": int(a.sum()),
                         "fdr": float((a & ~ev).sum() / a.sum()) if a.sum() else np.nan,
                         "recall": float((a & ev).sum() / ev.sum()),
                         "worst_task_fdr": float(max(per)) if per else np.nan,
                         "tasks_with_alerts": len(per)})
    print("q", q, flush=True)
r = pd.DataFrame(rows)
r.to_csv(qa.OUT / "quantile_alerts_curve.csv", index=False)
pd.set_option("display.width", 220)
print(r.round(3).to_string(index=False))
