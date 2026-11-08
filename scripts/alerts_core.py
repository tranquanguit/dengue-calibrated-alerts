"""Core of Paper 2: event definition, scores, alert procedures and evaluation (no data access here).

Event (endemic channel): province r exceeds, in target month m of year Y, the channel computed from
the same calendar month of the previous `window` years (the single highest year excluded, as epidemic-year
exclusion), threshold = mean + k * SD of log1p(cases), and at least `floor` cases.
Onset: an event month whose previous month is not an event.

Procedures choose, at each forecast origin t and for each country c and horizon h, which provinces to alert,
using only cells whose target month is observed at t (target_date <= t):
  point       : alert iff predicted median > threshold (no tuning)
  cs          : conformal selection (Jin & Candes 2023): conformal p-values against past NULL cells,
                p_j = (1 + #{i null: s_i >= s_j}) / (n + 1), then Benjamini-Hochberg at level q over the
                provinces of (c, t, h). FDR <= q if calibration and test cells are exchangeable.
  cs-w(HL)    : weighted conformal selection (Barber et al. 2023 non-exchangeable conformal): weights
                0.5 ** (age / HL months); the FDR excess is bounded by the weighted distribution shift.
  persistence : alert iff the province is above its channel in the origin month itself
  topk        : alert the k provinces with the highest score, k = round(past event rate * #provinces)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


# ----------------------------------------------------------------------------- events
def endemic_channel(panel: pd.DataFrame, window: int = 5, k: float = 2.0, exclude_max: bool = True) -> pd.DataFrame:
    """panel: region, date, cases. Returns region, date, thr (log1p scale) for every month with history."""
    p = panel.assign(ly=np.log1p(panel.cases), m=panel.date.dt.month, yr=panel.date.dt.year)
    piv = p.pivot_table(index=["region", "m"], columns="yr", values="ly")
    years = sorted(p.yr.unique())
    rows = []
    for Y in years:
        past = [y for y in years if Y - window <= y < Y and y in piv.columns]
        if len(past) < 3:
            continue
        x = piv[past].to_numpy(float)
        if exclude_max and x.shape[1] >= 4:
            x = np.where(np.arange(x.shape[1])[None, :] == np.nanargmax(np.nan_to_num(x, nan=-1), axis=1)[:, None], np.nan, x)
        thr = np.nanmean(x, axis=1) + k * np.nanstd(x, axis=1, ddof=1)
        rows.append(pd.DataFrame({"region": piv.index.get_level_values(0), "m": piv.index.get_level_values(1),
                                  "yr": Y, "thr": thr}))
    t = pd.concat(rows, ignore_index=True)
    t["date"] = pd.to_datetime(dict(year=t.yr, month=t.m, day=1))
    return t[["region", "date", "thr"]]


def add_events(d: pd.DataFrame, thr: pd.DataFrame, floor: int = 5) -> pd.DataFrame:
    """d: forecast rows (region, origin, target_date, y_true, ...). Adds thr, event, onset, persistence."""
    th = thr.rename(columns={"date": "target_date"})
    d = d.merge(th, on=["region", "target_date"], how="inner")
    d["event"] = (np.log1p(d.y_true) > d.thr) & (d.y_true >= floor)
    return d


def onset_flags(panel: pd.DataFrame, thr: pd.DataFrame, floor: int = 5) -> pd.DataFrame:
    """Per region-month: event and onset (event and previous month not an event), on observed data."""
    x = panel.merge(thr, on=["region", "date"], how="inner").sort_values(["region", "date"])
    x["event"] = (np.log1p(x.cases) > x.thr) & (x.cases >= floor)
    prev = x.groupby("region").event.shift(1)
    x["onset"] = x.event & (prev == False)  # noqa: E712  (unknown previous month -> not an onset)
    return x[["region", "date", "event", "onset", "cases", "thr"]]


# ----------------------------------------------------------------------------- scores
def scores(d: pd.DataFrame, qcols: list[str]) -> pd.DataFrame:
    lq = np.log1p(np.sort(d[qcols].to_numpy(float), axis=1))
    med = lq[:, qcols.index("q50")]
    spread = np.maximum(lq[:, qcols.index("q90")] - lq[:, qcols.index("q10")], 0.05)
    d = d.copy()
    d["s_margin"] = med - d.thr
    d["s_zmargin"] = (med - d.thr) / spread
    sigma = spread / (2 * 1.2816)
    d["s_lognorm"] = 1 - norm.cdf((d.thr - med) / sigma)  # smooth P(event), no tail saturation
    d["eligible"] = np.expm1(med) >= 5
    return d


# ----------------------------------------------------------------------------- procedures
def bh(p: np.ndarray, q: float) -> np.ndarray:
    n = len(p)
    if n == 0:
        return np.zeros(0, bool)
    order = np.argsort(p)
    ok = p[order] <= q * np.arange(1, n + 1) / n
    k = np.max(np.flatnonzero(ok)) + 1 if ok.any() else 0
    sel = np.zeros(n, bool)
    sel[order[:k]] = True
    return sel


def conformal_pvalues(s_test, s_cal, null_cal, w_cal=None):
    """Conformal p-values of test scores against calibration NULL cells (higher score = more alarming).

    NaN calibration scores are dropped; NaN test scores get p = 1 (never alerted).
    """
    s_test = np.asarray(s_test, float)
    s_cal, null_cal = np.asarray(s_cal, float), np.asarray(null_cal, bool)
    if w_cal is None:
        w_cal = np.ones(len(s_cal))
    keep = ~np.isnan(s_cal)
    s_cal, null_cal, w_cal = s_cal[keep], null_cal[keep], np.asarray(w_cal, float)[keep]
    sn, wn = s_cal[null_cal], w_cal[null_cal]
    order = np.argsort(sn)
    sn, wn = sn[order], wn[order]
    tail = np.concatenate([np.cumsum(wn[::-1])[::-1], [0.0]])  # weight of nulls with score >= value
    idx = np.searchsorted(sn, np.nan_to_num(s_test, nan=np.inf), side="left")
    p = (1.0 + tail[idx]) / (1.0 + w_cal.sum())
    return np.where(np.isnan(s_test), 1.0, p)


def ltt_level(past, q, delta=0.1, grid=None, n_prov=1):
    """Learn-then-Test choice of the BH level q' from past bulletins (bulletins = exchangeable units).

    past: list of (p-values, events) of realised past bulletins (their p-values computed at their own origin).
    Loss per bulletin L = (V - q R) / n_prov in [-q, 1 - q]; H0(q'): E[L(q')] > 0, i.e. pooled FDR > q.
    Hoeffding p-value exp(-2 n mean^2) when the mean is negative; fixed-sequence testing from the most
    conservative level upward; returns the last level whose null is rejected at delta (0 if none).
    """
    grid = grid if grid is not None else np.round(np.arange(0.02, q + 1e-9, 0.02), 3)
    chosen = 0.0
    for lv in grid:
        L = []
        for p, e in past:
            a = bh(p, lv)
            L.append(((a & ~e).sum() - q * a.sum()) / n_prov)
        L = np.asarray(L)
        m = L.mean() if len(L) else 0.0
        pval = np.exp(-2 * len(L) * m * m) if m < 0 else 1.0
        if pval <= delta:
            chosen = lv
        else:
            break
    return chosen


def ltt_mfdr_cutoff(s_cal, null_cal, q, delta=0.1, grid=None):
    """Learn-then-Test cut-off for the marginal FDR E[V]/E[R] <= q with province-months as units.

    Per-cell loss for cut-off c: L_c = 1{s >= c} (1{null} - q), and E[L_c] <= 0  <=>  mFDR(c) <= q.
    Each H0(c): E[L_c] > 0 is tested with an empirical-Bernstein upper bound (Maurer & Pontil 2009);
    fixed-sequence testing from the strictest cut-off downward; returns the last cut-off rejected at delta.
    """
    s_cal, null_cal = np.asarray(s_cal, float), np.asarray(null_cal, bool)
    ok = ~np.isnan(s_cal)
    s_cal, null_cal = s_cal[ok], null_cal[ok]
    n = len(s_cal)
    if grid is None:
        grid = np.quantile(s_cal, np.linspace(0.995, 0.5, 100))
    chosen = np.inf
    lg = np.log(2 / delta)
    for c in grid:
        L = (s_cal >= c) * (null_cal - q)
        m, v = L.mean(), L.var(ddof=1)
        ub = m + np.sqrt(2 * v * lg / n) + 7 * lg / (3 * (n - 1))
        if ub <= 0:
            chosen = c
        else:
            break
    return chosen


def run_procedure(d: pd.DataFrame, proc: str, score: str = "s_zmargin", q: float = 0.2,
                  half_life: float | None = None, min_cal: int = 200, delta: float = 0.1,
                  min_bulletins: int = 24, gamma: float = 0.5, q_min: float = 0.01) -> np.ndarray:
    """Alert decisions for every row of d (one country at a time is handled inside)."""
    alert = np.zeros(len(d), bool)
    pv = np.full(len(d), np.nan)  # proc "cs-p": conformal p-values (NaN where CS could not run: burn-in)
    if proc == "point":
        return ((d.s_margin > 0) & d.eligible).to_numpy()
    if proc in ("point-margin", "point-z"):  # frozen point-rule family: score > c (c passed as q)
        col = "s_margin" if proc == "point-margin" else "s_zmargin"
        return ((d[col] > q) & d.eligible).to_numpy()
    if proc == "persistence":
        return d.persist.fillna(False).to_numpy(bool)
    if proc == "ltt-mfdr":  # calibration pooled over all countries (score comparable), per horizon
        for h, g in d.groupby("horizon"):
            td, org = g.target_date.to_numpy(), g.origin.to_numpy()
            s_, e_, el_ = g[score].to_numpy(float), g.event.to_numpy(bool), g.eligible.to_numpy(bool)
            gi = g.index.to_numpy()
            for t in np.unique(org):
                cur, cal = org == t, (td <= t) & el_
                if cal.sum() < min_cal:
                    continue
                c = ltt_mfdr_cutoff(s_[cal], ~e_[cal], q, delta)
                alert[gi[cur]] = (s_[cur] >= c) & el_[cur]
        return alert
    if proc == "crc":  # pilot rule: smallest cut-off with (false+1)/(alerts+1) <= q on past cells (invalid)
        grid = np.nanquantile(d[score].to_numpy(float), np.linspace(0.5, 0.995, 100))  # NaN scores never alert
    for (_, _), g in d.groupby(["country", "horizon"]):
        td, org = g.target_date.to_numpy(), g.origin.to_numpy()
        s, e, el = g[score].to_numpy(float), g.event.to_numpy(bool), g.eligible.to_numpy(bool)
        idx = g.index.to_numpy()
        history = []  # (origin, target date, p-values, events) of bulletins already issued (for LTT)
        level = q     # BH level adapted online from realised bulletins (cs-aci)
        pending = []  # (target date, V, R, n) of issued bulletins awaiting their outcome
        for t in np.unique(org):
            cur = np.flatnonzero(org == t)
            cal = td <= t
            if cal.sum() < min_cal:
                continue
            if proc == "crc":
                ok = cal & el & ~np.isnan(s)
                sc, ec = s[ok], e[ok]
                cut = np.inf
                for c_ in grid:
                    a_ = sc >= c_
                    if (np.sum(a_ & ~ec) + 1) / (np.sum(a_) + 1) <= q:
                        cut = c_
                        break
                alert[idx[cur]] = (s[cur] >= cut) & el[cur]
                continue
            if proc == "topk":
                rate = e[cal].mean()
                k = int(round(rate * len(cur)))
                if k > 0:
                    top = cur[np.argsort(-s[cur])[:k]]
                    alert[idx[top]] = el[top]
                continue
            w = None
            if proc == "cs-w":
                age = (t - td[cal]).astype("timedelta64[D]").astype(float) / 30.44
                w = 0.5 ** (age / half_life)
            # only eligible cells are tested, so calibrate on eligible past cells (exchangeability given eligibility)
            cal = cal & el
            if cal.sum() < min_cal:
                continue
            p = conformal_pvalues(s[cur], s[cal], ~e[cal], None if w is None else w[el[td <= t]])
            p = np.where(el[cur], p, 1.0)
            if proc == "cs-p":
                pv[idx[cur]] = p
                continue
            if proc == "cs-el":  # BH over eligible cells only (m = m_el)
                a = np.zeros(len(cur), bool)
                a[el[cur]] = bh(p[el[cur]], q)
                alert[idx[cur]] = a
                continue
            if proc in ("cs-mondrian", "cs-mondrian-lag1"):
                # national epidemic state at the origin: share of provinces already above their channel;
                # calibration restricted to past cells whose origin had the same state (conditional exchangeability)
                st = g["state" if proc == "cs-mondrian" else "state_lag1"].to_numpy()
                same = cal & (st == st[cur[0]])
                if same.sum() >= min_cal // 2:
                    p = conformal_pvalues(s[cur], s[same], ~e[same])
                    p = np.where(el[cur], p, 1.0)
                alert[idx[cur]] = bh(p, q)
                continue
            if proc == "cs-aci":
                # feedback from bulletins whose target month is now observed: level += gamma (q R - V) / n
                ready = [b for b in pending if b[0] <= t]
                pending = [b for b in pending if b[0] > t]
                for _, V, R, n in ready:
                    level = max(q_min, level + gamma * (q * R - V) / n)
                a = bh(p, min(level, 1.0))
                alert[idx[cur]] = a
                pending.append((td[cur].max(), int((a & ~e[cur]).sum()), int(a.sum()), len(cur)))
                continue
            if proc == "cs-ltt":
                past = [(pp, ee) for (o, tt, pp, ee) in history if tt <= t]
                lv = ltt_level(past, q, delta, n_prov=len(cur)) if len(past) >= min_bulletins else 0.0
                alert[idx[cur]] = bh(p, lv) if lv > 0 else False
                history.append((t, td[cur].max(), p, e[cur]))
                continue
            alert[idx[cur]] = bh(p, q)
    return pv if proc == "cs-p" else alert


# ----------------------------------------------------------------------------- evaluation
def bulletin_table(d: pd.DataFrame, alert: np.ndarray) -> pd.DataFrame:
    x = d.assign(alert=alert, yr=d.target_date.dt.year, tp=alert & d.event.to_numpy(), fp=alert & ~d.event.to_numpy())
    b = x.groupby(["country", "origin", "horizon"]).agg(R=("alert", "sum"), V=("fp", "sum"), TP=("tp", "sum"),
                                                        E=("event", "sum"), yr=("yr", "first"))
    on = x[x.onset.fillna(False).astype(bool)] if "onset" in x else x.iloc[:0]
    o = on.groupby(["country", "origin", "horizon"]).agg(ON=("onset", "size"), ONA=("alert", "sum"))
    b = b.join(o).fillna({"ON": 0, "ONA": 0}).reset_index()
    return b


def metrics_from_bulletins(b: pd.DataFrame, w=None) -> dict:
    w = np.ones(len(b)) if w is None else w
    fdp = np.where(b.R > 0, b.V / b.R.clip(lower=1), 0.0)
    pos = (b.R > 0).to_numpy()
    sw = lambda v: float(np.sum(w * v))  # noqa: E731
    return {"fdr_bulletin": sw(fdp) / max(sw(np.ones(len(b))), 1e-12),
            "fdr_given_alert": sw(fdp * pos) / max(sw(pos), 1e-12) if sw(pos) > 0 else np.nan,
            "fdr_pooled": sw(b.V) / max(sw(b.R), 1e-12) if sw(b.R) > 0 else np.nan,
            "recall": sw(b.TP) / max(sw(b.E), 1e-12),
            "onset_recall": sw(b.ONA) / max(sw(b.ON), 1e-12) if sw(b.ON) > 0 else np.nan,
            "share_bulletins_alerting": sw(pos) / max(sw(np.ones(len(b))), 1e-12)}


def _gap_wrap(f):
    def g(b, w=None):
        m = f(b, w)
        m["gap_pooled"] = m["fdr_pooled"] - m["fdr_bulletin"] if not np.isnan(m["fdr_pooled"]) else np.nan
        m["gap_pos"] = m["fdr_given_alert"] - m["fdr_bulletin"] if not np.isnan(m["fdr_given_alert"]) else np.nan
        return m
    return g


metrics_from_bulletins = _gap_wrap(metrics_from_bulletins)


def evaluate_paired(d: pd.DataFrame, a1: np.ndarray, a2: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """Differences (procedure 2 - procedure 1) of every metric with a paired country-year bootstrap CI."""
    b1, b2 = bulletin_table(d, a1), bulletin_table(d, a2)
    m1, m2 = metrics_from_bulletins(b1), metrics_from_bulletins(b2)
    out = {f"d_{k}": m2[k] - m1[k] for k in m1}
    units = b1[["country", "yr"]].drop_duplicates().to_numpy()
    key = b1.country + ":" + b1.yr.astype(str)
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        pick = units[rng.integers(0, len(units), len(units))]
        w = key.map(pd.Series([f"{a}:{c}" for a, c in pick]).value_counts()).fillna(0).to_numpy()
        r1, r2 = metrics_from_bulletins(b1, w), metrics_from_bulletins(b2, w)
        reps.append({f"d_{k}": r2[k] - r1[k] for k in r1})
    reps = pd.DataFrame(reps)
    for k in reps:
        out[f"{k}_lo"], out[f"{k}_hi"] = float(reps[k].quantile(0.025)), float(reps[k].quantile(0.975))
    return out


def evaluate(d: pd.DataFrame, alert: np.ndarray, q: float | None = None, n_boot: int = 1000, seed: int = 0) -> dict:
    """Point estimates + country-year block-bootstrap 95 % CIs for every metric."""
    b = bulletin_table(d, alert)
    out = {"alerts": int(np.sum(alert)), **metrics_from_bulletins(b)}
    cy = b.groupby(["country", "yr"]).apply(lambda g: (np.where(g.R > 0, g.V / g.R.clip(lower=1), 0)).mean())
    out["fdr_cy"] = float(cy.mean())
    if q is not None:
        cyp = b[b.R > 0].groupby(["country", "yr"]).apply(lambda g: g.V.sum() / g.R.sum())
        out["share_cy_pooledfdp_over_q"] = float((cyp > q).mean()) if len(cyp) else np.nan
    if n_boot:
        units = b[["country", "yr"]].drop_duplicates().to_numpy()
        key = b.country + ":" + b.yr.astype(str)
        rng = np.random.default_rng(seed)
        reps = []
        for _ in range(n_boot):
            pick = units[rng.integers(0, len(units), len(units))]
            cnt = pd.Series([f"{a}:{c}" for a, c in pick]).value_counts()
            reps.append(metrics_from_bulletins(b, key.map(cnt).fillna(0).to_numpy()))
        reps = pd.DataFrame(reps)
        for k in reps:
            out[f"{k}_lo"], out[f"{k}_hi"] = float(reps[k].quantile(0.025)), float(reps[k].quantile(0.975))
    return out


def lead_time(d: pd.DataFrame, alert: np.ndarray) -> pd.DataFrame:
    """For each onset (region, target month): largest horizon at which an alert for it was issued."""
    x = d.assign(alert=alert)
    on = x[x.onset]
    lt = on.groupby(["country", "region", "target_date"]).apply(
        lambda g: g.loc[g.alert, "horizon"].max() if g.alert.any() else 0)
    return lt.rename("lead").reset_index()




def monitored_precision(b: pd.DataFrame, window: int = 24, level: float = 0.9, delay: int = 0) -> pd.DataFrame:
    """Trailing precision label for each bulletin, using only bulletins already verified at its origin.

    For bulletin (c, t, h) the verified history is every bulletin of the same country and horizon whose target month
    (origin + h) is <= t - delay and > t - window. Returns the pooled precision TP/R of that history and its
    one-sided Clopper-Pearson lower bound at `level` (NaN when the history has no alerts).
    """
    from scipy.stats import beta
    b = b.copy()
    b["target"] = b.origin + b.horizon.map(lambda h: pd.DateOffset(months=int(h)))
    out = []
    for (c, h), g in b.groupby(["country", "horizon"]):
        g = g.sort_values("origin")
        tt, tp, r = g.target.to_numpy(), g.TP.to_numpy(), g.R.to_numpy()
        for o in g.origin:
            hi = o - pd.DateOffset(months=delay)
            m = (tt <= hi) & (tt > o - pd.DateOffset(months=window))
            n, k = r[m].sum(), tp[m].sum()
            lb = beta.ppf(1 - level, k, n - k + 1) if n > 0 and k > 0 else (0.0 if n > 0 else np.nan)
            out.append((c, o, h, n, k / n if n else np.nan, lb))
    lab = pd.DataFrame(out, columns=["country", "origin", "horizon", "hist_alerts", "prec_hist", "prec_lb"])
    return b.merge(lab, on=["country", "origin", "horizon"], how="left")
