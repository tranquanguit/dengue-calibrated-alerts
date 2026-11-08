# Pre-registration v2 — Paper 2 confirmatory analysis (supersedes v1; written before any confirmation-country alert result)

Date: 2026-10-20. Frozen by pushing to the GitHub branch `prereg-p2-v2` before the confirmation runs.
v1 (`prereg-p2-v1`, commit `403f575`) is superseded after the supervisor's round-2 review: its single target
(bulletin FDR) is diluted by empty bulletins and its usefulness rule was too easy to pass.

Development set (already inspected): Thailand, Indonesia, Cambodia, Laos, Malaysia, pilot forecasts.
**Confirmation set (not inspected with these procedures):** Philippines, Viet Nam (self-compiled from
OpenDengue/Tycho), Sri Lanka, Brazil, Paper 1 confirmatory forecasts (`paper1_pooled_forecasting/results/unified_grid/`,
Block A for PH/VN, Block B for LK/BR).

## Frozen choices (unchanged from v1 unless marked)
- **Forecasts:** `ENS` = per-level average on the log1p scale of the quantiles of `Pooled-tuned` LightGBM and
  `Chronos2-cross` (levels 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95). Secondary: `Pooled-tuned` alone. Horizons 1–3.
- **Event ("endemic-channel exceedance"):** log1p cases above mean + 2 SD of the same calendar month in the previous
  5 years, the highest of those years excluded, and ≥ 5 cases. Onset = event month whose previous month is not an event.
- **Score:** `(log1p(q50) − thr) / max(log1p(q90) − log1p(q10), 0.05)`; eligibility `q50 ≥ 5`.
- **CS (primary procedure):** per bulletin (country × origin × h), conformal p-values of eligible provinces against all
  realised past cells of the same country and h (target ≤ origin, ≥ 200 cells), BH at q. NaN scores get p = 1 and are
  dropped from calibration (unit-tested).
- **Remedy (new, one, chosen on dev):** `cs-mondrian` — calibration restricted to past cells with the same epidemic
  state at their origin (state = ≥ 20 % of the country's provinces above their channel at the origin; falls back to
  CS when the same-state calibration set has < 100 cells).
- **Baselines:** point rule (q50 > thr and q50 ≥ 5), top-k (k = past event rate × provinces), CRC scan (same score and
  cells). Persistence is reported for recall only (its onset recall is 0 by construction).
- **Levels:** q = 0.2 (primary), 0.3.

## Metrics (all with 95 % country-year block-bootstrap CIs, 1,000 resamples)
Bulletin FDR (mean of V/max(R,1)); **FDR|R>0** (mean FDP over bulletins with an alert); **pooled FDR** (ΣV/ΣR, the
share of false alerts a user receives); recall; onset recall; share of onsets detected at any h; median lead time of
detected onsets; monitored-precision coverage (below).

## Hypotheses and decision rules (q = 0.2 primary; q = 0.3 reported identically)
- **H1 Guarantee.** Pooled over the four countries, the upper CI of CS bulletin FDR ≤ q at every h.
- **H2 Experienced gap.** CS pooled FDR > q (point estimate) at every h, and its lower CI > q at ≥ 2 of 3 horizons;
  FDR|R>0 > q (point estimate) at every h.
- **H3 Remedy.** `cs-mondrian` has lower pooled FDR than CS at every h (point estimates) and a recall loss < 0.05.
- **H4 Point rule.** Reported as a two-sided comparison: at each h, whether CS@0.2 / CS@0.3 / point dominate each
  other on (pooled FDR, recall). Recommendation of CS over the point rule requires CS@0.3 to detect a larger share of
  onsets at any horizon than the point rule.
- **H5 Monitored precision (secondary).** The one-sided 90 % Clopper–Pearson lower bound on trailing 24-month verified
  precision, issued at a country-year's first bulletin, is met by that year's realised precision in ≥ 85 % of
  country-years (dev result: 0.65–0.87, i.e. expected to fail; reported either way).

**Paper claim mapping (fixed now).**
- H1 holds, H2 holds → main claim: "the guaranteed FDR holds but is not the FDR users experience; the gap is
  quantified" (expected).
- H1 fails in ≥ 2 countries (point estimate > q) → main claim becomes "the finite-sample guarantee fails on
  real bulletins" plus the simulation mechanism.
- H2 fails (pooled FDR ≤ q + 0.05 at every h) → claim that CS also controls the experienced FDR at this data size.
- H3 fails → the remedy is reported as ineffective; no other remedy is tried on confirmation data.

## Not confirmatory
- Sri Lanka 2025-01 … 2026-04 (Block C): prospective illustration only.
- Event-definition sensitivity (k = 1.5, 2.5, 3; no max-year exclusion; floor 10), label delay 1–2 months,
  LTT procedures (bulletin LTT and cell LTT-mFDR: no alerts on dev), ACI (recall collapse on dev).
