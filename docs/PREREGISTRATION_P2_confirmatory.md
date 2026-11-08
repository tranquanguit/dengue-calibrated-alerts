# Pre-registration — Paper 2 confirmatory analysis (written before any confirmation-country alert result)

Date: 2026-10-20. Frozen by pushing to the GitHub branch `prereg-p2-v1` before the confirmation runs.
Development set (already inspected): Thailand, Indonesia, Cambodia, Laos, Malaysia, pilot forecasts.
**Confirmation set (not inspected with the procedures below):** Philippines, Viet Nam, Sri Lanka, Brazil,
using Paper 1's confirmatory forecasts (`paper1_pooled_forecasting/results/unified_grid/`).

## Frozen choices
- **Forecasts:** Paper 1 pre-specified ensemble `ENS` = mean of the quantiles of `Pooled-tuned` LightGBM and
  `Chronos2-cross` (levels 0.05 … 0.95). Secondary: `Pooled-tuned` alone.
- **Event:** endemic channel — log1p cases above mean + 2 SD of the same calendar month in the previous 5 years,
  the highest of those years excluded, and ≥ 5 cases. Onset = event month whose previous month is not an event.
- **Score:** standardised margin `(log1p(q50) − thr) / max(log1p(q90) − log1p(q10), 0.05)`; eligibility
  `q50 ≥ 5` cases.
- **Procedure (primary):** conformal selection — for each bulletin (country × origin month × horizon h),
  conformal p-values of the eligible provinces against all realised past cells of the same country and horizon
  (target month ≤ origin, ≥ 200 cells), p = (1 + #{null cells with score ≥ s}) / (n + 1); Benjamini–Hochberg at q.
- **Levels:** q = 0.2 (primary) and q = 0.3. Horizons 1, 2, 3 reported separately.
- **Baselines:** point rule (q50 > thr and q50 ≥ 5), persistence (province above its channel at the origin),
  top-k (k = past event rate × provinces), pilot CRC scan (same score, same calibration cells).

## Target quantity and metrics
- **Primary target:** bulletin FDR = mean over bulletins of V / max(R, 1) (the quantity BH controls).
- Secondary: country-year FDR (mean over country × year of FDP), pooled FDR, share of country-years with FDP > q,
  recall of event months, recall of onsets, onset lead time (largest h with an alert for the onset month).
- 95 % CIs by bootstrap over country-years (1,000 resamples).

## Decision rules
1. **Validity:** supported if, pooled over the four confirmation countries, the upper 95 % CI of the bulletin FDR
   is ≤ q at both levels and every horizon, and the point estimate is ≤ q in each confirmation country.
2. **Usefulness:** supported if, at every horizon, conformal selection at q = 0.3 is not dominated by the point
   rule (no lower recall at no lower bulletin FDR) and has higher onset recall than persistence.
3. **Pivot:** if (1) fails in two or more confirmation countries, the paper pivots to budgeted (top-k) alerts with
   conformal precision bounds, as agreed with the supervisor.

## Not confirmatory
- Sri Lanka 2025-01 … 2026-04 (Block C): prospective illustration only (too short for a year-level claim).
- Sensitivity of the event definition (k = 1.5; no max-year exclusion; floor 10 cases), label delay of 1–2 months,
  minimum calibration years — reported as exploratory.
