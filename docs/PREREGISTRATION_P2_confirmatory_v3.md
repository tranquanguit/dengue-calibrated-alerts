# Pre-registration v3 — Paper 2 confirmatory analysis (supersedes v2; written before any confirmation-country alert result)

Date: 2026-10-20. Frozen by pushing to the GitHub branch `prereg-p2-v3` before the confirmation runs.

**Why v3.** The supervisor's round-3 review found four problems with v2:
- CS was calibrated on all past cells, including ineligible ones. It is now calibrated on eligible cells only
  (Proposition 1 holds conditionally on eligibility; unit-tested).
- H2 compared against q and already failed on dev. It is now defined on a paired gap.
- The claim mapping had holes. It is now a full 2×2×2 table.
- The point-rule comparison used a single rule on a different score. It is now a frozen family on both scores.

Nothing else from v2 changes unless stated: data, forecasts (ENS = log1p-average of Pooled-tuned and Chronos2-cross
quantiles), event definition, score, horizons 1–3, and 1,000 country-year bootstrap resamples.

**Confirmation set.** Philippines, Viet Nam (self-compiled), Sri Lanka, Brazil. Paper 1 unified-grid forecasts:
Block A covers PH and VN 2006–2010; Block B covers LK 2016–2024 and BR 2017–2024. Only country-years whose origins
have ≥ 200 eligible past cells contribute.

## Frozen procedures
- **CS@q**, for q ∈ {0.1, 0.15, 0.2, 0.25, 0.3}. Primary: q = 0.2. The procedure is BH on conformal p-values
  against eligible past cells of the same country and horizon.
- **cs-mondrian@q**: the same q grid, with calibration on same-state cells (state = ≥ 20 % of provinces above their
  channel at the origin). Sensitivity: state lagged one month.
- **Point-rule family**: `s_margin > c` and `s_zmargin > c` with c ∈ {0, 0.1, 0.2, 0.3, 0.5}, and q50 ≥ 5. The
  family is used only to trace a frontier, never to pick c.
- Descriptive baselines: CRC scan (q = 0.2, 0.3) and top-k.

## Metrics
- Bulletin FDR, FDR|R>0, pooled FDR, recall, onset recall, share of onsets detected at any h, and median lead time.
- **Gaps**: Δ = pooled FDR − bulletin FDR and Δ' = FDR|R>0 − bulletin FDR. Paired differences between procedures
  are computed on the same bootstrap resamples.

## Hypotheses (all pooled over the four countries, at each h ∈ {1, 2, 3})
- **H2 (primary): experienced-FDR gap.** For CS@0.2, the lower 95 % CI of Δ is > 0 at every h.
  - Co-primary contrast H2b: the paired difference Δ(CS@0.2) − Δ(point-margin>0) has lower CI > 0 at every h, i.e.
    the gap is specific to per-bulletin BH and not a generic feature of alerting.
  - Δ' is reported alongside.
- **H1 (sanity): guarantee.** For CS@0.2, the upper 95 % CI of the bulletin FDR is ≤ 0.2 at every h. Per-country
  estimates are descriptive only.
- **H4: frontier.** At a given h, CS@0.2 is *dominated* if some member of the point-rule family has pooled FDR ≤
  and recall ≥ that of CS@0.2 (point estimates), with at least one inequality strict. H4 = "CS@0.2 dominated at
  ≥ 2 of 3 horizons".
  - Recommendation criterion: CS is recommended for onset detection only if, at ≥ 2 of 3 horizons, its onset recall
    exceeds that of the point-family member with the nearest pooled FDR.
- **H3 (secondary): remedy.** cs-mondrian@0.2 − CS@0.2 paired pooled-FDR difference has upper CI < 0 at ≥ 2 of 3
  horizons, and the Mondrian q-curve lies outside the CS q-curve on (pooled FDR, recall).
  - Dev result: every paired CI includes 0, so H3 is expected to be null.
  - Reported as "no detectable effect" when the CIs include 0.
- **H5 (secondary): monitored precision.** As in v2, with the caveat that Clopper–Pearson assumes independent
  alerts, which clustering in large bulletins violates. Label-tier rank reversal is reported.

**Inconclusive rule.** A hypothesis whose deciding CI straddles its boundary at a horizon counts as *not supported*
at that horizon. The claim mapping uses the result as computed, and the text says "not determined at this sample
size" when the CI half-width exceeds the projected 0.08–0.16 (dev bootstrap scaled to ~20 country-years).

## Claim mapping (H1 × H2 × H4; every combination has exactly one claim)
| H1 guarantee | H2 gap (Δ>0 all h) | H4 CS dominated | Paper claim |
|---|---|---|---|
| holds | yes | yes | **Expected.** The guarantee holds but users experience a materially higher FDR, and an uncalibrated point rule gives a better experienced precision–recall trade-off. Recommend reporting the guaranteed level only together with the experienced FDR. |
| holds | yes | no | The guarantee holds and the gap is real, but CS also reaches part of the frontier the point rule cannot. Recommend CS with experienced-FDR reporting. |
| holds | no | yes | The guarantee holds and the gap is not established at this size, but the point rule dominates. The benefit of a guarantee is only formal. |
| holds | no | no | CS is valid and competitive; the experienced gap is small. Positive method claim, limited to these countries. |
| fails | yes | yes | Neither guarantee nor experienced precision holds on real bulletins (non-exchangeability). Report the mechanism from simulation; recommend the point rule plus monitored precision. |
| fails | yes | no | The guarantee fails and the gap is large, yet CS reaches the frontier. Claim limited to ranking value; no guarantee claim. |
| fails | no | yes | The guarantee fails, the gap is unclear, and the point rule dominates. Negative paper: per-bulletin conformal selection is not suited to these data. |
| fails | no | no | The guarantee fails, but the experienced FDR equals the point rule's. Report as an empirical procedure without a guarantee. |
H2b, H3 and H5 modify only the wording of the mechanism, remedy and labels, never the row.

## Scope of negative claims
- LTT infeasibility is claimed only for the Hoeffding (bulletin LTT) and empirical-Bernstein (cell LTT-mFDR) bounds
  used. LTT-mFDR pools calibration across countries (score comparability), unlike the per-country CS.
- Sri Lanka 2025–26 (Block C) is a prospective illustration only.
- Event-definition sensitivity and label delay are exploratory.
