# Pre-registration v3.1 — addendum to v3 (tightening and clarification only)

Required by the supervisor's round-4 approval (`manuscript/reviews/advisor_r4.md`). Written and frozen on branch
`prereg-p2-v3.1` before any confirmation-country alert result. Everything in v3 holds except where it is changed below.

**A1. H1 outcomes split into three.**
- *Holds*: upper CI ≤ q at every h.
- *Violated*: the point estimate of the pooled bulletin FDR is > q at ≥ 1 horizon.
- *Not confirmed*: point estimate ≤ q at every h, but the upper CI is > q at ≥ 1 horizon.

In the claim table, the "fails" rows apply to *violated*. *Not confirmed* uses the "holds" row's claim with the
wording "the guarantee is not contradicted but not confirmed at this sample size".

**A2. Practical-significance threshold δ = 0.05.**
- "Materially higher" is used only if the lower CI of Δ(CS@0.2) is > 0.05 at every h.
- If Δ > 0 holds but this threshold is not met, the wording becomes "higher".
- H2b, Δ(CS@0.2) − Δ(point-margin>0) with lower CI > 0 at every h, is the main evidence: Δ > 0 alone is
  expected from BH's mechanism (the iid simulation gives Δ ≈ 0.10).

**A3. Frontier and onset criterion.**
- The point-rule family is extended with c ∈ {−0.3, −0.2, −0.1}, for both margin and zmargin. These are used only
  for frontier plots and paired comparisons.
- The onset recommendation criterion applies to CS@0.2. At each h, it is evaluated only if some point-family member
  has pooled FDR within ±0.03 of CS@0.2's. That member (the closest one) is compared on onset recall; otherwise the
  horizon is recorded as "not comparable". The criterion counts as met if CS@0.2's onset recall is higher at ≥ 2 of
  the comparable horizons, with at least 2 comparable horizons required.
- H4 dominance is also reported with paired CIs (CS@0.2 vs the dominating member) on pooled FDR and recall.

**A4. Additional reporting rules.**
- Proposition 1 is stated with the bound q·π0·m_el/m, because BH counts ineligible cells (p = 1) in m. No code change.
- Confirmation uses N_BOOT = 1,000.
- Per-country estimates are reported next to every pooled metric.
- The manuscript states publicly where v3 is easier than v2: H1 is on the pooled CI, and the original point family
  had only c ≥ 0.
