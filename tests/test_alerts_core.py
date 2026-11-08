import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import alerts_core as ac  # noqa: E402


def test_nan_test_score_never_alerted():
    p = ac.conformal_pvalues(np.array([np.nan, 5.0]), np.array([0.0, 1.0, 2.0, 3.0]), np.array([True] * 4))
    assert p[0] == 1.0
    assert p[1] == 1 / 5


def test_nan_calibration_scores_dropped():
    p1 = ac.conformal_pvalues(np.array([1.5]), np.array([0.0, 1.0, 2.0, np.nan]), np.array([True] * 4))
    p2 = ac.conformal_pvalues(np.array([1.5]), np.array([0.0, 1.0, 2.0]), np.array([True] * 3))
    assert np.isclose(p1[0], p2[0])


def test_non_null_calibration_cells_not_counted():
    # an event cell with a huge score must not make the test p-value larger
    p = ac.conformal_pvalues(np.array([1.0]), np.array([0.0, 10.0]), np.array([True, False]))
    assert np.isclose(p[0], 1 / 3)


def test_bh_basic():
    sel = ac.bh(np.array([0.001, 0.02, 0.9, 0.04]), 0.1)
    assert sel.tolist() == [True, True, False, True]
    assert ac.bh(np.array([]), 0.1).size == 0


def test_cs_fdr_controlled_iid():
    rng = np.random.default_rng(1)
    fdps = []
    for _ in range(400):
        n_cal, n_test = 400, 30
        x_cal, x_test = rng.normal(size=n_cal), rng.normal(size=n_test)
        e_cal, e_test = x_cal > 1.28, x_test > 1.28
        s_cal, s_test = x_cal + rng.normal(0, 0.5, n_cal), x_test + rng.normal(0, 0.5, n_test)
        a = ac.bh(ac.conformal_pvalues(s_test, s_cal, ~e_cal), 0.2)
        fdps.append((a & ~e_test).sum() / max(a.sum(), 1))
    assert np.mean(fdps) <= 0.2 + 0.02


def _toy_panel():
    import pandas as pd
    months = pd.date_range("2000-01-01", periods=30, freq="MS")
    rows = []
    for i, t in enumerate(months):
        for r in range(20):
            rows.append({"country": "x", "horizon": 1, "region": r, "origin": t, "target_date": t + pd.DateOffset(months=1),
                         "s_zmargin": float(r) / 10, "event": r >= 18, "eligible": r < 15, "state": 0})
    return pd.DataFrame(rows)


def test_cs_calibrates_on_eligible_cells_only():
    # ineligible cells are high-scoring nulls; if they entered calibration, eligible p-values would be inflated
    d = _toy_panel()
    d.loc[~d.eligible, "event"] = False
    a = ac.run_procedure(d, "cs", q=0.5, min_cal=50)
    d2 = d.copy()
    d2.loc[~d2.eligible, "s_zmargin"] = -99.0  # moving ineligible scores must not change any decision
    a2 = ac.run_procedure(d2, "cs", q=0.5, min_cal=50)
    assert (a == a2).all()
    assert not a[~d.eligible.to_numpy()].any()


def test_paired_gap_metrics_present():
    d = _toy_panel()
    a = ac.run_procedure(d, "point-z", q=1.0)
    m = ac.evaluate(d.assign(onset=False), a, 0.2, n_boot=20)
    assert "gap_pooled" in m and "gap_pooled_lo" in m
