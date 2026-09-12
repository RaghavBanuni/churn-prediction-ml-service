"""Tests for evaluation, calibration and campaign economics.

The economics tests use ten hand-built rows so the expected value can be computed
with a pencil - if the formula drifts, the arithmetic disagrees.
"""

from __future__ import annotations

import numpy as np
import pytest

from churn.evaluate import (
    Campaign,
    best_expected_value,
    calibration_table,
    classification_metrics,
    expected_calibration_error,
    expected_value_table,
    lift_by_decile,
    realised_campaign_value,
)

LABELS = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
SCORES = np.array([0.90, 0.85, 0.80, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10])
CAMPAIGN = Campaign(offer_cost=10.0, margin_saved=100.0, acceptance_rate=0.5)


def test_metrics_report_perfect_separation():
    metrics = classification_metrics(LABELS, SCORES)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["base_rate"] == pytest.approx(0.3)


def test_metrics_require_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        classification_metrics(np.zeros(20, dtype=int), np.linspace(0.1, 0.9, 20))


def test_probabilities_outside_the_unit_interval_are_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        classification_metrics(LABELS, SCORES * 2)


def test_perfect_probabilities_have_no_calibration_error():
    outcomes = np.array([1] * 10 + [0] * 10)
    assert expected_calibration_error(outcomes, outcomes.astype(float)) == pytest.approx(0.0)


def test_calibration_table_covers_every_populated_bin():
    rng = np.random.default_rng(0)
    probabilities = rng.uniform(0.0, 1.0, 2_000)
    outcomes = rng.binomial(1, probabilities)
    table = calibration_table(outcomes, probabilities, n_bins=10)
    assert table["customers"].sum() == 2_000
    assert len(table) == 10
    # a well-specified model should be close to the diagonal
    assert table["gap"].abs().max() < 0.10


def test_overconfident_probabilities_are_detected():
    rng = np.random.default_rng(1)
    truth = rng.uniform(0.05, 0.45, 4_000)
    outcomes = rng.binomial(1, truth)
    overconfident = np.clip(truth * 2.2, 0.0, 1.0)
    assert expected_calibration_error(outcomes, overconfident) > expected_calibration_error(
        outcomes, truth
    )


def test_lift_deciles_rank_correctly():
    outcomes = np.array([1] * 20 + [0] * 80)
    scores = np.linspace(1.0, 0.0, 100)
    table = lift_by_decile(outcomes, scores)
    assert len(table) == 10
    assert table.loc[0, "churn_rate"] == pytest.approx(1.0)
    assert table.loc[0, "lift"] == pytest.approx(5.0)
    assert table.loc[1, "cumulative_capture"] == pytest.approx(1.0)
    assert table["customers"].sum() == 100


def test_expected_value_matches_the_hand_computation():
    table = expected_value_table(LABELS, SCORES, CAMPAIGN)
    best = best_expected_value(table)
    # contacting the three churners: 3 * 0.5 * 100 - 3 * 10 = 120
    assert best["net_value"] == pytest.approx(120.0)
    assert best["contacted"] == 3
    assert best["precision"] == pytest.approx(1.0)


def test_contacting_everyone_is_worse_than_targeting():
    table = expected_value_table(LABELS, SCORES, CAMPAIGN)
    contact_all = table.iloc[0]
    assert contact_all["contacted"] == 10
    assert contact_all["net_value"] == pytest.approx(50.0)
    assert contact_all["net_value"] < best_expected_value(table)["net_value"]


def test_an_expensive_offer_shrinks_the_targeted_group():
    cheap = best_expected_value(expected_value_table(LABELS, SCORES, CAMPAIGN))
    expensive = best_expected_value(
        expected_value_table(
            LABELS, SCORES, Campaign(offer_cost=45.0, margin_saved=100.0, acceptance_rate=0.5)
        )
    )
    assert expensive["contacted"] <= cheap["contacted"]
    assert expensive["net_value"] < cheap["net_value"]


def test_realised_value_applies_a_committed_threshold():
    realised = realised_campaign_value(LABELS, SCORES, threshold=0.5, campaign=CAMPAIGN)
    assert realised["contacted"] == 3
    assert realised["recall"] == pytest.approx(1.0)
    assert realised["net_value"] == pytest.approx(120.0)


def test_invalid_campaign_economics_are_rejected():
    with pytest.raises(ValueError):
        Campaign(offer_cost=10.0, margin_saved=0.0, acceptance_rate=0.5).validate()
    with pytest.raises(ValueError):
        Campaign(offer_cost=10.0, margin_saved=100.0, acceptance_rate=1.5).validate()
