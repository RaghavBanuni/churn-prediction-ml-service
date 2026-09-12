"""Tests for the training run and the model artefact contract."""

from __future__ import annotations

import joblib
import numpy as np
import pytest

from churn.train import load_artifact


def test_split_sizes_are_sixty_twenty_twenty(trained, small_customers):
    rows = trained.artifact.metadata["rows"]
    total = sum(rows.values())
    assert total == len(small_customers)
    assert rows["train"] / total == pytest.approx(0.60, abs=0.01)
    assert rows["validation"] / total == pytest.approx(0.20, abs=0.01)


def test_champion_beats_a_coin_flip_on_held_out_data(trained):
    assert trained.artifact.metadata["test_metrics"]["roc_auc"] > 0.65
    assert (
        trained.artifact.metadata["test_metrics"]["average_precision"]
        > trained.artifact.metadata["test_metrics"]["base_rate"]
    )


def test_threshold_is_a_probability_selected_on_validation(trained):
    assert 0.0 <= trained.artifact.threshold <= 1.0
    assert trained.artifact.metadata["threshold_selected_on"] == "validation split"


def test_metadata_records_what_is_needed_to_reproduce_the_run(trained):
    metadata = trained.artifact.metadata
    for key in (
        "artifact_version",
        "model_name",
        "trained_at",
        "threshold",
        "feature_contract",
        "campaign",
        "validation_metrics",
        "test_metrics",
        "library_versions",
    ):
        assert key in metadata
    assert metadata["library_versions"]["scikit_learn"]


def test_artefact_scores_raw_rows_including_missing_values(trained, small_customers):
    probabilities = trained.artifact.predict_proba(small_customers.head(50))
    assert probabilities.shape == (50,)
    assert probabilities.min() >= 0.0 and probabilities.max() <= 1.0


def test_decisions_follow_the_committed_threshold(trained, small_customers):
    sample = small_customers.head(200)
    probabilities = trained.artifact.predict_proba(sample)
    decisions = trained.artifact.decide(sample)
    assert np.array_equal(decisions, probabilities >= trained.artifact.threshold)


def test_save_and_load_round_trip_preserves_predictions(trained, small_customers, tmp_path):
    path = trained.artifact.save(tmp_path / "model.joblib")
    assert path.exists()
    reloaded = load_artifact(path)
    assert reloaded.name == trained.artifact.name
    assert reloaded.threshold == pytest.approx(trained.artifact.threshold)
    sample = small_customers.head(100)
    assert np.allclose(
        reloaded.predict_proba(sample), trained.artifact.predict_proba(sample)
    )


def test_loading_a_missing_artefact_explains_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="churn.cli train"):
        load_artifact(tmp_path / "absent.joblib")


def test_loading_a_foreign_joblib_file_is_refused(tmp_path):
    path = tmp_path / "not_a_model.joblib"
    joblib.dump({"something": "else"}, path)
    with pytest.raises(ValueError, match="not a churn model artefact"):
        load_artifact(path)


def test_reported_tables_are_non_empty(trained):
    assert not trained.comparison.empty
    assert len(trained.lift) == 10
    assert not trained.expected_value.empty
    assert "champion" in trained.report
