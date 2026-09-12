"""Tests for the feature contract, coercion and the model pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn.pipeline import (
    FEATURE_CONTRACT,
    TARGET,
    build_model,
    coefficient_report,
    prepare_features,
)


def test_prepare_features_returns_the_contract_columns_in_order(small_customers):
    prepared = prepare_features(small_customers)
    assert list(prepared.columns) == FEATURE_CONTRACT.all_features
    assert TARGET not in prepared.columns


def test_prepare_features_coerces_types(small_customers):
    prepared = prepare_features(small_customers)
    for column in FEATURE_CONTRACT.numeric + FEATURE_CONTRACT.boolean:
        assert pd.api.types.is_float_dtype(prepared[column])
    for column in FEATURE_CONTRACT.categorical:
        assert prepared[column].dtype == object


def test_prepare_features_preserves_documented_missingness(small_customers):
    prepared = prepare_features(small_customers)
    # imputation belongs to the pipeline, not to the coercion step
    assert prepared["total_charges"].isna().sum() == small_customers["total_charges"].isna().sum()


def test_prepare_features_fills_missing_categories_with_a_label(small_customers):
    damaged = small_customers.copy()
    damaged.loc[damaged.index[:5], "region"] = None
    prepared = prepare_features(damaged)
    assert (prepared["region"].iloc[:5] == "unknown").all()


def test_prepare_features_rejects_a_missing_column(small_customers):
    with pytest.raises(ValueError, match="missing required features"):
        prepare_features(small_customers.drop(columns=["contract"]))


@pytest.mark.parametrize("kind", ["logistic", "gradient_boosting"])
def test_models_fit_data_that_contains_missing_values(small_customers, kind):
    features = prepare_features(small_customers)
    assert features.isna().any().any(), "fixture should contain missing values"
    model = build_model(kind, seed=3).fit(features, small_customers[TARGET])
    probabilities = model.predict_proba(features)
    assert probabilities.shape == (len(features), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities[:, 1].min() >= 0.0 and probabilities[:, 1].max() <= 1.0


def test_an_unseen_category_does_not_raise_at_inference(small_customers):
    features = prepare_features(small_customers)
    model = build_model("logistic", seed=3).fit(features, small_customers[TARGET])
    novel = features.iloc[[0]].copy()
    novel["contract"] = "quarterly_trial"  # a plan that did not exist during training
    assert 0.0 <= float(model.predict_proba(novel)[0, 1]) <= 1.0


def test_logistic_recovers_the_documented_effect_directions(small_customers):
    features = prepare_features(small_customers)
    model = build_model("logistic", seed=3).fit(features, small_customers[TARGET])
    coefficients = coefficient_report(model).set_index("feature")["coefficient"]
    assert coefficients["tenure_months"] < 0  # tenure protects
    assert coefficients["support_tickets_90d"] > 0  # complaints predict churn
    assert coefficients["contract_month_to_month"] > coefficients["contract_two_year"]


def test_coefficient_report_is_refused_for_tree_models(small_customers):
    features = prepare_features(small_customers)
    model = build_model("gradient_boosting", seed=3).fit(features, small_customers[TARGET])
    with pytest.raises(TypeError):
        coefficient_report(model)


def test_unknown_model_kind_is_rejected():
    with pytest.raises(ValueError, match="kind must be one of"):
        build_model("random_forest")


def test_feature_contract_serialises_for_the_artefact():
    payload = FEATURE_CONTRACT.as_dict()
    assert set(payload) == {"numeric", "boolean", "categorical", "target"}
    assert payload["target"] == [TARGET]
