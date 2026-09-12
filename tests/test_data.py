"""Tests for the synthetic customer generator.

Because the data-generating process is written down explicitly, these tests assert
the *direction* of every documented driver.  If a future change breaks the
relationship between contract type and churn, the suite says so.
"""

from __future__ import annotations

import pandas as pd
import pytest

from churn.data import GeneratorConfig, churn_rate_by, generate_customers


def config(**overrides: object) -> GeneratorConfig:
    defaults: dict[str, object] = {"n_customers": 4_000, "seed": 55}
    defaults.update(overrides)
    return GeneratorConfig(**defaults)  # type: ignore[arg-type]


def test_generation_is_reproducible():
    pd.testing.assert_frame_equal(generate_customers(config()), generate_customers(config()))


def test_churn_rate_lands_near_the_target():
    frame = generate_customers(config(target_churn_rate=0.25))
    assert 0.22 < frame["churned"].mean() < 0.28


def test_month_to_month_churns_more_than_two_year():
    frame = generate_customers(config())
    rates = frame.groupby("contract")["churned"].mean()
    assert rates["month_to_month"] > rates["one_year"] > rates["two_year"]


def test_short_tenure_churns_more_than_long_tenure():
    frame = generate_customers(config())
    short = frame[frame["tenure_months"] < 6]["churned"].mean()
    long = frame[frame["tenure_months"] > 36]["churned"].mean()
    assert short > long


def test_support_tickets_increase_churn():
    frame = generate_customers(config())
    quiet = frame[frame["support_tickets_90d"] == 0]["churned"].mean()
    noisy = frame[frame["support_tickets_90d"] >= 3]["churned"].mean()
    assert noisy > quiet


def test_autopay_is_protective():
    frame = generate_customers(config())
    rates = frame.groupby("autopay")["churned"].mean()
    assert rates[True] < rates[False]


def test_missing_values_are_injected_only_where_documented():
    frame = generate_customers(config(missing_rate=0.10))
    missing = frame.isna().sum()
    assert missing["total_charges"] > 0
    assert missing["avg_monthly_usage_gb"] > 0
    assert missing.drop(["total_charges", "avg_monthly_usage_gb"]).sum() == 0


def test_no_missing_values_when_the_rate_is_zero():
    assert generate_customers(config(missing_rate=0.0)).isna().sum().sum() == 0


def test_identifiers_are_unique():
    frame = generate_customers(config())
    assert frame["customer_id"].is_unique


def test_churn_rate_by_returns_one_row_per_level():
    frame = generate_customers(config())
    table = churn_rate_by(frame, "contract")
    assert len(table) == frame["contract"].nunique()
    assert table["customers"].sum() == len(frame)


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        generate_customers(GeneratorConfig(n_customers=10))
    with pytest.raises(ValueError):
        generate_customers(GeneratorConfig(target_churn_rate=0.95))
