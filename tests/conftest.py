"""Shared fixtures: one small trained artefact reused across the API tests."""

from __future__ import annotations

import pytest

from churn.data import GeneratorConfig, generate_customers
from churn.evaluate import Campaign
from churn.train import train


@pytest.fixture(scope="session")
def small_customers():
    return generate_customers(GeneratorConfig(n_customers=2_500, seed=101))


@pytest.fixture(scope="session")
def trained(small_customers):
    """Train once per session: calibration off to keep the suite quick."""
    return train(
        campaign=Campaign(offer_cost=10.0, margin_saved=200.0, acceptance_rate=0.4),
        seed=7,
        calibrate=False,
        frame=small_customers,
    )


@pytest.fixture(scope="session")
def artifact(trained):
    return trained.artifact


@pytest.fixture()
def sample_payload() -> dict:
    return {
        "customer_id": "C0000001",
        "tenure_months": 3.0,
        "contract": "month_to_month",
        "monthly_charges": 94.5,
        "total_charges": 283.5,
        "internet_service": "fiber",
        "payment_method": "electronic_check",
        "support_tickets_90d": 4,
        "avg_monthly_usage_gb": 310.0,
        "autopay": False,
        "num_addons": 0,
        "is_senior": False,
        "region": "south",
    }
