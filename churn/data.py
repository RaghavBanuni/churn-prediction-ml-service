"""Synthetic subscription customers with a transparent churn mechanism.

The generator is deliberately an explicit log-odds model rather than a black box:
because the true drivers are written down here, the tests can assert that the
model recovers them, and the README can describe the data honestly.

Two details exist purely to make the modelling realistic:

* **interaction** - price sensitivity is much stronger on month-to-month
  contracts, so ``monthly_charges`` and ``contract`` interact;
* **missing values** - ``total_charges`` and ``avg_monthly_usage_gb`` have
  missing-at-random gaps, so the pipeline must impute rather than assume clean
  input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CONTRACTS: tuple[str, ...] = ("month_to_month", "one_year", "two_year")
INTERNET_SERVICES: tuple[str, ...] = ("fiber", "dsl", "none")
PAYMENT_METHODS: tuple[str, ...] = (
    "bank_transfer",
    "credit_card",
    "electronic_check",
    "mailed_check",
)
REGIONS: tuple[str, ...] = ("north", "south", "east", "west")

# Log-odds coefficients of the data-generating process.
CONTRACT_EFFECT: dict[str, float] = {
    "month_to_month": 0.95,
    "one_year": -0.35,
    "two_year": -1.10,
}
PAYMENT_EFFECT: dict[str, float] = {
    "bank_transfer": -0.10,
    "credit_card": -0.15,
    "electronic_check": 0.40,
    "mailed_check": 0.25,
}
INTERNET_EFFECT: dict[str, float] = {"fiber": 0.30, "dsl": 0.0, "none": -0.25}


@dataclass(frozen=True)
class GeneratorConfig:
    """Generator settings.

    n_customers:
        Number of rows.
    target_churn_rate:
        Approximate share of churners; the intercept is solved for numerically.
    missing_rate:
        Share of missing values injected into two numeric columns.
    """

    n_customers: int = 20_000
    target_churn_rate: float = 0.20
    missing_rate: float = 0.04
    seed: int = 29
    contract_mix: tuple[float, float, float] = (0.55, 0.25, 0.20)
    region_mix: tuple[float, ...] = field(default=(0.25, 0.28, 0.24, 0.23))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _solve_intercept(log_odds: np.ndarray, target_rate: float) -> float:
    """Bisect for the intercept that produces the requested churn rate."""
    low, high = -12.0, 12.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if _sigmoid(log_odds + middle).mean() < target_rate:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def generate_customers(config: GeneratorConfig | None = None) -> pd.DataFrame:
    """Return a customer table with a ``churned`` label."""
    cfg = config or GeneratorConfig()
    if cfg.n_customers < 100:
        raise ValueError("n_customers must be at least 100")
    if not 0.01 <= cfg.target_churn_rate <= 0.60:
        raise ValueError("target_churn_rate must be between 0.01 and 0.60")
    if not 0.0 <= cfg.missing_rate < 0.5:
        raise ValueError("missing_rate must be in [0, 0.5)")

    rng = np.random.default_rng(cfg.seed)
    size = cfg.n_customers

    contract = rng.choice(CONTRACTS, size=size, p=cfg.contract_mix)
    # longer contracts correlate with longer tenure, as they do in real books
    tenure_scale = np.where(contract == "month_to_month", 14.0, 34.0)
    tenure_months = np.clip(rng.gamma(shape=2.0, scale=tenure_scale / 2.0, size=size), 1.0, 72.0)

    internet_service = rng.choice(INTERNET_SERVICES, size=size, p=(0.45, 0.35, 0.20))
    base_charge = np.where(
        internet_service == "fiber", 78.0, np.where(internet_service == "dsl", 52.0, 24.0)
    )
    num_addons = rng.binomial(4, np.where(internet_service == "none", 0.10, 0.35), size=size)
    monthly_charges = np.round(
        base_charge + 6.5 * num_addons + rng.normal(0.0, 6.0, size), 2
    ).clip(15.0, 160.0)
    total_charges = np.round(monthly_charges * tenure_months * rng.uniform(0.93, 1.03, size), 2)

    payment_method = rng.choice(PAYMENT_METHODS, size=size, p=(0.24, 0.26, 0.30, 0.20))
    autopay = np.where(
        np.isin(payment_method, ("bank_transfer", "credit_card")),
        rng.random(size) < 0.80,
        rng.random(size) < 0.15,
    )
    support_tickets = rng.poisson(
        np.where(internet_service == "fiber", 1.1, 0.7) + 0.4 * (tenure_months < 6), size
    )
    usage_base = np.where(
        internet_service == "fiber", 320.0, np.where(internet_service == "dsl", 130.0, 8.0)
    )
    avg_usage_gb = np.round(np.clip(rng.normal(usage_base, usage_base * 0.30), 0.0, None), 1)
    is_senior = rng.random(size) < 0.16
    region = rng.choice(REGIONS, size=size, p=cfg.region_mix)

    contract_term = np.array([CONTRACT_EFFECT[value] for value in contract])
    payment_term = np.array([PAYMENT_EFFECT[value] for value in payment_method])
    internet_term = np.array([INTERNET_EFFECT[value] for value in internet_service])
    charge_z = (monthly_charges - monthly_charges.mean()) / monthly_charges.std()

    log_odds = (
        contract_term
        + payment_term
        + internet_term
        - 0.85 * np.log1p(tenure_months)
        + 0.45 * charge_z
        # price sensitivity bites hardest where leaving is free
        + 0.35 * charge_z * (contract == "month_to_month")
        + 0.22 * support_tickets
        - 0.45 * autopay
        - 0.10 * num_addons
        + 0.18 * is_senior
        + rng.normal(0.0, 0.35, size)  # unobserved heterogeneity
    )
    log_odds += _solve_intercept(log_odds, cfg.target_churn_rate)
    churned = (rng.random(size) < _sigmoid(log_odds)).astype(int)

    frame = pd.DataFrame(
        {
            "customer_id": [f"C{index:07d}" for index in range(size)],
            "tenure_months": np.round(tenure_months, 1),
            "contract": contract,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "support_tickets_90d": support_tickets.astype(int),
            "avg_monthly_usage_gb": avg_usage_gb,
            "autopay": autopay,
            "num_addons": num_addons.astype(int),
            "is_senior": is_senior,
            "region": region,
            "churned": churned,
        }
    )

    if cfg.missing_rate > 0:
        for column in ("total_charges", "avg_monthly_usage_gb"):
            gaps = rng.random(size) < cfg.missing_rate
            frame.loc[gaps, column] = np.nan

    return frame


def churn_rate_by(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Churn rate and volume per level of a column - the first sanity check."""
    if column not in frame.columns:
        raise ValueError(f"unknown column: {column}")
    grouped = frame.groupby(column, dropna=False)["churned"]
    table = pd.DataFrame(
        {"customers": grouped.size(), "churn_rate": grouped.mean().round(4)}
    ).reset_index()
    return table.sort_values("churn_rate", ascending=False).reset_index(drop=True)
