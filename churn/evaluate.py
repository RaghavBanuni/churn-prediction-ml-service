"""Discrimination, calibration and campaign economics.

Three layers of evaluation, in increasing order of usefulness to the business:

1. **ranking** - ROC-AUC and average precision: can the model order customers?
2. **calibration** - Brier score, reliability bins and expected calibration
   error: do the probabilities mean what they claim?  A retention budget built on
   a model that says 40% when the truth is 15% overspends by design.
3. **decision value** - the expected-value sweep: given what an offer costs, what
   a saved customer is worth and how often offers are accepted, which threshold
   maximises net value?
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


@dataclass(frozen=True)
class Campaign:
    """Economics of a retention campaign.

    offer_cost:
        Cost of contacting one customer with the retention offer.
    margin_saved:
        Margin preserved when a would-be churner is retained.
    acceptance_rate:
        Share of contacted churners who accept and stay.
    """

    offer_cost: float = 12.0
    margin_saved: float = 220.0
    acceptance_rate: float = 0.35

    def validate(self) -> None:
        if self.offer_cost < 0 or self.margin_saved <= 0:
            raise ValueError("offer_cost must be >= 0 and margin_saved > 0")
        if not 0.0 < self.acceptance_rate <= 1.0:
            raise ValueError("acceptance_rate must be in (0, 1]")

    @property
    def value_per_saved_churner(self) -> float:
        return self.acceptance_rate * self.margin_saved

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _checked(y_true, probabilities) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(probabilities, dtype=float).ravel()
    if y.size != p.size:
        raise ValueError("labels and probabilities must have the same length")
    if y.size < 10:
        raise ValueError("at least ten observations are required")
    if p.min() < 0.0 or p.max() > 1.0:
        raise ValueError("probabilities must lie in [0, 1]")
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("labels must be binary 0/1")
    if y.sum() == 0 or y.sum() == y.size:
        raise ValueError("evaluation needs both classes present")
    return y, p


def classification_metrics(y_true, probabilities) -> dict[str, float]:
    """Ranking and probability quality in one block."""
    y, p = _checked(y_true, probabilities)
    return {
        "n": int(y.size),
        "base_rate": round(float(y.mean()), 5),
        "roc_auc": round(float(roc_auc_score(y, p)), 5),
        "average_precision": round(float(average_precision_score(y, p)), 5),
        "brier": round(float(brier_score_loss(y, p)), 5),
        "log_loss": round(float(log_loss(y, np.clip(p, 1e-9, 1 - 1e-9))), 5),
    }


def calibration_table(y_true, probabilities, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table: predicted probability against observed frequency."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    y, p = _checked(y_true, probabilities)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-closed bins so that a prediction of exactly 1.0 lands in the last bin
    assignment = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)

    rows: list[dict[str, float]] = []
    for index in range(n_bins):
        mask = assignment == index
        if not mask.any():
            continue
        rows.append(
            {
                "bin": index + 1,
                "lower": round(float(edges[index]), 3),
                "upper": round(float(edges[index + 1]), 3),
                "customers": int(mask.sum()),
                "mean_predicted": round(float(p[mask].mean()), 5),
                "observed_rate": round(float(y[mask].mean()), 5),
                "gap": round(float(p[mask].mean() - y[mask].mean()), 5),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, probabilities, n_bins: int = 10) -> float:
    """Volume-weighted mean absolute gap between predicted and observed rates."""
    table = calibration_table(y_true, probabilities, n_bins)
    weights = table["customers"].to_numpy(dtype=float)
    gaps = table["gap"].abs().to_numpy(dtype=float)
    return round(float(np.average(gaps, weights=weights)), 5)


def lift_by_decile(y_true, probabilities) -> pd.DataFrame:
    """Churn rate, lift and cumulative capture by risk decile (1 = riskiest)."""
    y, p = _checked(y_true, probabilities)
    n = y.size
    if n < 20:
        raise ValueError("deciles need at least twenty observations")

    order = np.argsort(-p, kind="mergesort")
    edges = np.linspace(0, n, 11).astype(int)
    decile = np.empty(n, dtype=int)
    for index in range(10):
        decile[order[edges[index] : edges[index + 1]]] = index + 1

    base_rate = float(y.mean())
    total_churners = int(y.sum())
    rows: list[dict[str, float]] = []
    captured = 0
    for index in range(1, 11):
        mask = decile == index
        churners = int(y[mask].sum())
        captured += churners
        rows.append(
            {
                "decile": index,
                "customers": int(mask.sum()),
                "mean_predicted": round(float(p[mask].mean()), 5),
                "churners": churners,
                "churn_rate": round(float(y[mask].mean()), 5),
                "lift": round(float(y[mask].mean() / base_rate) if base_rate else 0.0, 3),
                "cumulative_capture": round(captured / total_churners, 5),
            }
        )
    return pd.DataFrame(rows)


def expected_value_table(
    y_true,
    probabilities,
    campaign: Campaign | None = None,
    n_thresholds: int = 40,
) -> pd.DataFrame:
    """Net campaign value at every candidate targeting threshold.

    ``net_value = saved_churners * acceptance_rate * margin_saved
                  - contacted * offer_cost``
    """
    economics = campaign or Campaign()
    economics.validate()
    y, p = _checked(y_true, probabilities)

    quantiles = np.linspace(0.0, 0.99, max(n_thresholds, 2))
    thresholds = np.unique(np.round(np.quantile(p, quantiles), 6))
    total_churners = int(y.sum())

    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        targeted = p >= threshold
        contacted = int(targeted.sum())
        if contacted == 0:
            continue
        churners_targeted = int(y[targeted].sum())
        benefit = churners_targeted * economics.value_per_saved_churner
        cost = contacted * economics.offer_cost
        rows.append(
            {
                "threshold": float(threshold),
                "contacted": contacted,
                "contact_rate": round(contacted / y.size, 5),
                "precision": round(churners_targeted / contacted, 5),
                "recall": round(churners_targeted / total_churners, 5),
                "campaign_cost": round(cost, 2),
                "expected_benefit": round(benefit, 2),
                "net_value": round(benefit - cost, 2),
                "net_value_per_contact": round((benefit - cost) / contacted, 4),
            }
        )
    if not rows:
        raise ValueError("no threshold targeted any customer")
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def best_expected_value(table: pd.DataFrame) -> dict[str, float]:
    """Highest-net-value row; ties break towards contacting fewer customers."""
    if table.empty:
        raise ValueError("expected value table is empty")
    best_value = table["net_value"].max()
    candidates = table[table["net_value"] == best_value]
    chosen = candidates.loc[candidates["contacted"].idxmin()]
    return {key: float(value) for key, value in chosen.to_dict().items()}


def realised_campaign_value(
    y_true, probabilities, threshold: float, campaign: Campaign | None = None
) -> dict[str, float]:
    """Value of applying a *pre-committed* threshold to fresh data.

    Selecting the threshold and reporting its value on the same rows flatters the
    result; the training run therefore fixes the threshold on validation data and
    calls this on the untouched test set.
    """
    economics = campaign or Campaign()
    economics.validate()
    y, p = _checked(y_true, probabilities)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")

    targeted = p >= threshold
    contacted = int(targeted.sum())
    churners_targeted = int(y[targeted].sum())
    benefit = churners_targeted * economics.value_per_saved_churner
    cost = contacted * economics.offer_cost
    return {
        "threshold": float(threshold),
        "contacted": contacted,
        "contact_rate": round(contacted / y.size, 5),
        "precision": round(churners_targeted / contacted, 5) if contacted else 0.0,
        "recall": round(churners_targeted / int(y.sum()), 5),
        "net_value": round(benefit - cost, 2),
        "net_value_per_customer": round((benefit - cost) / y.size, 4),
    }
