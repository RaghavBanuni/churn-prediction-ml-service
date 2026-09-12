"""Feature contract, preprocessing and model definitions.

The feature contract is a first-class object rather than a comment.  Training,
the artefact metadata and the API request model all derive their column lists
from :data:`FEATURE_CONTRACT`, so a feature added in one place cannot silently go
missing in another - the mismatch raises instead.

All preprocessing lives *inside* the estimator.  That is what makes the saved
artefact self-contained: the service never re-implements imputation or encoding,
it hands raw customer fields to the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET: str = "churned"
ID_COLUMN: str = "customer_id"
MODEL_KINDS: tuple[str, ...] = ("logistic", "gradient_boosting")


@dataclass(frozen=True)
class FeatureContract:
    """The columns the model consumes, grouped by how they must be handled."""

    numeric: tuple[str, ...]
    boolean: tuple[str, ...]
    categorical: tuple[str, ...]
    target: str = TARGET

    @property
    def all_features(self) -> list[str]:
        return [*self.numeric, *self.boolean, *self.categorical]

    def validate(self, frame: pd.DataFrame) -> None:
        missing = [column for column in self.all_features if column not in frame.columns]
        if missing:
            raise ValueError(f"input is missing required features: {missing}")

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "numeric": list(self.numeric),
            "boolean": list(self.boolean),
            "categorical": list(self.categorical),
            "target": [self.target],
        }


FEATURE_CONTRACT = FeatureContract(
    numeric=(
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "support_tickets_90d",
        "avg_monthly_usage_gb",
        "num_addons",
    ),
    boolean=("autopay", "is_senior"),
    categorical=("contract", "internet_service", "payment_method", "region"),
)


def prepare_features(
    frame: pd.DataFrame, contract: FeatureContract = FEATURE_CONTRACT
) -> pd.DataFrame:
    """Coerce raw input into the exact frame the pipeline was fitted on.

    Called by both training and the service, which is the point: any coercion the
    model depends on happens in one function, not twice with slight differences.
    """
    contract.validate(frame)
    prepared = frame.loc[:, contract.all_features].copy()
    for column in contract.numeric:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    for column in contract.boolean:
        prepared[column] = prepared[column].astype("boolean").astype("float64")
    for column in contract.categorical:
        prepared[column] = (
            prepared[column].astype("string").fillna("unknown").astype(object)
        )
    return prepared


def build_preprocessor(contract: FeatureContract = FEATURE_CONTRACT) -> ColumnTransformer:
    """Impute, scale and encode - fitted on training folds only."""
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    binary = Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent"))])
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            # an unseen category at inference must not raise - it encodes as all zeros
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, list(contract.numeric)),
            ("binary", binary, list(contract.boolean)),
            ("categorical", categorical, list(contract.categorical)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model(
    kind: str = "gradient_boosting",
    contract: FeatureContract = FEATURE_CONTRACT,
    seed: int = 13,
) -> Pipeline:
    """Preprocessing plus an estimator, as one fittable object.

    ``logistic`` is the interpretable baseline; ``gradient_boosting`` is the
    challenger.  Both carry ``class_weight``-style handling of the ~20% positive
    rate: the logistic model through ``class_weight='balanced'``, the booster
    through the log-loss objective plus a calibrated wrapper downstream.
    """
    if kind not in MODEL_KINDS:
        raise ValueError(f"kind must be one of {MODEL_KINDS}, got {kind!r}")
    estimator = (
        LogisticRegression(
            class_weight="balanced", max_iter=2_000, C=0.5, solver="lbfgs", random_state=seed
        )
        if kind == "logistic"
        else HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.06,
            max_iter=400,
            max_depth=5,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=seed,
        )
    )
    return Pipeline(
        steps=[("preprocess", build_preprocessor(contract)), ("model", estimator)]
    )


def coefficient_report(pipeline: Pipeline) -> pd.DataFrame:
    """Signed logistic coefficients, strongest first - the interpretability view.

    Coefficients are on standardised inputs, so their magnitudes are comparable
    across features.  Raises for tree models, where coefficients do not exist.
    """
    estimator = pipeline.named_steps["model"]
    if not hasattr(estimator, "coef_"):
        raise TypeError("coefficient_report requires a linear model")
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefficients = estimator.coef_.ravel()
    table = pd.DataFrame({"feature": names, "coefficient": coefficients.round(5)})
    table["abs_coefficient"] = table["coefficient"].abs()
    return table.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
