"""Training run, champion selection and the versioned model artefact.

The split is three-way on purpose:

* **train** - fits the candidate models (and the isotonic calibrator, cross-fitted
  inside the training rows only),
* **validation** - selects the champion *and* the campaign threshold,
* **test** - untouched until the very end, used once to report what the committed
  model and threshold would actually have delivered.

Selecting a threshold on the same rows you report it on is one of the quietest
ways to overstate a model, so the code makes that impossible by construction.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from .data import GeneratorConfig, generate_customers
from .evaluate import (
    Campaign,
    best_expected_value,
    calibration_table,
    classification_metrics,
    expected_calibration_error,
    expected_value_table,
    lift_by_decile,
    realised_campaign_value,
)
from .pipeline import FEATURE_CONTRACT, TARGET, build_model, prepare_features

ARTIFACT_VERSION: int = 1


@dataclass
class ModelArtifact:
    """A fitted estimator plus everything needed to use it responsibly."""

    model: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def threshold(self) -> float:
        return float(self.metadata.get("threshold", 0.5))

    @property
    def name(self) -> str:
        return str(self.metadata.get("model_name", "unknown"))

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Churn probability for raw customer rows."""
        prepared = prepare_features(frame)
        return self.model.predict_proba(prepared)[:, 1]

    def decide(self, frame: pd.DataFrame) -> np.ndarray:
        """Boolean targeting decision at the committed threshold."""
        return self.predict_proba(frame) >= self.threshold

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "metadata": self.metadata}, destination)
        return destination


def load_artifact(path: str | Path) -> ModelArtifact:
    """Load an artefact, validating its shape and warning on version drift."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"no model artefact at {source}; run 'python -m churn.cli train' first"
        )
    payload = joblib.load(source)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"{source} is not a churn model artefact")
    metadata = dict(payload.get("metadata", {}))
    trained_with = metadata.get("library_versions", {}).get("scikit_learn")
    if trained_with and trained_with != sklearn.__version__:
        print(
            f"warning: artefact was trained with scikit-learn {trained_with}, "
            f"loading with {sklearn.__version__}"
        )
    return ModelArtifact(model=payload["model"], metadata=metadata)


@dataclass
class TrainingResult:
    """Artefact plus every table the CLI writes out."""

    artifact: ModelArtifact
    comparison: pd.DataFrame
    calibration: pd.DataFrame
    lift: pd.DataFrame
    expected_value: pd.DataFrame
    report: dict[str, Any]


def _three_way_split(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, ...]:
    """60/20/20 stratified split.

    A stratified random split is valid here because the generator has no time
    dimension.  Real churn data must be split by cohort or observation month.
    """
    train_part, holdout = train_test_split(
        frame, test_size=0.4, stratify=frame[TARGET], random_state=seed
    )
    validation, test = train_test_split(
        holdout, test_size=0.5, stratify=holdout[TARGET], random_state=seed
    )
    return train_part.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def _fit_candidates(train_frame: pd.DataFrame, seed: int, calibrate: bool) -> dict[str, Any]:
    features = prepare_features(train_frame)
    target = train_frame[TARGET].to_numpy(dtype=int)

    candidates: dict[str, Any] = {}
    for kind in ("logistic", "gradient_boosting"):
        model = build_model(kind, seed=seed)
        model.fit(features, target)
        candidates[kind] = model

    if calibrate:
        # isotonic regression cross-fitted inside the training rows only
        calibrated = CalibratedClassifierCV(
            build_model("gradient_boosting", seed=seed), method="isotonic", cv=4
        )
        calibrated.fit(features, target)
        candidates["gradient_boosting_calibrated"] = calibrated
    return candidates


def train(
    data_config: GeneratorConfig | None = None,
    campaign: Campaign | None = None,
    seed: int = 13,
    calibrate: bool = True,
    frame: pd.DataFrame | None = None,
) -> TrainingResult:
    """Fit candidates, choose a champion and a threshold, report on the test set."""
    economics = campaign or Campaign()
    economics.validate()
    customers = frame if frame is not None else generate_customers(data_config or GeneratorConfig())
    if TARGET not in customers.columns:
        raise ValueError(f"training data must contain the '{TARGET}' column")

    train_frame, validation_frame, test_frame = _three_way_split(customers, seed)
    candidates = _fit_candidates(train_frame, seed, calibrate)

    validation_target = validation_frame[TARGET].to_numpy(dtype=int)
    validation_features = prepare_features(validation_frame)

    rows: list[dict[str, Any]] = []
    scores: dict[str, np.ndarray] = {}
    for name, model in candidates.items():
        probabilities = model.predict_proba(validation_features)[:, 1]
        scores[name] = probabilities
        rows.append(
            {
                "model": name,
                **classification_metrics(validation_target, probabilities),
                "expected_calibration_error": expected_calibration_error(
                    validation_target, probabilities
                ),
            }
        )
    comparison = pd.DataFrame(rows).sort_values("average_precision", ascending=False)
    champion_name = str(comparison.iloc[0]["model"])
    champion = candidates[champion_name]

    value_table = expected_value_table(validation_target, scores[champion_name], economics)
    chosen = best_expected_value(value_table)
    threshold = float(chosen["threshold"])

    test_target = test_frame[TARGET].to_numpy(dtype=int)
    test_probabilities = champion.predict_proba(prepare_features(test_frame))[:, 1]
    test_metrics = classification_metrics(test_target, test_probabilities)
    test_metrics["expected_calibration_error"] = expected_calibration_error(
        test_target, test_probabilities
    )
    realised = realised_campaign_value(test_target, test_probabilities, threshold, economics)

    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "model_name": champion_name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "threshold": round(threshold, 6),
        "threshold_selected_on": "validation split",
        "feature_contract": FEATURE_CONTRACT.as_dict(),
        "campaign": economics.as_dict(),
        "rows": {
            "train": int(len(train_frame)),
            "validation": int(len(validation_frame)),
            "test": int(len(test_frame)),
        },
        "validation_metrics": {
            key: value
            for key, value in comparison.iloc[0].to_dict().items()
            if key != "model"
        },
        "test_metrics": test_metrics,
        "test_campaign_value": realised,
        "library_versions": {
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "python": platform.python_version(),
        },
    }

    report = {
        "champion": champion_name,
        "threshold": round(threshold, 6),
        "validation_selection": chosen,
        "test_metrics": test_metrics,
        "test_campaign_value": realised,
        "model_comparison": comparison.to_dict(orient="records"),
    }

    return TrainingResult(
        artifact=ModelArtifact(model=champion, metadata=metadata),
        comparison=comparison.reset_index(drop=True),
        calibration=calibration_table(test_target, test_probabilities),
        lift=lift_by_decile(test_target, test_probabilities),
        expected_value=value_table,
        report=report,
    )
