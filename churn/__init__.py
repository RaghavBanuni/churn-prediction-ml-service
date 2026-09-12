"""Calibrated customer churn prediction with a deployable service.

Modules
-------
data       synthetic customer generator with documented log-odds drivers
pipeline   preprocessing and model definitions plus the feature contract
evaluate   discrimination, calibration and expected-value threshold selection
train      training run, champion selection and the versioned artefact
service    FastAPI application factory
"""

from .data import GeneratorConfig, generate_customers
from .evaluate import (
    calibration_table,
    classification_metrics,
    expected_value_table,
    lift_by_decile,
)
from .pipeline import FEATURE_CONTRACT, build_model
from .train import ModelArtifact, load_artifact, train

__all__ = [
    "FEATURE_CONTRACT",
    "GeneratorConfig",
    "ModelArtifact",
    "build_model",
    "calibration_table",
    "classification_metrics",
    "expected_value_table",
    "generate_customers",
    "lift_by_decile",
    "load_artifact",
    "train",
]

__version__ = "1.0.0"
