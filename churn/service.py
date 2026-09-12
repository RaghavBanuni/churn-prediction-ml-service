"""FastAPI inference service.

Three properties this module is built around:

* **it never guesses.** Without a loaded artefact the scoring endpoints return
  ``503`` and say what to run, instead of returning a plausible-looking number.
* **the request schema is the feature contract.** Pydantic enforces types, ranges
  and allowed categories, so malformed payloads fail with ``422`` before the model
  is touched.  Optional fields are exactly the two columns whose missingness the
  pipeline was trained to impute.
* **it is testable.** :func:`create_app` accepts an artefact directly, so the API
  tests need no files, no environment variables and no network.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .train import ModelArtifact, load_artifact

DEFAULT_ARTIFACT_PATH = Path(os.environ.get("CHURN_MODEL_PATH", "artifacts/model.joblib"))
MAX_BATCH_SIZE: int = 500


class CustomerFeatures(BaseModel):
    """One customer as the model expects them.

    ``total_charges`` and ``avg_monthly_usage_gb`` are nullable because the trained
    pipeline imputes them; every other field is mandatory, because silently
    defaulting a driver of the prediction would be dishonest.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    customer_id: str | None = Field(default=None, max_length=64)
    tenure_months: float = Field(ge=0.0, le=600.0)
    contract: Literal["month_to_month", "one_year", "two_year"]
    monthly_charges: float = Field(ge=0.0, le=1_000.0)
    total_charges: float | None = Field(default=None, ge=0.0)
    internet_service: Literal["fiber", "dsl", "none"]
    payment_method: Literal[
        "bank_transfer", "credit_card", "electronic_check", "mailed_check"
    ]
    support_tickets_90d: int = Field(ge=0, le=200)
    avg_monthly_usage_gb: float | None = Field(default=None, ge=0.0)
    autopay: bool
    num_addons: int = Field(ge=0, le=20)
    is_senior: bool
    region: Literal["north", "south", "east", "west"]


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: list[CustomerFeatures] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class Prediction(BaseModel):
    customer_id: str | None
    churn_probability: float
    decision: Literal["target", "hold"]
    threshold: float


class PredictionResponse(Prediction):
    model_name: str


class BatchResponse(BaseModel):
    model_name: str
    threshold: float
    scored: int
    targeted: int
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service_version: str
    model_loaded: bool
    model_name: str | None


def _to_frame(customers: list[CustomerFeatures]) -> pd.DataFrame:
    return pd.DataFrame([customer.model_dump() for customer in customers])


def _predict(artifact: ModelArtifact, customers: list[CustomerFeatures]) -> list[Prediction]:
    probabilities = artifact.predict_proba(_to_frame(customers))
    threshold = artifact.threshold
    return [
        Prediction(
            customer_id=customer.customer_id,
            churn_probability=round(float(probability), 6),
            decision="target" if probability >= threshold else "hold",
            threshold=round(threshold, 6),
        )
        for customer, probability in zip(customers, probabilities)
    ]


def create_app(
    artifact: ModelArtifact | None = None, artifact_path: str | Path | None = None
) -> FastAPI:
    """Build the application, optionally with an in-memory artefact for tests."""
    state: dict[str, ModelArtifact | None] = {"artifact": artifact}
    path = Path(artifact_path) if artifact_path is not None else DEFAULT_ARTIFACT_PATH

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if state["artifact"] is None and path.exists():
            state["artifact"] = load_artifact(path)
        yield
        state["artifact"] = None

    app = FastAPI(
        title="Churn Prediction Service",
        version=__version__,
        summary="Calibrated churn probabilities with a cost-based targeting decision.",
        lifespan=lifespan,
    )

    def require_artifact() -> ModelArtifact:
        loaded = state["artifact"]
        if loaded is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"no model loaded from {path}; train one with "
                    "'python -m churn.cli train' or set CHURN_MODEL_PATH"
                ),
            )
        return loaded

    Artifact = Annotated[ModelArtifact, Depends(require_artifact)]

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        """Liveness plus whether the service can actually score."""
        loaded = state["artifact"]
        return HealthResponse(
            status="ok" if loaded is not None else "degraded",
            service_version=__version__,
            model_loaded=loaded is not None,
            model_name=loaded.name if loaded is not None else None,
        )

    @app.get("/metadata", tags=["ops"])
    def metadata(model: Artifact) -> dict[str, Any]:
        """Feature contract, metrics, threshold and library versions."""
        return model.metadata

    @app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
    def predict(customer: CustomerFeatures, model: Artifact) -> PredictionResponse:
        """Score one customer."""
        prediction = _predict(model, [customer])[0]
        return PredictionResponse(**prediction.model_dump(), model_name=model.name)

    @app.post("/predict/batch", response_model=BatchResponse, tags=["scoring"])
    def predict_batch(request: BatchRequest, model: Artifact) -> BatchResponse:
        """Score up to 500 customers in one call."""
        predictions = _predict(model, request.customers)
        return BatchResponse(
            model_name=model.name,
            threshold=round(model.threshold, 6),
            scored=len(predictions),
            targeted=sum(1 for item in predictions if item.decision == "target"),
            predictions=predictions,
        )

    return app


# Module-level application for 'uvicorn churn.service:app'.
app = create_app()
