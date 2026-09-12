"""API tests: the service is exercised through HTTP, not by calling functions.

``create_app`` takes the artefact directly, so these tests need no files, no
environment variables and no network.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from churn.service import create_app


@pytest.fixture()
def client(artifact):
    with TestClient(create_app(artifact=artifact)) as test_client:
        yield test_client


@pytest.fixture()
def modelless_client(tmp_path):
    with TestClient(create_app(artifact_path=tmp_path / "absent.joblib")) as test_client:
        yield test_client


def test_health_reports_a_loaded_model(client, artifact):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"] == artifact.name


def test_health_is_degraded_without_a_model(modelless_client):
    body = modelless_client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False


def test_metadata_exposes_the_feature_contract(client):
    body = client.get("/metadata").json()
    assert "feature_contract" in body
    assert "threshold" in body
    assert body["feature_contract"]["target"] == ["churned"]


def test_predict_returns_a_probability_and_a_decision(client, sample_payload, artifact):
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["decision"] in {"target", "hold"}
    assert body["threshold"] == pytest.approx(round(artifact.threshold, 6))
    assert body["customer_id"] == sample_payload["customer_id"]
    expected = "target" if body["churn_probability"] >= body["threshold"] else "hold"
    assert body["decision"] == expected


def test_nullable_fields_may_be_omitted(client, sample_payload):
    payload = {
        key: value
        for key, value in sample_payload.items()
        if key not in {"total_charges", "avg_monthly_usage_gb"}
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert 0.0 <= response.json()["churn_probability"] <= 1.0


def test_batch_scores_every_customer(client, sample_payload):
    loyal = {
        **sample_payload,
        "customer_id": "C0000002",
        "contract": "two_year",
        "tenure_months": 60.0,
        "support_tickets_90d": 0,
        "autopay": True,
    }
    response = client.post("/predict/batch", json={"customers": [sample_payload, loyal]})
    assert response.status_code == 200
    body = response.json()
    assert body["scored"] == 2
    assert len(body["predictions"]) == 2
    assert body["targeted"] == sum(
        1 for item in body["predictions"] if item["decision"] == "target"
    )
    risky, safe = body["predictions"]
    assert risky["churn_probability"] > safe["churn_probability"]


def test_scoring_without_a_model_returns_503(modelless_client, sample_payload):
    response = modelless_client.post("/predict", json=sample_payload)
    assert response.status_code == 503
    assert "churn.cli train" in response.json()["detail"]


@pytest.mark.parametrize(
    "mutation",
    [
        {"contract": "weekly"},
        {"tenure_months": -4},
        {"support_tickets_90d": "many"},
        {"region": "antarctica"},
        {"unexpected_field": 1},
    ],
)
def test_invalid_payloads_are_rejected_before_scoring(client, sample_payload, mutation):
    response = client.post("/predict", json={**sample_payload, **mutation})
    assert response.status_code == 422


def test_a_missing_mandatory_field_is_rejected(client, sample_payload):
    payload = {key: value for key, value in sample_payload.items() if key != "contract"}
    assert client.post("/predict", json=payload).status_code == 422


def test_an_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_openapi_schema_documents_both_scoring_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/predict" in paths
    assert "/predict/batch" in paths
