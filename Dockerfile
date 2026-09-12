# Slim runtime image: train once at build time, then serve the artefact.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHURN_MODEL_PATH=/app/artifacts/model.joblib

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY churn ./churn
COPY tests ./tests

# Bake a model into the image so the container is useful on first run.
RUN python -m churn.cli train --customers 20000 --out artifacts/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "churn.service:app", "--host", "0.0.0.0", "--port", "8000"]
