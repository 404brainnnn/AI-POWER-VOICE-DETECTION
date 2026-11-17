FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements-core.txt requirements-ml.txt ./
RUN python -m pip install --no-cache-dir --prefix=/install -r requirements-core.txt -r requirements-ml.txt

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /app
COPY --from=builder /install /opt/conda
COPY app audio dsp models fusion evaluation datasets pipelines training inference monitoring notifications storage configs pyproject.toml .env.example ./
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
