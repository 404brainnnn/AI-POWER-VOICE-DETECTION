FROM python:3.11-slim
WORKDIR /app
COPY requirements-core.txt .
RUN pip install --no-cache-dir -r requirements-core.txt
COPY app audio dsp models fusion evaluation datasets pipelines monitoring notifications storage tests pyproject.toml .env.example ./
RUN useradd --create-home --uid 10001 appuser
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
