# Operations

Local: activate a virtual environment, then run `python -m pip install -r requirements-core.txt` and `python -m uvicorn app.main:app --reload`, or use `docker compose up --build`. Stop with `docker compose down`; named volumes contain only MongoDB/MLflow metadata. Production requires TLS at a trusted proxy, authentication, secrets management, resource limits, crash-dump/swap controls, and external Redis/Mongo/MLflow hardening. Docker is optional for direct local development.
