# Installation in restricted environments

Use the repository environment explicitly when it exists:

```fish
python -m pip install -r requirements-core.txt
python -m uvicorn app.main:app --reload
python scripts/run_stream_simulation.py
```

The system Python and Fish PATH do not necessarily include repository entry points. If PyPI access times out, install from an approved internal wheelhouse or configure a proxy; do not mark the installation successful. Docker commands require Docker Engine separately and are unavailable when `docker` is not installed.
