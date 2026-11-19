#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
# Demo-safe defaults: start only the inference API. Optional services are
# enabled explicitly with START_INFRA=1 and START_FRONTEND=1.
START_INFRA="${START_INFRA:-0}"
START_FRONTEND="${START_FRONTEND:-0}"
START_BACKEND="${START_BACKEND:-1}"
STOP_INFRA_ON_EXIT="${STOP_INFRA_ON_EXIT:-0}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

BACKEND_PID=""
FRONTEND_PID=""

usage() {
  cat <<'EOF'
Usage: ./start.sh [command]

Commands:
  start       Start the FastAPI inference backend (default)
  stop        Stop local processes and Docker Compose services
  status      Show local process and Docker Compose status
  simulate    Run the offline VoIP simulation using its normal CLI options
  help        Show this help

Environment switches:
  START_INFRA=1             Also start Redis/MongoDB/MLflow with Docker
  START_FRONTEND=1          Also start the Next.js frontend
  START_BACKEND=0           Do not start FastAPI
  STOP_INFRA_ON_EXIT=1      Stop Docker services when this script exits
  API_PORT=8000             FastAPI port
  FRONTEND_PORT=3000        Next.js port
EOF
}

load_environment() {
  if [[ -f "$ROOT_DIR/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.env"
    set +a
  fi
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: required command not found: $1" >&2
    exit 1
  }
}

start_infrastructure() {
  if [[ "$START_INFRA" != "1" ]]; then
    echo "Skipping Docker infrastructure (START_INFRA=$START_INFRA)."
    return
  fi
  if ! command -v docker >/dev/null 2>&1; then
    echo "WARNING: Docker is not installed; skipping Redis, MongoDB, and MLflow."
    echo "Install Docker or rerun with START_INFRA=0 to suppress this warning."
    return
  fi
  echo "Starting Redis, MongoDB, and MLflow..."
  docker compose up -d redis mongo mlflow
}

start_backend() {
  if [[ "$START_BACKEND" != "1" ]]; then
    echo "Skipping FastAPI backend (START_BACKEND=$START_BACKEND)."
    return
  fi
  [[ -x "$PYTHON_BIN" ]] || {
    echo "ERROR: Python environment not found at $PYTHON_BIN" >&2
    echo "Create it with: python3.12 -m venv .venv && .venv/bin/python -m pip install -e ." >&2
    exit 1
  }
  "$PYTHON_BIN" -c 'import fastapi, uvicorn' >/dev/null 2>&1 || {
    echo "ERROR: FastAPI/Uvicorn are not installed in $PYTHON_BIN" >&2
    echo "Install with: $PYTHON_BIN -m pip install -e ." >&2
    exit 1
  }
  echo "Starting FastAPI on http://localhost:$API_PORT ..."
  mkdir -p "$LOG_DIR"
  "$PYTHON_BIN" -m uvicorn app.main:app --host "$API_HOST" --port "$API_PORT" --reload \
    >"$LOG_DIR/backend.log" 2>&1 &
  BACKEND_PID=$!
}

start_frontend() {
  if [[ "$START_FRONTEND" != "1" ]]; then
    echo "Skipping Next.js frontend (START_FRONTEND=$START_FRONTEND)."
    return
  fi
  require_command npm
  [[ -f "$ROOT_DIR/frontend/package.json" ]] || {
    echo "ERROR: frontend/package.json not found" >&2
    exit 1
  }
  if [[ ! -x "$ROOT_DIR/frontend/node_modules/.bin/next" ]]; then
    echo "Installing frontend dependencies..."
    (cd "$ROOT_DIR/frontend" && npm install)
  fi
  [[ -x "$ROOT_DIR/frontend/node_modules/.bin/next" ]] || {
    echo "ERROR: Next.js was not installed in frontend/node_modules" >&2
    echo "Run: cd frontend && npm install" >&2
    exit 1
  }
  echo "Starting Next.js on http://localhost:$FRONTEND_PORT ..."
  mkdir -p "$LOG_DIR"
  (
    cd "$ROOT_DIR/frontend"
    NEXT_PUBLIC_API_BASE_URL="${NEXT_PUBLIC_API_BASE_URL:-http://localhost:$API_PORT}" \
      npm run dev -- --hostname 0.0.0.0 --port "$FRONTEND_PORT"
  ) >"$LOG_DIR/frontend.log" 2>&1 &
  FRONTEND_PID=$!
  sleep 2
  if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    echo "ERROR: Next.js exited during startup. See $LOG_DIR/frontend.log" >&2
    sed -n '1,80p' "$LOG_DIR/frontend.log" >&2 || true
    exit 1
  fi
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  [[ -n "$FRONTEND_PID" ]] && kill "$FRONTEND_PID" 2>/dev/null || true
  [[ -n "$BACKEND_PID" ]] && kill "$BACKEND_PID" 2>/dev/null || true
  if [[ "$STOP_INFRA_ON_EXIT" == "1" ]] && command -v docker >/dev/null 2>&1; then
    docker compose stop redis mongo mlflow >/dev/null 2>&1 || true
  fi
  exit "$exit_code"
}

stop_all() {
  pkill -f "uvicorn app.main:app" 2>/dev/null || true
  pkill -f "next dev.*--port $FRONTEND_PORT" 2>/dev/null || true
  if command -v docker >/dev/null 2>&1; then
    docker compose stop redis mongo mlflow >/dev/null 2>&1 || true
  fi
  echo "Local processes stopped. Docker infrastructure stopped when available."
}

status_all() {
  echo "API:      $(curl -fsS "http://localhost:$API_PORT/healthz" 2>/dev/null || echo unavailable)"
  echo "Frontend: $(curl -fsSI "http://localhost:$FRONTEND_PORT" 2>/dev/null | head -n 1 || echo unavailable)"
  if command -v docker >/dev/null 2>&1; then
    docker compose ps redis mongo mlflow
  fi
}

run_simulation() {
  shift || true
  exec "$PYTHON_BIN" scripts/simulate_voip.py "$@"
}

main() {
  local command="${1:-start}"
  case "$command" in
    help|-h|--help) usage ;;
    stop) stop_all ;;
    status) status_all ;;
    simulate) run_simulation "$@" ;;
    start)
      load_environment
      trap cleanup EXIT INT TERM
      start_infrastructure
      start_backend
      start_frontend
      echo
      echo "Backend started. Log: $LOG_DIR/backend.log"
      echo "API:      http://localhost:$API_PORT/healthz"
      if [[ "$START_FRONTEND" == "1" ]]; then
        echo "Frontend: http://localhost:$FRONTEND_PORT"
      fi
      echo "Press Ctrl-C to stop the backend."
      while true; do sleep 2; done
      ;;
    *) echo "Unknown command: $command" >&2; usage >&2; exit 2 ;;
  esac
}

main "$@"
