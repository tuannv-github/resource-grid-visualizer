#!/usr/bin/env bash
# Build and start the RX + NAI resource-grid visualizer.
# UI http://127.0.0.1:5173, RX API :8082, NAI API :8083.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_DIR="${ROOT}/compose"
ENV_FILE="${COMPOSE_DIR}/.env"

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
  echo "Usage: ./rgv-deploy.sh"
  echo "Build and start the RX + NAI resource-grid visualizer."
  echo "UI http://127.0.0.1:5173  RX :8082  NAI :8083"
  exit 0
fi

if docker compose version >/dev/null 2>&1; then
  dc() { docker compose -p rgv -f docker-compose.rxnai.yaml "$@"; }
elif command -v docker-compose >/dev/null 2>&1; then
  dc() { docker-compose -p rgv -f docker-compose.rxnai.yaml "$@"; }
else
  echo "docker compose is not installed" >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  cp "${COMPOSE_DIR}/.env.example" "$ENV_FILE"
  echo "Created ${ENV_FILE} from .env.example"
fi

cd "$COMPOSE_DIR"
echo "Building and starting resource-grid visualizer"
dc up -d --build --remove-orphans

ready=0
for _ in $(seq 1 120); do
  if python3 - <<'PY'
import urllib.request
def ok(url):
    try:
        urllib.request.urlopen(url, timeout=2).read()
        return True
    except Exception:
        return False
raise SystemExit(0 if ok("http://127.0.0.1:8082/health") and ok("http://127.0.0.1:8083/health") and ok("http://127.0.0.1:5173/") else 1)
PY
  then
    ready=1
    break
  fi
  sleep 1
done

if [[ "$ready" -ne 1 ]]; then
  echo "resource-grid visualizer did not become ready on :5173 / :8082 / :8083" >&2
  dc logs --tail 80 >&2 || true
  exit 1
fi

echo "UI:  http://127.0.0.1:5173"
echo "RX:  http://127.0.0.1:8082"
echo "NAI: http://127.0.0.1:8083"
