#!/usr/bin/env bash
# Stop the resource-grid visualizer started by ./rgv-deploy.sh.
# The image is kept.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_DIR="${ROOT}/compose"

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
  echo "Usage: ./rgv-undeploy.sh"
  echo "Stop the RX + NAI resource-grid visualizer. The image is kept."
  exit 0
fi

if docker compose version >/dev/null 2>&1; then
  dc() { docker compose -f docker-compose.rxnai.yaml "$@"; }
elif command -v docker-compose >/dev/null 2>&1; then
  dc() { docker-compose -f docker-compose.rxnai.yaml "$@"; }
else
  echo "docker compose is not installed" >&2
  exit 1
fi

cd "$COMPOSE_DIR"
echo "Stopping resource-grid visualizer"
dc down --remove-orphans
