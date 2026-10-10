#!/usr/bin/env bash
# Start the RX + NAI resource-grid visualizer.
# PRB and symbol counts are taken from the first ZeroMQ resource-grid message.
# Extra args are passed to docker compose, for example: ./start_resouce_grid_visualizer.sh -d

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_DIR="${ROOT}/compose"
ENV_FILE="${COMPOSE_DIR}/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  cp "${COMPOSE_DIR}/.env.example" "$ENV_FILE"
  echo "Created ${ENV_FILE} from .env.example"
  echo "Set MTX_WEBRTC_ADDITIONAL_HOSTS in that file to this machine's LAN IP before opening the UI from another computer."
fi

cd "$COMPOSE_DIR"
exec docker compose -f docker-compose.rxnai.yaml up --build "$@"
