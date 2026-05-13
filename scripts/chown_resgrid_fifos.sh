#!/usr/bin/env bash
# Chown every named pipe in the host tmp dir (same tree mounted as /tmp in compose).
# Default owner matches docker-compose / nws: "${UID:-1000}:${GID:-1000}".
set -euo pipefail

owner="${UID:-1000}:${GID:-1000}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dir="${1:-$here/host-tmp}"

if [[ ! -d "$dir" ]]; then
  echo "Directory not found: $dir" >&2
  exit 1
fi

mapfile -d '' fifos < <(find "$dir" -maxdepth 1 -type p -print0 2>/dev/null || true)
if ((${#fifos[@]} == 0)); then
  echo "No FIFOs (-type p) in $dir"
  exit 0
fi

sudo chown "$owner" "${fifos[@]}"
echo "chown $owner: ${#fifos[@]} FIFO(s) in $dir"
