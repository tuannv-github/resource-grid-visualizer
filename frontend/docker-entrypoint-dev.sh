#!/bin/sh
set -e
cd /app
STAMP_FILE=node_modules/.npm_lock_stamp
LOCK_STAMP=$(md5sum package-lock.json | cut -d" " -f1)
if [ ! -f "$STAMP_FILE" ] || [ "$(cat "$STAMP_FILE")" != "$LOCK_STAMP" ]; then
  npm ci
  echo "$LOCK_STAMP" > "$STAMP_FILE"
fi

exec npm run dev -- --host 0.0.0.0 --port 5173
