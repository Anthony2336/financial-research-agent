#!/bin/sh
set -eu

if [ "${1:-}" = "db-upgrade" ]; then
  exec uv run --no-sync alembic upgrade head
fi

exec uv run --no-sync "$@"
