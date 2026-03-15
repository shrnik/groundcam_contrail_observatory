#!/bin/sh
# Local dev only — in production each container runs its own command.
set -e
uvicorn live.server:app --host 0.0.0.0 --port 8000 &
exec python -m live.main --config live/config.yaml
