#!/bin/sh
set -e
uvicorn live.server:app --host 0.0.0.0 --port 8000 &
exec python -m live.main --config live/config.yaml
