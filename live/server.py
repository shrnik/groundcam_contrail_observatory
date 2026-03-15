"""
server.py — FastAPI JSON API server for contrail detection data.

Run from the project root:
    uvicorn live.server:app --reload
    OUTPUT_DIR=live_output uvicorn live.server:app --port 8000
"""

import csv
import glob
import os
from datetime import date

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "live_output")

app = FastAPI(title="Contrail Observatory")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _read_day(date_str: str) -> list[dict]:
    path = os.path.join(OUTPUT_DIR, f"{date_str}.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _group_frames(rows: list[dict]) -> list[dict]:
    """Group CSV rows by (timestamp, camera_name) → list of frame dicts."""
    groups: dict[tuple, dict] = {}
    for row in rows:
        ts = row["timestamp"]
        side = row.get("camera_name", "")
        key = (ts, side)
        if key not in groups:
            groups[key] = {
                "timestamp": ts,
                "camera_name": side,
                "image_url": row.get("image_url", ""),
                "aircraft": [],
            }
        groups[key]["aircraft"].append(row)
    return sorted(groups.values(), key=lambda x: x["timestamp"])


def _available_dates() -> list[str]:
    paths = sorted(glob.glob(os.path.join(OUTPUT_DIR, "????-??-??.csv")))
    return [os.path.basename(p).removesuffix(".csv") for p in paths]


def _parse_date(date_str: str) -> date:
    try:
        return date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date '{date_str}' — use YYYY-MM-DD")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/dates")
def api_dates():
    return {"dates": _available_dates()}


@app.get("/api/{date_str}")
def api_day(date_str: str):
    _parse_date(date_str)
    rows = _read_day(date_str)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No data found for {date_str}")
    frames = _group_frames(rows)
    total = len({r["ident"] for r in rows})
    contrail = len({r["ident"] for r in rows if r.get("contrail") == "1"})
    return {"date": date_str, "total_aircraft": total, "contrail_aircraft": contrail, "frames": frames}
