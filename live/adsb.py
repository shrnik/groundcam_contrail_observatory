import time
import logging
from typing import TypedDict

import httpx

logger = logging.getLogger(__name__)


class _RawACRequired(TypedDict):
    lat: float
    lon: float


class RawAC(_RawACRequired, total=False):
    """Raw aircraft object from api.adsb.lol /v2 response."""
    hex: str
    type: str
    flight: str
    r: str              # registration
    alt_baro: int|str  # feet; may be "ground"
    alt_geom: int|str       # geometric altitude, feet maybe ground
    gs: float           # ground speed, knots
    track: float        # true track, degrees
    baro_rate: int      # ft/min
    squawk: str
    emergency: str
    category: str
    nav_qnh: float
    nav_altitude_mcp: int
    nav_modes: list[str]
    nic: int
    rc: int
    seen_pos: float     # seconds since last position update
    seen: float         # seconds since last message
    version: int
    nic_baro: int
    nac_p: int
    nac_v: int
    sil: int
    sil_type: str
    gva: int
    sda: int
    alert: int
    spi: int
    mlat: list
    tisb: list
    messages: int
    rssi: float
    dst: float          # distance from query point, nautical miles
    dir: float          # bearing from query point, degrees


class Ping(TypedDict):
    hex: str
    flight: str
    lat: float
    lon: float
    alt_baro: int | str
    alt_geom: int
    alt_gnss_meters: float
    gs: float
    track: float
    obs_time: float
    fetched_at: float


async def poll(lat: float, lon: float, radius_km: float) -> list[Ping]:
    """Poll api.adsb.lol for aircraft within radius_km of (lat, lon).

    Returns a list of ping dicts with keys:
        hex, flight, lat, lon, alt_baro, alt_geom, alt_gnss_meters,
        gs, track, obs_time (Unix float, UTC), fetched_at (Unix float).
    """
    radium_nautical_miles = radius_km * 0.539957
    url = f"https://api.adsb.lol/v2/lat/{lat}/lon/{lon}/dist/{int(radium_nautical_miles)}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"[adsb] poll failed: {e}")
        return []

    now = data.get("now", time.time() * 1000)  # data.now is in ms
    fetched_at = time.time()
    pings = []
    raw_aircraft: list[RawAC] = data.get("ac", [])
    for ac in raw_aircraft:
        if "lat" not in ac or "lon" not in ac:
            continue
        alt_geom = ac.get("alt_geom") or ac.get("alt_baro") or 0
        if isinstance(alt_geom, str):
            if alt_geom.lower() == "ground":
                alt_geom = 0
            else:
                try:
                    alt_geom = int(alt_geom)
                except ValueError:
                    alt_geom = 0
        ping = {
            "hex": ac.get("hex", ""),
            "flight": (ac.get("flight") or ac.get("hex", "")).strip(),
            "lat": float(ac["lat"]),
            "lon": float(ac["lon"]),
            "alt_baro": ac.get("alt_baro", 0),
            "alt_geom": alt_geom,
            "alt_gnss_meters": float(alt_geom) * 0.3048,
            "gs": ac.get("gs", 0) or 0,
            "track": ac.get("track", 0) or 0,
            # obs_time: when the position was last observed (server time minus staleness)
            "obs_time": now - ((ac.get("seen_pos") or ac.get("seen") or 0)*1000),
            "fetched_at": fetched_at,
        }
        # filter anything below 2500m
        if ping["alt_gnss_meters"] < 2500:
            continue
        pings.append(ping)

    logger.debug(f"[adsb] polled {len(pings)} aircraft")
    return pings


def trim(pings: list[Ping], max_age_s: float = 600) -> None:
    """Remove pings older than max_age_s from the shared list (in-place)."""
    cutoff = time.time() - max_age_s
    pings[:] = [p for p in pings if p["fetched_at"] >= cutoff]
