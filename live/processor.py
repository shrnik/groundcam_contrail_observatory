"""
processor.py — Project ADS-B pings onto camera pixels and run contrail detection.
"""

import logging
from datetime import datetime, timezone

import cv2
import numpy as np
import pandas as pd

import utils.adsb_utils as adsb_utils
import utils.projection_utils as proj_utils
import utils.detection_utils as detection_utils
from live.adsb import Ping

logger = logging.getLogger(__name__)

# Cache the previous frame's numpy array for frame-differencing in Canny detection
_prev_frame: np.ndarray | None = None


def _pings_to_dataframe(pings: list[Ping], origin_gps: list) -> pd.DataFrame:
    """Convert the raw ping buffer to a DataFrame suitable for processing."""
    if not pings:
        return pd.DataFrame()

    rows = []
    for p in pings:
        rows.append({
            "ident": p["flight"] or p["hex"],
            "transponder_id": p["hex"],
            "lat": p["lat"],
            "lon": p["lon"],
            "alt_gnss_meters": p["alt_gnss_meters"],
            "heading": p["track"],
            "distance_m": adsb_utils.haversine_km(
                p["lat"], p["lon"], origin_gps[0], origin_gps[1]
            ) * 1000,
            # Convert Unix obs_time to UTC-aware Timestamp
            "time": pd.Timestamp(round(p["obs_time"]), unit="ms").tz_localize("UTC"),
        })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["lat", "lon", "alt_gnss_meters"])
    return df


def _build_upsampled(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Filter by altitude/distance and upsample each aircraft to 1-second intervals."""
    min_alt = config["adsb"]["min_alt_m"]
    max_range_m = config["detection"]["max_range_km"] * 1000

    df = df[df["alt_gnss_meters"] > min_alt]
    df = df[df["distance_m"] < max_range_m]

    if df.empty:
        return df

    groups = []
    for _, group in df.groupby("ident"):
        upsampled = adsb_utils.upsample_aircraft(group)
        groups.append(upsampled)

    if not groups:
        return pd.DataFrame()

    result = pd.concat(groups, ignore_index=True)
    return result.sort_values(["ident", "time"]).reset_index(drop=True)


def process_frame(
    timestamp: datetime,
    image_bytes: bytes,
    pings: list[dict],
    cam_params: tuple,
    config: dict,
) -> list[tuple]:
    """
    Process one camera frame against the current ADS-B ping buffer.

    Args:
        timestamp:    UTC datetime of the image.
        image_bytes:  Raw JPEG bytes from the camera.
        pings:        Shared ping buffer (list of dicts from adsb.poll).
        cam_params:   Tuple (intrinsics, distortion, rvec, tvec, origin_gps)
                      from proj_utils.load_camera_parameters().
        config:       Loaded config dict.

    Returns:
        List of (timestamp, transponder_id, px, py, alt_m, contrail_bool) tuples.
    """
    global _prev_frame

    intrinsics, distortion, rvec, tvec, origin_gps = cam_params

    # --- Decode image ---
    img_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img_np is None:
        logger.warning("[processor] failed to decode image bytes")
        return [], None

    # Use previous frame for differencing; fall back to current if none yet
    prev_np = _prev_frame if _prev_frame is not None else img_np

    # --- Build ping DataFrame ---
    df = _pings_to_dataframe(pings, origin_gps)
    if df.empty:
        logger.info("[processor] no valid pings in buffer")
        _prev_frame = img_np
        return [], img_np

    # --- Upsample to 1-second intervals (for dense motion history) ---
    df_upsampled = _build_upsampled(df, config)
    if df_upsampled.empty:
        logger.info("[processor] no pings pass altitude/distance filters")
        _prev_frame = img_np
        return [], img_np

    # --- Project all upsampled positions to camera pixels ---
    image_x, image_y, _ = proj_utils.gps_to_camxy_vasha_fixed(
        df_upsampled["lat"].values,
        df_upsampled["lon"].values,
        df_upsampled["alt_gnss_meters"].values,
        cam_k=intrinsics,
        cam_r=rvec,
        cam_t=tvec,
        camera_gps=origin_gps,
        distortion=distortion,
    )
    df_upsampled["image_x"] = image_x
    df_upsampled["image_y"] = image_y

    # --- df_filtered: best position estimate for each aircraft at frame time ---
    ts_pd = pd.Timestamp(timestamp).tz_convert("UTC") if timestamp.tzinfo else pd.Timestamp(timestamp, tz="UTC")
    df_upsampled["_time_diff"] = abs(df_upsampled["time"] - ts_pd)
    # For each aircraft, pick the row closest in time to the frame timestamp
    idx = df_upsampled.groupby("ident")["_time_diff"].idxmin()
    df_filtered = df_upsampled.loc[idx].copy()
    # Only keep aircraft whose closest ping is within 60s of the frame
    df_filtered = df_filtered[df_filtered["_time_diff"] <= pd.Timedelta("60s")]
    df_filtered = df_filtered.drop(columns=["_time_diff"])
    df_upsampled = df_upsampled.drop(columns=["_time_diff"])

    # Drop aircraft with no valid pixel projection
    df_filtered = df_filtered.dropna(subset=["image_x", "image_y"])

    if df_filtered.empty:
        logger.info("[processor] no aircraft visible in frame")
        _prev_frame = img_np
        return [], img_np

    logger.info(f"[processor] {len(df_filtered)} aircraft visible in frame")

    # --- Run contrail detection ---
    rectangles = detection_utils.get_directional_rectangle(
        img_np, df_filtered, ts_pd, df_upsampled,
        length_px=200, width_px=100,
    )

    if not rectangles:
        _prev_frame = img_np
        return [], img_np

    _, edge_data, _ = detection_utils.apply_canny_to_rectangles(
        img_np, prev_np, rectangles,
        blur_kernel=(3, 3),
        min_line_length=config["detection"]["min_line_length"],
    )

    _prev_frame = img_np

    # --- Draw overlays ---
    annotated = img_np.copy()
    for ident, (rect_poly, _, _dir) in rectangles.items():
        is_contrail = edge_data.get(ident, {}).get("is_making_contrails", False)
        color = (0, 255, 255) if is_contrail else (255, 0, 0)  # yellow or blue
        cv2.polylines(annotated, [rect_poly], isClosed=True, color=color, thickness=2)
    for _, row in df_filtered.iterrows():
        px_f, py_f = row["image_x"], row["image_y"]
        if np.isnan(px_f) or np.isnan(py_f):
            continue
        px_i, py_i = int(px_f), int(py_f)
        if 0 <= px_i < annotated.shape[1] and 0 <= py_i < annotated.shape[0]:
            cv2.circle(annotated, (px_i, py_i), 6, (0, 0, 255), -1)
            cv2.putText(annotated, str(row["ident"]), (px_i + 8, py_i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- Assemble results ---
    results = []
    for ident, data in edge_data.items():
        row = df_filtered[df_filtered["ident"] == ident]
        if row.empty:
            continue
        row = row.iloc[0]
        results.append((
            timestamp,
            row.get("transponder_id", ident),
            float(row["image_x"]),
            float(row["image_y"]),
            float(row["alt_gnss_meters"]),
            bool(data["is_making_contrails"]),
        ))

    n_contrails = sum(1 for r in results if r[5])
    logger.info(f"[processor] {n_contrails}/{len(results)} aircraft making contrails")
    return results, annotated
