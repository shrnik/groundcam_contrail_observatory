import logging
import os
import time
from datetime import datetime
import asyncio
import cv2
import httpx
import numpy as np
from live import azure_upload
from live.camera import Camera

logger = logging.getLogger(__name__)

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip('"')
print(f"[alerts] SLACK_WEBHOOK_URL set: {bool(SLACK_WEBHOOK_URL)}")
class AlertCache:
    """TTL cache for contrail alerts — fires at most once per aircraft per TTL window,
    but groups all contrailing aircraft in a frame into a single alert message."""

    def __init__(self, ttl_s: float = 300, output_dir: str = "live_output"):
        self.ttl_s = ttl_s
        self.output_dir = output_dir
        self._last_alert: dict[str, float] = {}  # ident → last alert time

    async def check(self, results: list[tuple], annotated_img: np.ndarray | None, cam: Camera, config) -> None:
        """Fire a single per-frame alert listing all contrailing aircraft not within TTL."""
        now = time.time()

        # Collect all contrailing aircraft in this frame
        contrail_aircraft = [
            (timestamp, ident, px, py, alt_m)
            for timestamp, ident, px, py, alt_m, contrail in results
            if contrail
        ]

        if not contrail_aircraft:
            return

        # Only fire if at least one aircraft is outside the TTL window
        new_aircraft = [
            entry for entry in contrail_aircraft
            if now - self._last_alert.get(entry[1], 0) > self.ttl_s
        ]

        if new_aircraft:
            frame_ts = contrail_aircraft[0][0]
            await self._fire(frame_ts, contrail_aircraft, annotated_img, cam, config)
            for _, ident, *_ in contrail_aircraft:
                self._last_alert[ident] = now

        # Lazy eviction of expired entries
        self._last_alert = {
            k: v for k, v in self._last_alert.items() if now - v <= self.ttl_s * 2
        }

    async def _fire(
        self,
        timestamp: datetime,
        aircraft: list[tuple],
        annotated_img: np.ndarray | None,
        cam: Camera,
        config,
    ) -> None:
        time_str = timestamp.strftime("%H:%M:%S UTC")

        # --- Save annotated image to disk ---
        img_path = None
        image_url = None
        if annotated_img is not None:
            # alerts_dir = os.path.join(self.output_dir, "alerts")
            # os.makedirs(alerts_dir, exist_ok=True)
            # fname = f"{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
            # img_path = os.path.join(alerts_dir, fname)
            # cv2.imwrite(img_path, annotated_img)
            try:
                image_url = await asyncio.get_event_loop().run_in_executor(
                            None,
                            azure_upload.upload_annotated_frame,
                            annotated_img, timestamp, cam.side, config,
                        )
            except Exception as e:
                logger.warning(f"[alerts] failed to upload annotated image: {e}")
            logger.info(f"[alerts] saved annotated image to {img_path}")

        # --- Log to console ---
        ids = ", ".join(t[1] for t in aircraft)
        print(f"[ALERT] Contrails at {time_str} — {len(aircraft)} aircraft: {ids}")
        for _, ident, px, py, alt_m in aircraft:
            alt_ft = int(alt_m * 3.28084)
            logger.info(
                f"[ALERT] Contrail — flight={ident}  "
                f"alt={alt_m:.0f}m ({alt_ft:,}ft)  px=({px:.0f},{py:.0f})  {time_str}"
            )

        # --- Build Slack message with all aircraft ---
        aircraft_lines = "\n".join(
            f"  • `{tid}`  {alt_m:,.0f} m ({int(alt_m * 3.28084):,} ft)  px=({px:.0f},{py:.0f})"
            for _, tid, px, py, alt_m in aircraft
        )
        slack_text = (
            f":airplane_arriving: *Contrail detected!*  *Time:* {time_str}\n"
            f"*{len(aircraft)} aircraft:*\n{aircraft_lines}"
        )

        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": slack_text}}]
        if image_url:
            blocks.append({
                "type": "image",
                "image_url": image_url,
                "alt_text": f"Contrails detected — {len(aircraft)} aircraft",
            })
        payload = {"blocks": blocks}

        if not SLACK_WEBHOOK_URL:
            logger.warning("[alerts] SLACK_WEBHOOK_URL not set, skipping Slack notification")
            return
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(SLACK_WEBHOOK_URL, json=payload)
                resp.raise_for_status()
            logger.info(f"[alerts] Slack notification sent for frame {time_str} ({len(aircraft)} aircraft)")
        except Exception as e:
            logger.warning(f"[alerts] Slack notification failed: {e}")
