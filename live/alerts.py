import logging
import os
import time
from datetime import datetime

import cv2
import httpx
import numpy as np

logger = logging.getLogger(__name__)

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip('"')
print(f"[alerts] SLACK_WEBHOOK_URL set: {bool(SLACK_WEBHOOK_URL)}")
class AlertCache:
    """TTL cache for contrail alerts — fires once per aircraft per TTL window."""

    def __init__(self, ttl_s: float = 300, output_dir: str = "live_output"):
        self.ttl_s = ttl_s
        self.output_dir = output_dir
        self._last_alert: dict[str, float] = {}  # transponder_id → last alert time

    async def check(self, results: list[tuple], annotated_img: np.ndarray | None, image_url: str | None = None) -> None:
        """Fire alerts for any contrail result that isn't within the TTL."""
        now = time.time()
        for timestamp, transponder_id, px, py, alt_m, contrail in results:
            if not contrail:
                continue
            last = self._last_alert.get(transponder_id, 0)
            if now - last > self.ttl_s:
                await self._fire(timestamp, transponder_id, px, py, alt_m, annotated_img, image_url)
                self._last_alert[transponder_id] = now

        # Lazy eviction of expired entries
        self._last_alert = {
            k: v for k, v in self._last_alert.items() if now - v <= self.ttl_s * 2
        }

    async def _fire(
        self,
        timestamp: datetime,
        transponder_id: str,
        px: float,
        py: float,
        alt_m: float,
        annotated_img: np.ndarray | None,
        image_url: str | None = None,
    ) -> None:
        time_str = timestamp.strftime("%H:%M:%S UTC")
        alt_ft = int(alt_m * 3.28084)

        # --- Save annotated image to disk ---
        img_path = None
        if annotated_img is not None:
            alerts_dir = os.path.join(self.output_dir, "alerts")
            os.makedirs(alerts_dir, exist_ok=True)
            fname = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{transponder_id}.jpg"
            img_path = os.path.join(alerts_dir, fname)
            cv2.imwrite(img_path, annotated_img)
            # image_url = await asyncio.get_event_loop().run_in_executor(
                    #     None,
                    #     azure_upload.upload_annotated_frame,
                    #     annotated_img, ts, cam.side, config,
                    # )
            logger.info(f"[alerts] saved annotated image to {img_path}")

        # --- Log to console ---
        msg_text = (
            f"[ALERT] Contrail — flight={transponder_id}  "
            f"alt={alt_m:.0f}m ({alt_ft:,}ft)  px=({px:.0f},{py:.0f})  {time_str}"
        )
        print(msg_text)
        logger.info(msg_text)

        # --- Send to Slack ---
        slack_text = (
            f":airplane_arriving: *Contrail detected!*\n"
            f"*Flight:* `{transponder_id}`   "
            f"*Altitude:* {alt_m:,.0f} m ({alt_ft:,} ft)   "
            f"*Time:* {time_str}"
        )
        if img_path:
            slack_text += f"\n*Image saved:* `{img_path}`"

        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": slack_text}}]
        if image_url:
            blocks.append({
                "type": "image",
                "image_url": image_url,
                "alt_text": f"Contrail detected — {transponder_id}",
            })
        payload = {"blocks": blocks}

        if not SLACK_WEBHOOK_URL:
            logger.warning("[alerts] SLACK_WEBHOOK_URL not set, skipping Slack notification")
            return
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(SLACK_WEBHOOK_URL, json=payload)
                resp.raise_for_status()
            logger.info(f"[alerts] Slack notification sent for {transponder_id}")
        except Exception as e:
            logger.warning(f"[alerts] Slack notification failed: {e}")
