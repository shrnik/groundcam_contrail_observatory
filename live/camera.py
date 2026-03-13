import re
import asyncio
import logging
from datetime import datetime, timezone

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


class Camera:
    """Manages polling and downloading frames for a single camera."""

    def __init__(self, cam_config: dict) -> None:
        """
        cam_config should contain:
          lat, lon, alt, side, params_path, image_dir_base_url, poll_interval_s
        """
        self.config = cam_config
        self.side = cam_config["side"]
        self.timezone = cam_config.get("timezone", "America/Chicago")
        self._last_processed_filename: str | None = None

    def _build_dir_url(self, date: datetime) -> str:
        base = self.config["image_dir_base_url"].rstrip("/")
        return f"{base}/{self.side}/img/{date.year}/{date.month:02d}/{date.day:02d}/orig/"

    @staticmethod
    def _parse_timestamp(filename: str, date: datetime) -> datetime:
        time_part = filename.split(".")[0]  # e.g. "14_32_05"
        date_str = date.strftime("%Y-%m-%d")
        naive_dt = datetime.strptime(f"{date_str} {time_part}", "%Y-%m-%d %H_%M_%S")
        ts = pd.Timestamp(naive_dt, tz="America/Chicago").tz_convert("UTC")
        return ts.to_pydatetime()

    async def _list_directory(self, url: str) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
            pattern = r"(\d{2}_\d{2}_\d{2}\.trig\+00\.jpg)"
            filenames = re.findall(pattern, resp.text)
            return sorted(set(filenames))
        except Exception as e:
            logger.warning(f"[camera/{self.side}] directory listing failed for {url}: {e}")
            return []

    @staticmethod
    async def _download_one(
        client: httpx.AsyncClient, url: str, filename: str, sem: asyncio.Semaphore
    ) -> tuple[str, bytes] | None:
        async with sem:
            try:
                resp = await client.get(url + filename, timeout=30.0)
                resp.raise_for_status()
                return filename, resp.content
            except Exception as e:
                logger.warning(f"[camera] download failed for {filename}: {e}")
                return None

    async def fetch_new_frames(self) -> list[tuple[datetime, bytes]]:
        """Return new (timestamp_utc, image_bytes) tuples since last call.

        On the first call, returns only the two most-recent frames to avoid
        replaying a large backlog.
        """
        now_utc = datetime.now(timezone.utc)
        # convert to camera local date for directory path; timestamps in filenames are in local time
        now_local = now_utc.astimezone(timezone(self.timezone))
        url = self._build_dir_url(now_local)
        filenames = await self._list_directory(url)

        # Midnight rollover: also check yesterday if it's very early UTC
        if (
            now_local.hour == 0
            and self._last_processed_filename
            and self._last_processed_filename.startswith("23")
        ):
            from datetime import timedelta

            yesterday = now_local - timedelta(days=1)
            yesterday_url = self._build_dir_url(yesterday)
            yesterday_files = await self._list_directory(yesterday_url)
            straggler_files = [f for f in yesterday_files if f > self._last_processed_filename]
            if straggler_files:
                filenames = straggler_files + filenames

        if not filenames:
            return []

        if self._last_processed_filename:
            new_filenames = [f for f in filenames if f > self._last_processed_filename]
        else:
            # First call: only process the latest 2 frames to avoid a large backlog
            new_filenames = filenames[-2:]

        if not new_filenames:
            return []

        sem = asyncio.Semaphore(12)
        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [self._download_one(client, url, fn, sem) for fn in new_filenames]
            results = await asyncio.gather(*tasks)

        frames: list[tuple[datetime, bytes]] = []
        for result in results:
            if result is None:
                continue
            filename, image_bytes = result
            try:
                ts = self._parse_timestamp(filename, now_utc)
                frames.append((ts, image_bytes))
            except Exception as e:
                logger.warning(f"[camera/{self.side}] failed to parse timestamp for {filename}: {e}")

        frames.sort(key=lambda x: x[0])

        if new_filenames:
            self._last_processed_filename = sorted(new_filenames)[-1]

        logger.info(f"[camera/{self.side}] fetched {len(frames)} new frame(s)")
        return frames
