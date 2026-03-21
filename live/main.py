"""
main.py — Live contrail detection pipeline.

Run from the project root:
    python -m live.main
    # or
    python -m live.main --config live/config.yaml
"""

import argparse
import asyncio
import logging
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Ensure the project root is on sys.path so utils.* imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live import adsb, analytics, azure_upload
from live.processor import FrameProcessor
from live.alerts import AlertCache
from live.camera import Camera
from live.config import load_config
import utils.projection_utils as proj_utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def main(config_path: str = "live/config.yaml") -> None:
    config = load_config(config_path)

    output_dir = config["analytics"]["output_dir"]
    alert_cache = AlertCache(ttl_s=config["alerts"]["ttl_s"], output_dir=output_dir)

    cam_configs = config["cameras"]
    cameras = [Camera(cam_cfg) for cam_cfg in cam_configs]
    processors = []
    for cam_cfg in cam_configs:
        cam_params = proj_utils.load_camera_parameters(cam_cfg["params_path"])
        processors.append(FrameProcessor(cam_params, config))
        logger.info(f"[main] loaded camera params from {cam_cfg['params_path']}")

    # Use the first camera's location for ADS-B filtering (all cameras share the same site)
    cam_lat = cam_configs[0]["lat"]
    cam_lon = cam_configs[0]["lon"]
    adsb_radius = config["adsb"]["radius_km"]
    adsb_interval = config["adsb"]["poll_interval_s"]
    adsb_max_age = config["adsb"]["max_age_s"]

    pings: list[dict] = []  # shared ADS-B buffer; single-threaded async — no locking needed

    async def adsb_task() -> None:
        """Poll ADS-B every adsb_interval seconds and maintain the ping buffer."""
        while True:
            new_pings = await adsb.poll(cam_lat, cam_lon, adsb_radius)
            pings.extend(new_pings)
            adsb.trim(pings, max_age_s=adsb_max_age)
            logger.info(f"[adsb] buffer: {len(pings)} pings")
            await asyncio.sleep(adsb_interval)

    async def image_task(cam: Camera, proc: FrameProcessor) -> None:
        """Fetch new frames every poll_interval_s seconds and run detection."""
        interval = cam.config["poll_interval_s"]
        while True:
            frames = await cam.fetch_new_frames()
            for ts, image_bytes in frames:
                results, annotated_img = proc.process_frame(ts, image_bytes, pings)
                image_url = None
                if any(r[5] for r in results) and annotated_img is not None:
                    try:
                        image_url = await asyncio.get_event_loop().run_in_executor(
                            None,
                            azure_upload.upload_annotated_frame,
                            annotated_img, ts, cam.side, config,
                        ) or ""
                    except Exception as e:
                        logger.warning(f"[main] failed to upload frame for {ts}: {e}")
                await alert_cache.check(results, annotated_img, cam, config, image_url=image_url)
                analytics.log(results, output_dir, camera_name=cam.name, image_url=image_url)
            await asyncio.sleep(interval)

    logger.info(f"[main] starting live contrail detection pipeline with {len(cameras)} camera(s)")
    await asyncio.gather(
        adsb_task(),
        *[image_task(cam, proc) for cam, proc in zip(cameras, processors)],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live contrail detection pipeline")
    parser.add_argument(
        "--config", default="live/config.yaml",
        help="Path to config.yaml (default: live/config.yaml)"
    )
    args = parser.parse_args()
    asyncio.run(main(args.config))
