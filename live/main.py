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

import cv2
from live import adsb, analytics, azure_upload, processor
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
    cam_params_list = [
        proj_utils.load_camera_parameters(cam_cfg["params_path"]) for cam_cfg in cam_configs
    ]
    for cam_cfg in cam_configs:
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

    async def image_task(cam: Camera, cam_params: dict) -> None:
        """Fetch new frames every poll_interval_s seconds and run detection."""
        interval = cam.config["poll_interval_s"]
        while True:
            frames = await cam.fetch_new_frames()
            for ts, image_bytes in frames:
                results, annotated_img = processor.process_frame(
                    ts, image_bytes, pings, cam_params, config
                )
                image_url = None
                if annotated_img is not None:
                    frame_dir = os.path.join(output_dir, "frames", cam.side)
                    os.makedirs(frame_dir, exist_ok=True)
                    frame_path = os.path.join(frame_dir, ts.strftime("%Y%m%d_%H%M%S") + ".jpg")
                    cv2.imwrite(frame_path, annotated_img)
                    # image_url = await asyncio.get_event_loop().run_in_executor(
                    #     None,
                    #     azure_upload.upload_annotated_frame,
                    #     annotated_img, ts, cam.side, config,
                    # )
                await alert_cache.check(results, annotated_img, image_url)
                analytics.log(results, output_dir)
            await asyncio.sleep(interval)

    logger.info(f"[main] starting live contrail detection pipeline with {len(cameras)} camera(s)")
    await asyncio.gather(
        adsb_task(),
        *[image_task(cam, cam_params) for cam, cam_params in zip(cameras, cam_params_list)],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live contrail detection pipeline")
    parser.add_argument(
        "--config", default="live/config.yaml",
        help="Path to config.yaml (default: live/config.yaml)"
    )
    args = parser.parse_args()
    asyncio.run(main(args.config))
