#!/usr/bin/env python3

import requests
from pathlib import Path
import re
from typing import List
from urllib.parse import urljoin
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
from tqdm import tqdm
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagesDownloader:
    def __init__(self,date: str):
        path = f"downloaded_images/east/{date}/"
        os.makedirs(path, exist_ok=True)
        self.images_dir = Path(path)
        # Create a session for connection pooling and reuse
        self.session = requests.Session()
        # Configure session with connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=30,
            pool_maxsize=30,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

    def get_image_list(self) -> List[str]:
        """Fetch the list of available images from the server."""
        try:
            response = self.session.get(self.base_url, timeout=1000)
            response.raise_for_status()

            # Extract image filenames using regex
            image_pattern = r'(\d{2}_\d{2}_\d{2}\.trig\+00\.jpg)'
            images = re.findall(image_pattern, response.text)

            # Sort images to ensure consistent ordering
            images.sort()
            logger.info(f"Found {len(images)} images")
            return images

        except Exception as e:
            logger.error(f"Error fetching image list: {e}")
            return []

    def _download_single_image(self, img_name: str) -> Path:
        """Download a single image using streaming for better memory efficiency."""
        img_url = urljoin(self.base_url, img_name)
        img_path = self.images_dir / img_name

        # Skip if already downloaded
        if img_path.exists():
            # logger.info(f"Skipping {img_name} (already exists)")
            return img_path

        try:
            # logger.info(f"Downloading {img_name}")
            # Use streaming to avoid loading entire image into memory
            response = self.session.get(img_url, timeout=30, stream=True)
            response.raise_for_status()

            # Write in chunks for better memory efficiency
            with open(img_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            return img_path

        except Exception as e:
            logger.error(f"Error downloading {img_name}: {e}")
            return None

    def download_images(self, base_url: str, max_workers: int = 30) -> List[Path]:
        """Download every 6th image from the server concurrently."""
        images = self.get_image_list()
        # Log the total number of images found
        logger.info(f"Fetched {len(images)} images from server")

        # Make the list unique by converting to a set and back to a list
        images = sorted(list(set(images)))
        logger.info(f"After removing duplicates: {len(images)} unique images")
        if not images:
            return []

        # Select every 6th image
        selected_images = images
        logger.info(f"Selected {len(selected_images)} images (every 6th)")
        downloaded_paths = []

        # Download images concurrently with progress bar
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_img = {executor.submit(self._download_single_image, img_name): img_name
                           for img_name in selected_images}

            with tqdm(total=len(selected_images), desc="Downloading images", unit="img") as pbar:
                for future in as_completed(future_to_img):
                    img_path = future.result()
                    if img_path is not None:
                        downloaded_paths.append(img_path)
                    pbar.update(1)

        return downloaded_paths

def main():
    import argparse
    from datetime import date, timedelta

    BASE_URL = "https://metobs.ssec.wisc.edu/pub/cache/aoss/cameras/east/img/"

    parser = argparse.ArgumentParser(description="Download UWisc camera images.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dates", nargs="+", metavar="YYYY-MM-DD",
                       help="One or more specific dates (e.g. 2025-03-13 2025-03-15)")
    group.add_argument("--start", metavar="YYYY-MM-DD",
                       help="Start date of a range (inclusive)")
    parser.add_argument("--end", metavar="YYYY-MM-DD",
                        help="End date of a range (inclusive, required with --start)")
    args = parser.parse_args()

    if args.start:
        if not args.end:
            parser.error("--end is required when --start is specified")
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
        dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    else:
        dates = [date.fromisoformat(d) for d in args.dates]

    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        date_path = f"{d.year}/{d.month:02d}/{d.day:02d}/orig/"
        processor = ImagesDownloader(date=date_str)
        processor.base_url = urljoin(BASE_URL, date_path)
        logger.info(f"Processing images for date: {date_str}")
        processor.download_images(processor.base_url)


if __name__ == "__main__":
    main()