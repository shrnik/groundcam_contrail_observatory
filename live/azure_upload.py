"""
azure_upload.py — Upload annotated contrail frames to Azure Blob Storage.

Requires the environment variable AZURE_STORAGE_CONNECTION_STRING, or
set `azure.connection_string` in config.yaml.

Blob path: <container>/<side>/<YYYY-MM-DD>/<HH_MM_SS>.jpg
"""

import io
import logging
import os
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas

logger = logging.getLogger(__name__)


def _get_connection_string(config: dict) -> str | None:
    return (
        config.get("azure", {}).get("connection_string")
        or os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    )


def upload_annotated_frame(
    annotated_img: np.ndarray,
    ts: datetime,
    side: str,
    config: dict,
) -> str | None:
    """Encode annotated_img as JPEG and upload to Azure Blob Storage.

    Returns the blob URL on success, None on failure.
    """
    conn_str = _get_connection_string(config)
    if not conn_str:
        logger.warning("[azure] no connection string configured — skipping upload")
        return None

    container = config.get("azure", {}).get("container", "contrail-frames")

    blob_name = f"{side}/{ts.strftime('%Y-%m-%d')}/{ts.strftime('%H_%M_%S')}.jpg"

    ok, buf = cv2.imencode(".jpg", annotated_img)
    if not ok:
        logger.error("[azure] failed to encode frame as JPEG")
        return None

    try:
        service_client = BlobServiceClient.from_connection_string(conn_str)
        blob_client = service_client.get_blob_client(container=container, blob=blob_name)
        blob_client.upload_blob(io.BytesIO(buf.tobytes()), overwrite=True)

        # Generate a SAS URL valid for 7 days so Slack can render it
        sas_token = generate_blob_sas(
            account_name=service_client.account_name,
            container_name=container,
            blob_name=blob_name,
            account_key=service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(days=30),
        )
        sas_url = f"{blob_client.url}?{sas_token}"
        logger.info(f"[azure] uploaded {blob_name} → {sas_url}")
        return sas_url
    except Exception as e:
        logger.error(f"[azure] upload failed: {e}")
        return None
