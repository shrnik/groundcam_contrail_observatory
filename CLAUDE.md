# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ground-based camera system for detecting aircraft contrails. It correlates real-time ADS-B aircraft tracking data with camera images to identify when and where contrails form. The system projects GPS coordinates onto camera pixels using calibrated camera parameters, then applies computer vision to detect contrail edges in images.

**Primary camera site**: UW-Madison AOSS Building, Madison WI (43.0706°N, -89.4069°W)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Pipeline

The main batch pipeline is [non_live/contrail_pipeline_uwisc.py](non_live/contrail_pipeline_uwisc.py). Edit the `main()` function to set the date/time window, then run:

```bash
python non_live/contrail_pipeline_uwisc.py
```

Expected inputs:
- ADS-B CSV: `/Users/shrenikborad/pless/easy_adsb/data/madison_pings_YYYY_MM_DD.csv`
- Camera images: `/Users/shrenikborad/Downloads/NNDL/images_uwisc/east/YYYY-MM-DD/{east,south}/`
- Camera params: `calibration_data/uwisc/{east,south}/camera_params.json`

To download images from the UW-Madison server:
```bash
python utils/uwisc_downloader.py --dates 2025-03-13 2025-03-15
python utils/uwisc_downloader.py --start 2025-03-01 --end 2025-03-10
```

## Architecture

### Data Flow

```
ADS-B API (api.adsb.lol) → adsb_utils.py (poll/filter/upsample to 1-sec intervals)
        ↓
GPS positions (lat, lon, alt, timestamp) per aircraft
        ↓
projection_utils.py (GPS → camera pixel via calibration)
        ↓
Camera images (UW-Madison server, polled or batch)
        ↓
detection_utils.py (Canny edges + Hough lines in aircraft region)
        ↓
Contrail classification → video output + CSV + DuckDB
```

### Key Modules (`utils/`)

- **[adsb_utils.py](utils/adsb_utils.py)** — ADS-B data: polling `api.adsb.lol`, filtering by altitude (>8000 ft) and distance (<80 km), temporal upsampling to 1-second intervals via interpolation
- **[projection_utils.py](utils/projection_utils.py)** — Core geometry: converts GPS (lat/lon/alt) → camera pixel using ECEF→ENU→camera-frame transform with OpenCV distortion model. `load_camera_parameters()` reads `camera_params.json`; `gps_to_camxy_vasha_fixed()` is the main projection function
- **[detection_utils.py](utils/detection_utils.py)** — Contrail detection: draws directional rectangles along aircraft motion vectors (based on 10-second history), runs Canny edge detection, applies Hough transform, classifies as contrail if ≥2 lines ≥40 px aligned within 8° of motion direction
- **[image_data_utils.py](utils/image_data_utils.py)** — Filename parsing for 3 camera sources (UW-Madison `HH_MM_SS.trig+00.jpg`, Arizona `YYYYMMDDHHMMSS.jpg`, MIT `frame_YYYYMMDD_HHMMSS.jpg`)
- **[db_utils.py](utils/db_utils.py)** — DuckDB storage via `ContrailDatabase` class; stores edge detection results with metadata; can export to GeoJSON
- **[uwisc_downloader.py](utils/uwisc_downloader.py)** — Concurrent HTTP download from UW-Madison SSEC server with session pooling and retry logic

### Camera Calibration

Two cameras calibrated at the same GPS origin (AOSS building):
- **East camera**: focal ~1365 px, barrel distortion (K1=-0.361), image 2592×1944
- **South camera**: focal ~1237 px, pincushion distortion (K1=0.140), image 2592×1944

Calibration was performed by matching sun/moon positions in images to computed astronomical positions. Calibration data (matches, reference images) lives in `calibration_data/uwisc/{east,south}/`.

`camera_params.json` contains: `K` (intrinsics), `dist_coeffs`, `R` (rotation world→camera), `T` (translation), `origin_gps`.

### Detection Parameters

| Parameter | Value |
|-----------|-------|
| Altitude threshold | >8000 ft (2438 m) |
| Distance threshold | <80 km from camera |
| Rectangle width | 70–100 px |
| Rectangle length | 50–500 px (scales with speed) |
| Min Hough line length | 40 px |
| Min aligned lines for detection | 2 |
| Angle tolerance | 8° |

## External Services

- **ADS-B data**: `api.adsb.lol` (no auth required)
- **Camera images**: UW-Madison SSEC server (`metobs.ssec.wisc.edu`)
