## Plan: Contrail Detection Pipeline

### Files

```
live/
├── config.yaml
├── main.py          # async loop tying everything together
├── adsb.py          # poll adsb.lol, manage ping buffer
├── camera.py        # poll directory, download images, parse timestamps
├── processor.py     # FrameProcessor class: project pings onto image, run CV, label contrails
├── alerts.py        # TTL cache, fire alerts
├── analytics.py     # append to CSV, daily/monthly rollups
├── azure_upload.py  # upload annotated frames to Azure Blob Storage
```

### config.yaml

Camera location (AOSS building Madison WI), heading, FOV, tilt. ADS-B radius and poll interval. Alert TTL. Image poll interval. Directory base URL. Per-camera timezone (defaults to `America/Chicago`).

### main.py

Single `asyncio` loop with recurring tasks:

1. **ADS-B task** — runs every `adsb.poll_interval_s`, calls `adsb.poll()`, appends pings to a shared list, trims anything older than `adsb.max_age_s`.
2. **Image task (one per camera)** — runs every `poll_interval_s`, calls `camera.fetch_new_frames()`, then for each frame calls `proc.process_frame(ts, image_bytes, pings)`. Passes results to `azure_upload`, `alerts.check()`, and `analytics.log()`.

All tasks run concurrently via `asyncio.gather`. The ping list is shared across cameras (same physical site). Each camera has its own `FrameProcessor` instance.

### adsb.py

- `poll(lat, lon, radius)` → hits `https://api.adsb.lol/v2/lat/{lat}/lon/{lon}/dist/{dist}` → returns list of `Ping` dicts with hex, lat, lon, alt, heading, velocity, timestamp.
- `trim(pings, max_age_s)` → drops old pings in-place.

### camera.py

`Camera` class. State per instance:
- `_last_processed_filename` — last filename fetched (e.g. `"01_45_58.trig+00.jpg"`)
- `_last_processed_date` — local date string `"YYYY-MM-DD"` for rollover detection
- `timezone` — camera local timezone (from config, default `"America/Chicago"`)

`fetch_new_frames()`:
1. Get current time in camera-local timezone for directory path and date
2. Detect date rollover: if local date changed, reset `_last_processed_filename`
3. Build directory URL using local date (`base/{side}/img/YYYY/MM/DD/orig/`)
4. HTTP GET the directory listing, regex parse `HH_MM_SS.trig+00.jpg` filenames
5. Filter to filenames > `_last_processed_filename` (zero-padded string compare); on first call take only last 2 frames
6. Download concurrently (semaphore of 12)
7. Parse timestamp using `_parse_timestamp(filename, now_local)` — interprets filename time in camera timezone, converts to UTC-aware datetime
8. Return sorted list of `(datetime_utc, image_bytes)` tuples; update `_last_processed_filename`

### processor.py

`FrameProcessor` class (one instance per camera). State:
- `cam_params` — `(intrinsics, distortion, rvec, tvec, origin_gps)` from `proj_utils.load_camera_parameters()`
- `config` — loaded config dict
- `_prev_frame` — previous frame numpy array for Canny frame-differencing
- `_prev_frame_ts` — UTC-naive datetime of the previous frame

`process_frame(timestamp, image_bytes, pings)`:
1. Normalize `timestamp` to UTC-naive
2. Reset `_prev_frame` if it's older than 10 minutes
3. Decode JPEG bytes
4. Build ping DataFrame, filter by altitude and distance, upsample to 1-second intervals
5. Project all positions to camera pixels via `proj_utils.gps_to_camxy_vasha_fixed()`
6. For each aircraft, pick the ping closest in time to the frame (within 60s)
7. Run `detection_utils.get_directional_rectangle()` to build motion-aligned search boxes
8. Run `detection_utils.apply_canny_to_rectangles()` with frame differencing
9. Draw overlays (yellow rect = contrail, blue rect = no contrail; red dot + ident label)
10. Update `_prev_frame` / `_prev_frame_ts`
11. Return `(results, annotated_img)` where results is a list of `(timestamp, ident, px, py, alt_m, contrail_bool)`

Frames within a single camera are processed sequentially in timestamp order.

### alerts.py

`AlertCache` class with a dict `{ident: last_alert_time}`.

`check(results, annotated_img, cam, config, image_url)`: for each result where `contrail=True`, look up the identifier in the cache. If missing or expired (TTL from config), fire alert and update cache.

### analytics.py

`log(results, output_dir, camera_name, image_url)`: append each result row to a daily CSV.

### azure_upload.py

`upload_annotated_frame(annotated_img, ts, side, config)`: encodes frame as JPEG and uploads to Azure Blob Storage. Only called when at least one contrail is detected in the frame.
