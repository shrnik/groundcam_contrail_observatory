## Plan: Contrail Detection Pipeline

### Files

```
contrail_pipeline/
├── config.yaml
├── main.py          # async loop tying everything together
├── adsb.py          # poll adsb.lol, manage ping buffer
├── camera.py        # poll directory, download images, parse timestamps
├── processor.py     # project pings onto image, run CV, label contrails
├── alerts.py        # TTL cache, fire alerts
├── analytics.py     # append to CSV, daily/monthly rollups
```

### config.yaml

Camera location (AOSS building Madison WI), heading, FOV, tilt. ADS-B radius and poll interval. Alert TTL. Image poll interval. Directory base URL.

### main.py

Single `asyncio` loop with two recurring tasks:

1. **ADS-B task** — runs every 10s, calls `adsb.poll()`, appends pings to a shared list, trims anything older than 5 min.
2. **Image task** — runs every 120s, calls `camera.fetch_new_frames()`, then for each frame calls `processor.process_frame(frame, pings)`, which returns labeled results. Pass results to `alerts.check()` and `analytics.log()`.

Both tasks share the ping list (just a plain list, single-threaded async so no locking needed).

### adsb.py

One function: `poll(lat, lon, radius)` → hits `https://api.adsb.lol/v2/lat/{lat}/lon/{lon}/dist/{dist}` → returns list of dicts with hex, lat, lon, alt, heading, velocity, timestamp. One function: `trim(pings, max_age_s=300)` → drops old pings in-place.

### camera.py

Maintains `last_processed_filename` (string, e.g. `"01_45_58.trig+00.jpg"`).

`fetch_new_frames()`:
1. Build today's URL from UTC date
2. HTTP GET the directory listing
3. Regex parse all filenames from the HTML
4. Filter to filenames > `last_processed_filename` (string comparison works since they're zero-padded HH_MM_SS)
5. Download each new image (async, could do 12 concurrent)
6. Return list of `(datetime_utc, image_bytes)` tuples
7. Update `last_processed_filename`

Handle midnight rollover: if current UTC hour is 00 and last processed was hour 23, also check yesterday's directory for stragglers.

### processor.py

`process_frame(timestamp, image_bytes, pings)`:
1. Filter pings to those within ±5s of the image timestamp
2. For each ping, call `project_to_pixel(ping_lat, ping_lon, ping_alt, camera_config)` → returns `(px, py)` or `None` if outside FOV
3. For each visible aircraft, call `detect_contrail(image, px, py)` → returns `bool`
4. Return list of `(timestamp, transponder_id, px, py, alt, contrail_bool)`

**Projection stub:** takes aircraft lat/lon/alt, computes azimuth and elevation angle relative to camera, maps to pixel coords using FOV and image dimensions. You'll refine this with your calibration.

**Contrail detection stub:** your CV pipeline goes here. Input: image + pixel coordinate. Output: bool.

### alerts.py

`AlertCache` class with a dict `{transponder_id: last_alert_time}`.

`check(results)`: for each result where `contrail=True`, look up the transponder in the cache. If missing or expired (>5 min), fire alert and update cache. Lazy eviction — no background cleanup needed.

Alert action: start with just `print()` / log to file. Easy to swap in webhook/email later.

### analytics.py

`log(results)`: append each result row to a daily CSV file (`YYYY-MM-DD.csv`) with columns: `timestamp, transponder_id, px, py, alt, contrail`.

`daily_summary(date)`: read CSV, count unique transponders with at least one `contrail=True` vs total unique transponders.

`monthly_summary(year, month)`: glob all CSVs for that month, aggregate daily summaries.

---

### Execution order to build

1. `config.yaml` + config loader — so everything else can reference it
2. `camera.py` — get images flowing first, easiest to test standalone
3. `adsb.py` — get pings flowing, test standalone
4. `processor.py` — wire them together with stubs
5. `alerts.py` — simple, build fast
6. `analytics.py` — simple, build fast
7. `main.py` — tie it all together

Want me to start coding?