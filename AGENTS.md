# Parking Spot Monitor: agent handoff

Read this before changing anything. It records how the system works, how it is
deployed, and the traps already hit, so you don't have to rediscover them.

## What it does

Watches a UniFi Protect camera pointed at parking spaces in front of the owner's
house and reports, per space, whether a vehicle is present. Results are served
over a REST API and as Prometheus metrics. The owner's Grafana (not in this
repo; `grafana/` is gitignored) reads those.

- Camera: **`Buiten G5`** (UVC G5 Turret Ultra) on a UDM Pro at `192.168.1.1`.
  The older camera `Buiten` (G4 Instant) is no longer used. `Woonkamer` is an
  unrelated indoor camera.
- Monitored spaces: the **three left-most** spaces of a longer row. Others to
  the right are deliberately ignored.
- Spot 1 (`Left Spot`) is currently **`"blocked": true`** because temporary
  building material sits there. Remove the flag when the material is gone.

## How it works

```
UniFi Protect (UDM) --RTSP ch0 2688x1512--> OpenCV frame grab
        -> YOLOv8s vehicle detection (car/truck/motorcycle/bus)
        -> overlap of each detection bbox with each spot bbox
        -> SpotManager (hysteresis) -> FastAPI + Prometheus
```

Loop: every `detection.interval_seconds` (10s) `run_detection_loop` in
`src/parking_monitor/main.py` fetches a frame, detects, matches, updates state.

| Module | Role |
|---|---|
| `main.py` | FastAPI app, lifespan startup, detection loop, daily calibration loop |
| `config.py` | Pydantic models for `config/config.yaml`; `${VAR}` in `password` is resolved from env |
| `camera/unifi_protect.py` | `UniFiProtectClient`: login, camera cache, frame capture, camera list |
| `detection/yolo_detector.py` | YOLOv8 wrapper; CLAHE low-light enhancement when mean brightness < 100 |
| `detection/region_overlap.py` | `ParkingSpot`, overlap math, `match_vehicles_to_spots` |
| `state/spot_manager.py` | Per-spot state with hysteresis; `blocked` spots; `reset_spots()` |
| `calibration/drift_detector.py` | ORB feature matching vs. reference image to detect camera movement |
| `calibration/auto_calibrator.py` | Rewrites `spots.json` after drift (affine transform, else vehicle-based) |
| `api/router.py` | All endpoints (prefix `/api/v1`) |
| `metrics.py` | Prometheus metrics on a private registry |
| `config_tool/` | Desktop GUI/CLI to capture a snapshot and draw spots. Needs a display; interactive |

### Occupancy rules

- Occupied when `intersection(vehicle bbox, spot bbox) / spot bbox area >= min_overlap` (0.2).
  It is bbox-vs-bbox. The polygon is only used for its bounding box, so
  rectangular spots are all that matter.
- A spot takes the best-overlap vehicle. One vehicle can occupy several spots.
- **Hysteresis**: state changes only after `hysteresis_count` (2) consecutive
  agreeing detections. A new spot starts `unknown`.
- **Blocked spots** (`"blocked": true` in `spots.json`) start `occupied`, are
  skipped by detection and never change. They still count as occupied in status
  and metrics.
- Spots are stored in `config/spots.json` with `image_dimensions`. At startup they
  are scaled to the real frame size, so the stored dimensions must match the
  image the coordinates were measured on.

## Configuration

`config/config.yaml` (baked into the image): camera host/user, `camera_id`
(name or id, case-insensitive), snapshot size, YOLO model and thresholds,
calibration settings, API port. The password is **only** read from env
`UFP_PASSWORD`. Never commit it. `.env` is gitignored; `.env.example` shows the
format.

UniFi login is a **local-access** user (`camera`), not an Ubiquiti SSO account.
A 403 `AUTHENTICATION_FAILED_INVALID_CREDENTIALS` means wrong username or
password, not a network problem.

## API (`http://<host>:9878/api/v1`)

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness, `camera_connected`, uptime |
| `GET /status` | All spots and counts |
| `GET /spots/{id}` | One spot |
| `GET /snapshot` | Raw frame (JPEG). Check its size is 2688x1512 |
| `GET /snapshot/annotated` | Frame with spots drawn and coloured by state; best way to verify calibration |
| `GET /cameras` | Cameras the UDM exposes; use to find the exact `camera_id` |
| `GET /camera/info` | Channel list and requested resolution |
| `POST /calibration/check` | Drift check, no changes |
| `POST /calibration/recalibrate` | Force vehicle-based recalibration (**rewrites `spots.json`; see traps**) |
| `POST /calibration/reset-reference` | Replace the reference image with the current frame |
| `GET /metrics` | Prometheus text format |

Metrics (all `parking_*`): `spot_occupied{spot_id,spot_name}`, `spots_total`,
`spots_available`, `spots_occupied`, `spot_state_changes_total`,
`detection_confidence`, `detection_latency_seconds`, `detection_cycles_total`,
`camera_connected`, `rtsp_capture_success_total` / `_failures_total` /
`_retries_total`, `low_light_enhancements_total`, `app_start_time_seconds`,
`last_detection_time_seconds`.

## Deployment

- Source: `github.com/sotoz/parking-spot-monitoring`, branch `main`.
- **Push to `main` builds and pushes the image automatically**
  (`.github/workflows/build.yml` -> `ghcr.io/sotoz/parking-spot-monitoring`,
  tags `latest` and the short SHA, `no-cache: true`, about 5 min). There is
  no Docker on the dev machine. CI is the build path. Watch it with
  `gh run list --workflow build.yml` and `gh run watch <id>`.
- The owner runs it as a **Portainer stack** (not `docker-compose.yml` from the
  repo). The stack uses `image: ghcr.io/sotoz/parking-spot-monitoring:latest`,
  port 9878, env `UFP_PASSWORD`, a healthcheck on `/api/v1/health`, json-file
  logging, and **no volumes**.
- To ship a change: commit, push to `main`, wait for CI, then in Portainer use
  **Re-pull image and redeploy**. `restart` alone does not fetch a new `latest`.
- Because there is no config volume, `config.yaml` and `spots.json` are the
  ones committed in the repo. **Changing spots or camera = commit + push +
  redeploy.**
- The container needs: the UDM at `192.168.1.1` on **443 and 7447 (RTSP)**, and
  outbound internet at first start (see traps).
- Docs-only commits also trigger a rebuild. Push only when you mean to.

### Verify after a deploy

1. `/health` -> `camera_connected: true`
2. `/snapshot` is **2688x1512** (640x360 means the RTSP path is broken)
3. `/snapshot/annotated` boxes sit on the right spaces
4. `/status` matches reality; spot 1 shows `occupied` while blocked
5. Metrics: `parking_rtsp_capture_failures_total` stays 0

## Calibrating a new camera or moved camera

1. Get a **full-resolution** frame from the camera. Use the app's own
   `UniFiProtectClient.get_snapshot` or `GET /snapshot`, not a UniFi preview, which is 640x360.
2. View it, pick the spaces, measure pixel boxes (a grid overlay on a crop
   helps). Cars' bboxes should fall mostly inside the box you draw.
3. Write `config/spots.json` with matching `image_dimensions`. Optionally add
   `"blocked": true`.
4. Render the polygons onto the frame and look at them. Run the real detector
   on the frame and confirm each occupied space overlaps >= `min_overlap`.
5. Commit, push, redeploy. Verify as above.

The GUI in `config_tool/` does the same by mouse, but needs a display.

## Local run / smoke test

The system Python on the dev Mac is 3.9; the project needs **>= 3.11**.

```bash
brew install uv                      # if missing
uv venv --python 3.12 /tmp/pm-venv
uv pip install --python /tmp/pm-venv/bin/python -r requirements.txt
set -a; . ./.env; set +a             # provides UFP_PASSWORD
PYTHONPATH=src /tmp/pm-venv/bin/python -m parking_monitor.main
curl localhost:9878/api/v1/status
```

`PYTHONPATH=src` is required (the package is not installed). The first run
downloads `yolov8s.pt` into the current directory (gitignored). There is no
test suite. Verification is the smoke run above. The harmless
`objc ... AVFFrameReceiver is implemented in both` and
`h264 ... reference picture missing` lines are noise.

## Traps and lessons (already paid for)

- **RTSP URL**: `uiprotect` 25 no longer exposes `rtspAlias` on `Camera` or
  channels, so the old code silently fell back to a 640x360 API snapshot. The
  client now reads `channels[0].rtspAlias` from the raw `bootstrap` JSON and
  builds `rtsp://<host>:7447/<alias>`. If frames become small again, check
  this first. The API snapshot endpoint ignores width/height; the public API
  snapshot needs an API key we don't have. Channel 0 must have RTSP enabled in
  Protect.
- **Daily drift check (03:00) can corrupt `spots.json`.** `detect_drift` returns
  `has_drifted=True` when it finds fewer than 10 ORB matches. A reference
  taken at a different light level (e.g. a night reference vs. a day frame)
  can look like drift. `_recalibrate` then tries an affine transform and, failing that,
  **vehicle-based recalibration**, which snaps spot boxes to detections sorted
  left to right, with no regard for which spot is blocked or which cars are in
  the row. `POST /calibration/recalibrate` does the same. Treat both as
  destructive. If spots look wrong after a day or two, suspect this. Consider
  disabling calibration (`calibration.enabled: false`) if it misbehaves.
- **No persistence**: with no volume, the reference image and any rewritten
  `spots.json` are lost on every redeploy. The reference image is recreated
  from the first frame at startup. Start the container at similar lighting to
  the usual scene or call `/calibration/reset-reference` once.
- **`yolov8n.pt` vs `yolov8s.pt`**: the Dockerfile pre-downloads `yolov8n.pt`
  but `config.yaml` asks for `yolov8s.pt`, which is fetched from GitHub on
  every fresh container start. Offline hosts would fail. Fix the Dockerfile
  if that matters.
- **Night scene**: the camera is in night mode after dark (IR/low light).
  `confidence_threshold: 0.25` and `enhance_low_light: true` exist for that. A
  dark car at night can score low (the black car at the right edge scored 0.26).
- **README is partly stale**: it still describes the G4 Instant and example
  values (`Front Driveway`, port mismatch in troubleshooting). The code and this file are the source of truth.
- `export-image.ps1` and `docker-compose.yml` in the repo are legacy ways of
  getting the image onto the host; the live path is GHCR + Portainer.
- Never print or commit the password. When checking `.env`, inspect shape
  only (key present, length), not value.

## Open items worth knowing

- Building material in spot 1 is temporary. When it's gone, remove `"blocked": true`
  from `config/spots.json`, push and redeploy.
- The reference image and calibration were created at night.
- No automated tests exist.
