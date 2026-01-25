# Portainer Deployment Guide

This guide explains how to deploy the Parking Spot Monitor to Portainer without using a public registry.

## Option 1: Build on Portainer Host (Recommended)

### Step 1: Transfer files to your Portainer host

Copy the project to your Portainer host:
```bash
# From your local machine, use scp or rsync
scp -r /path/to/parking_spot_monitoring user@portainer-host:/opt/stacks/parking-monitor
```

### Step 2: Build the image on the host

SSH into your Portainer host and build:
```bash
ssh user@portainer-host
cd /opt/stacks/parking-monitor
docker build -t parking-monitor:latest .
```

### Step 3: Create the stack in Portainer

1. Go to **Stacks** > **Add stack**
2. Name: `parking-monitor`
3. Build method: **Web editor**
4. Paste the contents of `portainer-stack.yml`
5. Under **Environment variables**, add:
   - `UFP_PASSWORD` = your UniFi Protect password
6. Click **Deploy the stack**

---

## Option 2: Import Docker Image

### Step 1: Export the image locally

On your Windows machine (where Docker Desktop is running):
```powershell
docker save parking_spot_monitoring-parking-monitor:latest -o parking-monitor.tar
```

### Step 2: Transfer to Portainer host
```powershell
scp parking-monitor.tar user@portainer-host:/tmp/
```

### Step 3: Load the image on Portainer host
```bash
ssh user@portainer-host
docker load -i /tmp/parking-monitor.tar
docker tag parking_spot_monitoring-parking-monitor:latest parking-monitor:latest
rm /tmp/parking-monitor.tar
```

### Step 4: Deploy stack in Portainer

Same as Option 1, Step 3.

---

## Option 3: Portainer Git Repository (Business Edition)

Since you have Portainer Business, you can use the Git repository feature:

1. Push this project to a **private Git repository** (GitHub, GitLab, Gitea, etc.)
2. In Portainer, go to **Stacks** > **Add stack**
3. Build method: **Repository**
4. Configure:
   - Repository URL: `https://github.com/your-user/parking-spot-monitoring.git`
   - Repository reference: `main`
   - Compose path: `docker-compose.yml`
   - Enable **Build image from Dockerfile**
5. Add authentication if private repo
6. Add environment variables:
   - `UFP_PASSWORD` = your UniFi Protect password
7. Click **Deploy the stack**

---

## Configuration Files

Make sure to create the config directory on your Portainer host with:

```
/opt/stacks/parking-monitor/config/
├── config.yaml          # Main configuration
├── spots.json           # Parking spot definitions
└── reference_calibration.jpg  # (auto-generated)
```

### config.yaml
```yaml
camera:
  host: "192.168.1.1"
  port: 443
  username: "camera"
  password: "${UFP_PASSWORD}"
  verify_ssl: false
  camera_id: "Buiten"
  snapshot_width: 1920
  snapshot_height: 1080

detection:
  model_path: "yolov8n.pt"
  confidence_threshold: 0.5
  min_overlap: 0.3
  interval_seconds: 10
  hysteresis_count: 2

api:
  host: "0.0.0.0"
  port: 8000

calibration:
  enabled: true
  check_hour: 3
  drift_threshold_pixels: 20.0
  reference_image_path: "config/reference_calibration.jpg"
```

---

## Verifying Deployment

After deployment, verify the service is running:

```bash
curl http://your-portainer-host:8000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "camera_connected": true,
  "detection_running": true,
  "uptime_seconds": 123.45
}
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/health` | Health check |
| `GET /api/v1/status` | All parking spots status |
| `GET /api/v1/spots/{id}` | Single spot status |
| `GET /api/v1/snapshot` | Raw camera image |
| `GET /api/v1/snapshot/annotated` | Image with spot overlays |
| `POST /api/v1/calibration/check` | Check for camera drift |
| `POST /api/v1/calibration/recalibrate` | Force recalibration |
