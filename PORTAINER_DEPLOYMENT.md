# Portainer Deployment Guide

This guide explains how to deploy the Parking Spot Monitor to Portainer using the pre-built Docker image from GitHub Container Registry.

## Prerequisites

- Portainer Business Edition (or Community with registry support)
- GitHub account with access to the repository

---

## Deployment Steps (Recommended)

### Step 1: Create GitHub Personal Access Token

1. Go to https://github.com/settings/tokens
2. Click **Generate new token (classic)**
3. Name: `portainer-ghcr`
4. Select scope: **read:packages**
5. Generate and **copy the token**

### Step 2: Add GitHub Container Registry to Portainer

1. In Portainer, go to **Registries** → **Add registry**
2. Select **Custom registry**
3. Configure:
   - **Name**: `GitHub Container Registry`
   - **Registry URL**: `ghcr.io`
   - **Authentication**: On
   - **Username**: your GitHub username
   - **Password**: the token from Step 1
4. Click **Add registry**

### Step 3: Deploy the Stack

1. Go to **Stacks** → **Add stack**
2. **Name**: `parking-monitor`
3. **Build method**: Web editor
4. Paste the following:

```yaml
services:
  parking-monitor:
    image: ghcr.io/sotoz/parking-spot-monitoring:latest
    container_name: parking-monitor
    restart: unless-stopped
    ports:
      - "9878:9878"
    environment:
      - UFP_PASSWORD=${UFP_PASSWORD}
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:9878/api/v1/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

5. Under **Environment variables**, add:
   - **Name**: `UFP_PASSWORD`
   - **Value**: your UniFi Protect password
6. Click **Deploy the stack**

---

## Automatic Image Updates

The Docker image is automatically built and pushed to GitHub Container Registry whenever changes are pushed to the `main` branch via GitHub Actions.

To update your deployment:
1. Go to your stack in Portainer
2. Click **Pull and redeploy**

---

## Configuration

The configuration is baked into the Docker image. To modify settings, update the files in the repository and push to trigger a new build:

- `config/config.yaml` - Main configuration
- `config/spots.json` - Parking spot definitions

### Current Configuration

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
  port: 9878

calibration:
  enabled: true
  check_hour: 3
  drift_threshold_pixels: 20.0
```

---

## Verifying Deployment

After deployment, verify the service is running:

```bash
curl http://your-portainer-host:9878/api/v1/health
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

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/status` | GET | All parking spots status |
| `/api/v1/spots/{id}` | GET | Single spot status |
| `/api/v1/snapshot` | GET | Raw camera image (JPEG) |
| `/api/v1/snapshot/annotated` | GET | Image with spot overlays |
| `/api/v1/cameras` | GET | List available cameras |
| `/api/v1/calibration/check` | POST | Check for camera drift |
| `/api/v1/calibration/recalibrate` | POST | Force recalibration |
| `/api/v1/calibration/reset-reference` | POST | Reset reference image |

---

## Troubleshooting

### "denied" error when deploying

Portainer can't authenticate to ghcr.io. Verify:
1. Registry is added in Portainer with correct credentials
2. GitHub token has `read:packages` scope
3. Token hasn't expired

### NNPACK warning in logs

```
Could not initialize NNPACK! Reason: Unsupported hardware.
```

This is harmless - PyTorch falls back to standard CPU operations. Add this to suppress:
```yaml
environment:
  - UFP_PASSWORD=${UFP_PASSWORD}
  - TORCH_NNPACK_ENABLED=0
```

### Container keeps restarting

Check logs in Portainer. Common issues:
- Wrong UniFi Protect password
- UDM Pro not reachable from container
- Camera name doesn't match
