"""Main application entry point."""

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, time
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI

from .api.router import init_calibration_router, init_router, router
from .calibration.auto_calibrator import AutoCalibrator
from .camera.unifi_protect import UniFiProtectClient
from .config import AppConfig, load_config
from .detection.region_overlap import ParkingSpot, match_vehicles_to_spots
from .detection.yolo_detector import VehicleDetector
from .state.spot_manager import SpotManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global state
camera_client: UniFiProtectClient | None = None
detector: VehicleDetector | None = None
spot_manager: SpotManager | None = None
auto_calibrator: AutoCalibrator | None = None
spots: list[ParkingSpot] = []
config: AppConfig | None = None
detection_task: asyncio.Task | None = None
calibration_task: asyncio.Task | None = None


async def run_detection_loop(interval_seconds: int = 10) -> None:
    """
    Main detection loop that runs periodically.

    Fetches snapshots from the camera, runs vehicle detection,
    and updates parking spot states.
    """
    global camera_client, detector, spot_manager, spots, config

    logger.info(f"Starting detection loop (interval: {interval_seconds}s)")

    while True:
        try:
            if camera_client is None or detector is None or spot_manager is None:
                logger.warning("Detection components not initialized")
                await asyncio.sleep(interval_seconds)
                continue

            # Get snapshot from camera
            image_bytes = await camera_client.get_snapshot(
                config.camera.camera_id,
                width=config.camera.snapshot_width,
                height=config.camera.snapshot_height,
            )

            spot_manager.set_camera_connected(True)

            # Run vehicle detection
            detections = detector.detect_from_bytes(image_bytes)
            logger.debug(f"Detected {len(detections)} vehicle(s)")

            # Match vehicles to parking spots
            spot_status = match_vehicles_to_spots(
                detections,
                spots,
                min_overlap=config.detection.min_overlap,
            )

            # Update spot manager
            changed = spot_manager.update_from_detection(spot_status)

            if changed:
                logger.info(f"Spot state changed: {changed}")

        except Exception as e:
            logger.error(f"Detection loop error: {e}")
            if spot_manager:
                spot_manager.set_camera_connected(False)

        await asyncio.sleep(interval_seconds)


def load_spots_config(path: str | Path) -> list[ParkingSpot]:
    """Load parking spot definitions from JSON file."""
    spots_path = Path(path)

    if not spots_path.exists():
        logger.warning(f"Spots config not found: {spots_path}")
        logger.warning("Run the configuration tool first to define parking spots")
        return []

    with open(spots_path) as f:
        data = json.load(f)

    return [ParkingSpot.from_dict(s) for s in data["spots"]]


async def run_calibration_loop(check_hour: int = 3) -> None:
    """
    Daily calibration check loop.

    Runs once per day at the specified hour to check for camera drift
    and automatically recalibrate parking spots if needed.

    Args:
        check_hour: Hour of day (0-23) to run the calibration check
    """
    global camera_client, detector, auto_calibrator, spots, spot_manager, config

    logger.info(f"Starting daily calibration loop (check at {check_hour:02d}:00)")

    last_check_date = None

    while True:
        try:
            now = datetime.now()
            current_hour = now.hour
            current_date = now.date()

            # Check if it's time to run calibration (once per day at check_hour)
            if current_hour == check_hour and last_check_date != current_date:
                logger.info("Running daily camera calibration check...")

                if camera_client is None or auto_calibrator is None:
                    logger.warning("Calibration components not initialized")
                else:
                    # Get current snapshot
                    image_bytes = await camera_client.get_snapshot(
                        config.camera.camera_id,
                        width=config.camera.snapshot_width,
                        height=config.camera.snapshot_height,
                    )

                    # Convert to numpy array
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    current_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    # Check for drift and recalibrate if needed
                    recalibrated, drift_result = auto_calibrator.check_and_recalibrate(
                        current_image
                    )

                    if recalibrated:
                        logger.info("Camera drift detected and spots recalibrated!")

                        # Reload spots configuration
                        spots_path = Path("config/spots.json")
                        new_spots = load_spots_config(spots_path)

                        if new_spots:
                            # Update global spots
                            spots.clear()
                            spots.extend(new_spots)

                            # Reinitialize spot manager with new spots
                            spots_config = [{"id": s.id, "name": s.name} for s in spots]
                            spot_manager.spots.clear()
                            for spot_cfg in spots_config:
                                from .state.models import SpotStatus, ParkingSpotState
                                spot_manager.spots[spot_cfg["id"]] = ParkingSpotState(
                                    id=spot_cfg["id"],
                                    name=spot_cfg["name"],
                                    status=SpotStatus.UNKNOWN,
                                    vehicle=None,
                                    last_changed=now,
                                    last_checked=now,
                                )

                            logger.info(f"Reloaded {len(new_spots)} parking spots after recalibration")
                    else:
                        logger.info(f"Calibration check complete: {drift_result.message}")

                last_check_date = current_date

        except Exception as e:
            logger.error(f"Calibration loop error: {e}")

        # Sleep for an hour before checking again
        await asyncio.sleep(3600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global camera_client, detector, spot_manager, auto_calibrator
    global spots, config, detection_task, calibration_task

    logger.info("Starting Parking Spot Monitor...")

    # Load configuration
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.error("Please create config/config.yaml from the example")
        sys.exit(1)

    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")

    # Load parking spot definitions
    spots_path = Path("config/spots.json")
    spots = load_spots_config(spots_path)

    if not spots:
        logger.warning("No parking spots defined - detection will not work")
        logger.warning("Run: python -m config_tool.gui to define spots")

    # Initialize camera client
    camera_client = UniFiProtectClient(
        host=config.camera.host,
        username=config.camera.username,
        password=config.camera.password,
        port=config.camera.port,
        verify_ssl=config.camera.verify_ssl,
    )

    try:
        await camera_client.connect()
        logger.info("Connected to UniFi Protect")
    except Exception as e:
        logger.error(f"Failed to connect to UniFi Protect: {e}")
        logger.error("Check your camera configuration in config/config.yaml")

    # Initialize detector
    detector = VehicleDetector(
        model_path=config.detection.model_path,
        confidence_threshold=config.detection.confidence_threshold,
    )
    logger.info("Loaded YOLOv8 model")

    # Initialize spot manager
    spots_config = [{"id": s.id, "name": s.name} for s in spots]
    spot_manager = SpotManager(
        spots_config=spots_config,
        hysteresis_count=config.detection.hysteresis_count,
    )

    # Initialize API router
    async def get_snapshot():
        return await camera_client.get_snapshot(
            config.camera.camera_id,
            config.camera.snapshot_width,
            config.camera.snapshot_height,
        )

    async def list_cameras():
        return await camera_client.list_cameras()

    init_router(spot_manager, get_snapshot, list_cameras, spots)

    # Start detection loop
    if spots:
        detection_task = asyncio.create_task(
            run_detection_loop(config.detection.interval_seconds)
        )
        logger.info("Detection loop started")
    else:
        logger.warning("Detection loop not started - no spots defined")

    # Initialize auto-calibrator
    if config.calibration.enabled:
        auto_calibrator = AutoCalibrator(
            spots_config_path="config/spots.json",
            reference_image_path=config.calibration.reference_image_path,
            drift_threshold_pixels=config.calibration.drift_threshold_pixels,
            vehicle_detector=detector,
        )

        # Initialize with current image if reference doesn't exist
        reference_path = Path(config.calibration.reference_image_path)
        if not reference_path.exists():
            try:
                image_bytes = await camera_client.get_snapshot(
                    config.camera.camera_id,
                    width=config.camera.snapshot_width,
                    height=config.camera.snapshot_height,
                )
                nparr = np.frombuffer(image_bytes, np.uint8)
                current_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                auto_calibrator.initialize(current_image)
                logger.info("Initialized calibration reference image")
            except Exception as e:
                logger.error(f"Failed to initialize calibration reference: {e}")

        # Function to reload spots after recalibration
        async def reload_spots_after_calibration():
            global spots, spot_manager
            new_spots = load_spots_config("config/spots.json")
            if new_spots:
                spots.clear()
                spots.extend(new_spots)
                # Update spot manager
                from .state.models import SpotStatus, ParkingSpotState
                now = datetime.now()
                spot_manager.spots.clear()
                for s in new_spots:
                    spot_manager.spots[s.id] = ParkingSpotState(
                        id=s.id,
                        name=s.name,
                        status=SpotStatus.UNKNOWN,
                        vehicle=None,
                        last_changed=now,
                        last_checked=now,
                    )
                logger.info(f"Reloaded {len(new_spots)} spots after calibration")

        # Initialize calibration API endpoints
        init_calibration_router(auto_calibrator, reload_spots_after_calibration)

        # Start daily calibration check loop
        calibration_task = asyncio.create_task(
            run_calibration_loop(config.calibration.check_hour)
        )
        logger.info(f"Daily calibration check enabled (runs at {config.calibration.check_hour:02d}:00)")
    else:
        logger.info("Auto-calibration disabled")

    logger.info(f"Parking Spot Monitor ready on http://{config.api.host}:{config.api.port}")

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down...")

    if detection_task:
        detection_task.cancel()
        try:
            await detection_task
        except asyncio.CancelledError:
            pass

    if calibration_task:
        calibration_task.cancel()
        try:
            await calibration_task
        except asyncio.CancelledError:
            pass

    if camera_client:
        await camera_client.close()

    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Parking Spot Monitor",
    description="API for monitoring parking spot availability using computer vision",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")


def main():
    """Run the application."""
    # Load config just to get API settings
    config_path = Path("config/config.yaml")
    if config_path.exists():
        cfg = load_config(config_path)
        host = cfg.api.host
        port = cfg.api.port
    else:
        host = "0.0.0.0"
        port = 8000

    uvicorn.run(
        "parking_monitor.main:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
