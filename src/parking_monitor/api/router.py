"""FastAPI route definitions."""

import base64
import logging
from datetime import datetime
from typing import Callable, Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Response

from ..detection.region_overlap import ParkingSpot
from ..state.models import SpotStatus
from ..state.spot_manager import SpotManager
from .schemas import (
    CameraInfo,
    CamerasResponse,
    HealthResponse,
    SpotResponse,
    StatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependencies injected at startup
_spot_manager: Optional[SpotManager] = None
_get_snapshot_func: Optional[Callable] = None
_list_cameras_func: Optional[Callable] = None
_spots_config: list[ParkingSpot] = []
_start_time: datetime = datetime.now()


def init_router(
    spot_manager: SpotManager,
    get_snapshot_func: Callable,
    list_cameras_func: Callable,
    spots_config: list[ParkingSpot],
) -> None:
    """
    Initialize router with dependencies.

    Args:
        spot_manager: SpotManager instance for state
        get_snapshot_func: Async function to get camera snapshot
        list_cameras_func: Async function to list cameras
        spots_config: List of parking spot configurations
    """
    global _spot_manager, _get_snapshot_func, _list_cameras_func
    global _spots_config, _start_time

    _spot_manager = spot_manager
    _get_snapshot_func = get_snapshot_func
    _list_cameras_func = list_cameras_func
    _spots_config = spots_config
    _start_time = datetime.now()

    logger.info("API router initialized")


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns basic health information about the service.
    """
    uptime = (datetime.now() - _start_time).total_seconds()

    camera_connected = False
    if _spot_manager:
        state = _spot_manager.get_state()
        camera_connected = state.camera_connected

    return HealthResponse(
        status="healthy",
        camera_connected=camera_connected,
        detection_running=_spot_manager is not None,
        uptime_seconds=uptime,
    )


@router.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """
    Get overall parking status.

    Returns the current status of all monitored parking spots including
    availability counts and individual spot details.
    """
    if _spot_manager is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    state = _spot_manager.get_state()

    spots = [
        SpotResponse(
            id=s.id,
            name=s.name,
            status=s.status,
            vehicle_type=s.vehicle.vehicle_type if s.vehicle else None,
            confidence=s.vehicle.confidence if s.vehicle else None,
            last_changed=s.last_changed,
            last_checked=s.last_checked,
        )
        for s in state.spots
    ]

    return StatusResponse(
        total_spots=len(spots),
        available=sum(1 for s in spots if s.status == SpotStatus.AVAILABLE),
        occupied=sum(1 for s in spots if s.status == SpotStatus.OCCUPIED),
        unknown=sum(1 for s in spots if s.status == SpotStatus.UNKNOWN),
        spots=spots,
        last_update=state.last_detection_time,
        camera_connected=state.camera_connected,
    )


@router.get("/spots/{spot_id}", response_model=SpotResponse)
async def get_spot(spot_id: str) -> SpotResponse:
    """
    Get status for a specific parking spot.

    Args:
        spot_id: The ID of the parking spot to query
    """
    if _spot_manager is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    spot = _spot_manager.get_spot(spot_id)
    if spot is None:
        raise HTTPException(status_code=404, detail=f"Spot '{spot_id}' not found")

    return SpotResponse(
        id=spot.id,
        name=spot.name,
        status=spot.status,
        vehicle_type=spot.vehicle.vehicle_type if spot.vehicle else None,
        confidence=spot.vehicle.confidence if spot.vehicle else None,
        last_changed=spot.last_changed,
        last_checked=spot.last_checked,
    )


@router.get("/snapshot")
async def get_snapshot() -> Response:
    """
    Get current camera snapshot as JPEG.

    Returns the raw camera image without annotations.
    """
    if _get_snapshot_func is None:
        raise HTTPException(status_code=503, detail="Camera not configured")

    try:
        image_bytes = await _get_snapshot_func()
        return Response(content=image_bytes, media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Failed to get snapshot: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get snapshot: {e}")


@router.get("/snapshot/annotated")
async def get_annotated_snapshot() -> Response:
    """
    Get camera snapshot with parking spot overlays and status indicators.

    Returns the camera image with:
    - Parking spot boundaries drawn
    - Green overlay for available spots
    - Red overlay for occupied spots
    - Yellow overlay for unknown spots
    - Labels with spot names and status
    """
    if _get_snapshot_func is None or _spot_manager is None:
        raise HTTPException(status_code=503, detail="Service not configured")

    try:
        # Get current snapshot
        image_bytes = await _get_snapshot_func()

        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=500, detail="Failed to decode image")

        # Get current state
        state = _spot_manager.get_state()
        spot_states = {s.id: s for s in state.spots}

        # Draw spot overlays
        for spot in _spots_config:
            spot_state = spot_states.get(spot.id)
            if spot_state is None:
                continue

            # Determine color based on status
            if spot_state.status == SpotStatus.AVAILABLE:
                color = (0, 255, 0)  # Green
            elif spot_state.status == SpotStatus.OCCUPIED:
                color = (0, 0, 255)  # Red
            else:
                color = (0, 255, 255)  # Yellow

            # Draw polygon outline
            pts = np.array(spot.polygon, np.int32)
            cv2.polylines(image, [pts], True, color, 2)

            # Semi-transparent fill
            overlay = image.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)

            # Draw label
            center = np.mean(pts, axis=0).astype(int)
            label = f"{spot.name}: {spot_state.status.value}"

            # Background for label
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                image,
                (center[0] - 5, center[1] - label_size[1] - 10),
                (center[0] + label_size[0] + 5, center[1] + 5),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                image,
                label,
                (center[0], center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # Encode annotated image
        _, buffer = cv2.imencode(".jpg", image)

        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create annotated snapshot: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create annotated snapshot: {e}"
        )


@router.get("/cameras", response_model=CamerasResponse)
async def list_cameras() -> CamerasResponse:
    """
    List all available cameras from UniFi Protect.

    Useful for finding the correct camera_id to use in configuration.
    """
    if _list_cameras_func is None:
        raise HTTPException(status_code=503, detail="Camera not configured")

    try:
        cameras = await _list_cameras_func()
        return CamerasResponse(
            cameras=[
                CameraInfo(
                    id=c["id"],
                    name=c["name"],
                    type=c["type"],
                    is_connected=c["is_connected"],
                )
                for c in cameras
            ]
        )
    except Exception as e:
        logger.error(f"Failed to list cameras: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list cameras: {e}")


# Calibration endpoints - these will be initialized separately
_calibrator = None
_reload_spots_func = None


def init_calibration_router(calibrator, reload_spots_func) -> None:
    """Initialize calibration endpoints."""
    global _calibrator, _reload_spots_func
    _calibrator = calibrator
    _reload_spots_func = reload_spots_func


@router.post("/calibration/check")
async def check_calibration():
    """
    Manually trigger a camera drift check.

    Returns drift detection results without forcing recalibration.
    """
    if _calibrator is None or _get_snapshot_func is None:
        raise HTTPException(status_code=503, detail="Calibration not configured")

    try:
        image_bytes = await _get_snapshot_func()
        nparr = np.frombuffer(image_bytes, np.uint8)
        current_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        drift_result = _calibrator.drift_detector.detect_drift(current_image)

        # Convert numpy types to Python types for JSON serialization
        return {
            "has_drifted": bool(drift_result.has_drifted),
            "shift_x": float(drift_result.shift_x),
            "shift_y": float(drift_result.shift_y),
            "confidence": float(drift_result.confidence),
            "match_count": int(drift_result.match_count),
            "message": str(drift_result.message),
        }
    except Exception as e:
        logger.error(f"Calibration check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Calibration check failed: {e}")


@router.post("/calibration/recalibrate")
async def force_recalibration():
    """
    Force recalibration of parking spots.

    Use this after manually adjusting the camera to update spot positions.
    """
    if _calibrator is None or _get_snapshot_func is None:
        raise HTTPException(status_code=503, detail="Calibration not configured")

    try:
        image_bytes = await _get_snapshot_func()
        nparr = np.frombuffer(image_bytes, np.uint8)
        current_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        success = _calibrator.force_recalibration(current_image)

        if success and _reload_spots_func:
            await _reload_spots_func()

        return {
            "success": success,
            "message": "Recalibration completed" if success else "Recalibration failed",
        }
    except Exception as e:
        logger.error(f"Recalibration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recalibration failed: {e}")


@router.post("/calibration/reset-reference")
async def reset_calibration_reference():
    """
    Reset the calibration reference image to current camera view.

    Use this after confirming spot positions are correct.
    """
    if _calibrator is None or _get_snapshot_func is None:
        raise HTTPException(status_code=503, detail="Calibration not configured")

    try:
        image_bytes = await _get_snapshot_func()
        nparr = np.frombuffer(image_bytes, np.uint8)
        current_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        _calibrator.initialize(current_image)

        return {
            "success": True,
            "message": "Calibration reference reset to current camera view",
        }
    except Exception as e:
        logger.error(f"Failed to reset calibration reference: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to reset reference: {e}"
        )
