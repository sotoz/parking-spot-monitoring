"""API request and response schemas."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from ..state.models import SpotStatus


class SpotResponse(BaseModel):
    """Response schema for a single parking spot."""

    id: str
    name: str
    status: SpotStatus
    vehicle_type: Optional[str] = None
    confidence: Optional[float] = None
    last_changed: datetime
    last_checked: datetime


class StatusResponse(BaseModel):
    """Response schema for overall parking status."""

    total_spots: int
    available: int
    occupied: int
    unknown: int
    spots: list[SpotResponse]
    last_update: datetime
    camera_connected: bool


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    camera_connected: bool
    detection_running: bool
    uptime_seconds: float


class CameraInfo(BaseModel):
    """Camera information."""

    id: str
    name: str
    type: str
    is_connected: bool


class CamerasResponse(BaseModel):
    """Response for listing cameras."""

    cameras: list[CameraInfo]
