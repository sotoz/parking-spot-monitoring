"""Data models for parking spot state."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class SpotStatus(str, Enum):
    """Status of a parking spot."""

    AVAILABLE = "available"
    OCCUPIED = "occupied"
    UNKNOWN = "unknown"


class VehicleInfo(BaseModel):
    """Information about a detected vehicle in a spot."""

    vehicle_type: str
    confidence: float
    overlap: float
    detected_at: datetime


class ParkingSpotState(BaseModel):
    """Current state of a parking spot."""

    id: str
    name: str
    status: SpotStatus
    vehicle: Optional[VehicleInfo] = None
    last_changed: datetime
    last_checked: datetime


class SystemState(BaseModel):
    """Overall system state."""

    spots: list[ParkingSpotState]
    last_detection_time: datetime
    camera_connected: bool
    detection_enabled: bool
