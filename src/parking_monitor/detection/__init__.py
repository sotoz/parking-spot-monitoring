"""Vehicle detection module."""

from .yolo_detector import VehicleDetector, Detection
from .region_overlap import ParkingSpot, match_vehicles_to_spots

__all__ = ["VehicleDetector", "Detection", "ParkingSpot", "match_vehicles_to_spots"]
