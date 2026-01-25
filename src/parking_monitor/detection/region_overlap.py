"""Region overlap calculation for parking spot occupancy detection."""

from dataclasses import dataclass
from typing import Optional

from .yolo_detector import Detection


@dataclass
class ParkingSpot:
    """Represents a user-defined parking spot region."""

    id: str
    name: str
    polygon: list[tuple[int, int]]  # List of (x, y) vertices

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Get bounding box (x1, y1, x2, y2) from polygon."""
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return (min(xs), min(ys), max(xs), max(ys))

    @classmethod
    def from_dict(cls, data: dict) -> "ParkingSpot":
        """Create ParkingSpot from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            polygon=[tuple(p) for p in data["polygon"]],
        )


def calculate_iou(
    box1: tuple[int, int, int, int],
    box2: tuple[int, int, int, int],
) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.

    Args:
        box1, box2: Bounding boxes as (x1, y1, x2, y2)

    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def calculate_overlap_percentage(
    vehicle_bbox: tuple[int, int, int, int],
    spot: ParkingSpot,
) -> float:
    """
    Calculate what percentage of the parking spot is covered by the vehicle.

    This is more useful than IoU for determining if a vehicle is "in" a spot,
    as it accounts for the relative size difference between vehicles and spots.

    Args:
        vehicle_bbox: Vehicle bounding box as (x1, y1, x2, y2)
        spot: Parking spot to check

    Returns:
        Overlap percentage (0.0 to 1.0)
    """
    spot_bbox = spot.bbox

    # Calculate intersection rectangle
    x1 = max(vehicle_bbox[0], spot_bbox[0])
    y1 = max(vehicle_bbox[1], spot_bbox[1])
    x2 = min(vehicle_bbox[2], spot_bbox[2])
    y2 = min(vehicle_bbox[3], spot_bbox[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    spot_area = (spot_bbox[2] - spot_bbox[0]) * (spot_bbox[3] - spot_bbox[1])

    return intersection / spot_area if spot_area > 0 else 0


def check_vehicle_in_spot(
    vehicle_bbox: tuple[int, int, int, int],
    spot: ParkingSpot,
    min_overlap: float = 0.3,
) -> bool:
    """
    Check if a vehicle is considered to be in a parking spot.

    Args:
        vehicle_bbox: Vehicle bounding box as (x1, y1, x2, y2)
        spot: Parking spot to check
        min_overlap: Minimum overlap percentage threshold

    Returns:
        True if vehicle is in the spot
    """
    overlap = calculate_overlap_percentage(vehicle_bbox, spot)
    return overlap >= min_overlap


@dataclass
class VehicleMatch:
    """Information about a vehicle matched to a parking spot."""

    vehicle_type: str
    confidence: float
    overlap: float
    bbox: tuple[int, int, int, int]


def match_vehicles_to_spots(
    detections: list[Detection],
    spots: list[ParkingSpot],
    min_overlap: float = 0.3,
) -> dict[str, Optional[VehicleMatch]]:
    """
    Match detected vehicles to parking spots.

    Each spot is matched to at most one vehicle (the one with highest overlap).
    A vehicle can occupy multiple spots if it's large enough.

    Args:
        detections: List of detected vehicles
        spots: List of parking spots to check
        min_overlap: Minimum overlap percentage to consider occupied

    Returns:
        Dict mapping spot_id to VehicleMatch (or None if empty)
    """
    spot_status: dict[str, Optional[VehicleMatch]] = {spot.id: None for spot in spots}

    for spot in spots:
        best_match: Optional[VehicleMatch] = None
        best_overlap = 0.0

        for detection in detections:
            overlap = calculate_overlap_percentage(detection.bbox, spot)

            if overlap >= min_overlap and overlap > best_overlap:
                best_overlap = overlap
                best_match = VehicleMatch(
                    vehicle_type=detection.class_name,
                    confidence=detection.confidence,
                    overlap=overlap,
                    bbox=detection.bbox,
                )

        spot_status[spot.id] = best_match

    return spot_status
