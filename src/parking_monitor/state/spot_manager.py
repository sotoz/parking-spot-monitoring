"""Parking spot state management with hysteresis."""

import logging
from datetime import datetime
from typing import Optional

from ..detection.region_overlap import VehicleMatch
from ..metrics import record_spot_change, update_spot_status, update_spot_counts, record_detection_confidence
from .models import ParkingSpotState, SpotStatus, SystemState, VehicleInfo

logger = logging.getLogger(__name__)


class SpotManager:
    """
    Manages parking spot state with hysteresis to prevent flapping.

    Hysteresis: A spot must be detected as changed for N consecutive
    checks before the state actually changes. This prevents false
    positives from brief occlusions or detection errors.
    """

    def __init__(
        self,
        spots_config: list[dict],
        hysteresis_count: int = 2,
    ):
        """
        Initialize the spot manager.

        Args:
            spots_config: List of spot configuration dicts with 'id' and 'name'
            hysteresis_count: Number of consecutive detections before state change
        """
        self.hysteresis_count = hysteresis_count

        # Initialize spot states
        self.spots: dict[str, ParkingSpotState] = {}
        self._pending_changes: dict[str, int] = {}  # spot_id -> consecutive count
        self._pending_state: dict[str, SpotStatus] = {}
        self._camera_connected = False
        self._detection_enabled = True

        now = datetime.now()
        for spot in spots_config:
            spot_id = spot["id"]
            self.spots[spot_id] = ParkingSpotState(
                id=spot_id,
                name=spot["name"],
                status=SpotStatus.UNKNOWN,
                vehicle=None,
                last_changed=now,
                last_checked=now,
            )
            self._pending_changes[spot_id] = 0

        logger.info(f"Initialized SpotManager with {len(self.spots)} spots")

    def update_from_detection(
        self,
        detection_results: dict[str, Optional[VehicleMatch]],
    ) -> list[str]:
        """
        Update spot states based on detection results.

        Args:
            detection_results: Dict mapping spot_id to VehicleMatch or None

        Returns:
            List of spot IDs that changed state
        """
        now = datetime.now()
        changed_spots = []

        for spot_id, vehicle_match in detection_results.items():
            if spot_id not in self.spots:
                logger.warning(f"Unknown spot ID in detection results: {spot_id}")
                continue

            spot = self.spots[spot_id]
            spot.last_checked = now

            # Determine detected status
            detected_status = (
                SpotStatus.OCCUPIED if vehicle_match else SpotStatus.AVAILABLE
            )

            # Check if this matches current state
            if detected_status == spot.status:
                # Reset pending change counter
                self._pending_changes[spot_id] = 0
                self._pending_state.pop(spot_id, None)
                continue

            # State differs - apply hysteresis
            if self._pending_state.get(spot_id) == detected_status:
                # Same pending state - increment counter
                self._pending_changes[spot_id] += 1
            else:
                # Different pending state - reset
                self._pending_changes[spot_id] = 1
                self._pending_state[spot_id] = detected_status

            logger.debug(
                f"Spot {spot_id}: pending change to {detected_status.value} "
                f"({self._pending_changes[spot_id]}/{self.hysteresis_count})"
            )

            # Check if hysteresis threshold reached
            if self._pending_changes.get(spot_id, 0) >= self.hysteresis_count:
                # Apply state change
                old_status = spot.status
                spot.status = detected_status
                spot.last_changed = now

                if vehicle_match:
                    spot.vehicle = VehicleInfo(
                        vehicle_type=vehicle_match.vehicle_type,
                        confidence=vehicle_match.confidence,
                        overlap=vehicle_match.overlap,
                        detected_at=now,
                    )
                else:
                    spot.vehicle = None

                # Reset pending state
                self._pending_changes[spot_id] = 0
                self._pending_state.pop(spot_id, None)

                changed_spots.append(spot_id)
                logger.info(
                    f"Spot '{spot.name}' changed: {old_status.value} -> {detected_status.value}"
                )

                # Record metrics for state change
                record_spot_change(
                    spot_id=spot_id,
                    spot_name=spot.name,
                    became_occupied=(detected_status == SpotStatus.OCCUPIED),
                    hour=now.hour,
                )

            # Record confidence metrics when vehicle detected
            if vehicle_match:
                record_detection_confidence(
                    spot_id=spot_id,
                    spot_name=self.spots[spot_id].name,
                    confidence=vehicle_match.confidence,
                )

        # Update spot status gauges
        for spot_id, spot in self.spots.items():
            update_spot_status(
                spot_id=spot_id,
                spot_name=spot.name,
                is_occupied=(spot.status == SpotStatus.OCCUPIED),
            )

        # Update overall counts
        update_spot_counts(
            total=len(self.spots),
            available=self.get_available_count(),
            occupied=self.get_occupied_count(),
        )

        return changed_spots

    def get_state(self) -> SystemState:
        """Get current system state."""
        spots_list = list(self.spots.values())

        last_detection = max(
            (s.last_checked for s in spots_list),
            default=datetime.now(),
        )

        return SystemState(
            spots=spots_list,
            last_detection_time=last_detection,
            camera_connected=self._camera_connected,
            detection_enabled=self._detection_enabled,
        )

    def get_spot(self, spot_id: str) -> Optional[ParkingSpotState]:
        """Get state for a specific spot."""
        return self.spots.get(spot_id)

    def get_available_count(self) -> int:
        """Get count of available spots."""
        return sum(1 for s in self.spots.values() if s.status == SpotStatus.AVAILABLE)

    def get_occupied_count(self) -> int:
        """Get count of occupied spots."""
        return sum(1 for s in self.spots.values() if s.status == SpotStatus.OCCUPIED)

    def set_camera_connected(self, connected: bool) -> None:
        """Update camera connection status."""
        self._camera_connected = connected

    def set_detection_enabled(self, enabled: bool) -> None:
        """Update detection enabled status."""
        self._detection_enabled = enabled
