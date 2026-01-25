"""Automatic parking spot recalibration."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..detection.region_overlap import ParkingSpot
from ..detection.yolo_detector import VehicleDetector
from .drift_detector import CameraDriftDetector, DriftResult

logger = logging.getLogger(__name__)


class AutoCalibrator:
    """
    Automatically recalibrates parking spot positions when camera drift is detected.

    Strategies for recalibration:
    1. Transform existing spots using the detected camera shift
    2. Re-detect vehicles and adjust spots to match new positions
    """

    def __init__(
        self,
        spots_config_path: str | Path,
        reference_image_path: str | Path,
        drift_threshold_pixels: float = 20.0,
        vehicle_detector: Optional[VehicleDetector] = None,
    ):
        """
        Initialize the auto-calibrator.

        Args:
            spots_config_path: Path to spots.json configuration
            reference_image_path: Path to store reference image
            drift_threshold_pixels: Minimum shift to trigger recalibration
            vehicle_detector: Optional detector for vehicle-based recalibration
        """
        self.spots_config_path = Path(spots_config_path)
        self.reference_image_path = Path(reference_image_path)

        self.drift_detector = CameraDriftDetector(
            reference_path=reference_image_path,
            drift_threshold_pixels=drift_threshold_pixels,
        )

        self.vehicle_detector = vehicle_detector
        self._last_calibration: Optional[datetime] = None
        self._original_spots: Optional[list[dict]] = None

    def initialize(self, current_image: np.ndarray) -> None:
        """
        Initialize calibration with current image as reference.

        Call this when first setting up or after manual reconfiguration.

        Args:
            current_image: Current camera snapshot
        """
        # Save reference image
        self.drift_detector.save_reference(current_image)

        # Store original spot configuration
        if self.spots_config_path.exists():
            with open(self.spots_config_path) as f:
                config = json.load(f)
                self._original_spots = config.get("spots", [])

        self._last_calibration = datetime.now()
        logger.info("Auto-calibration initialized with reference image")

    def check_and_recalibrate(
        self, current_image: np.ndarray
    ) -> tuple[bool, Optional[DriftResult]]:
        """
        Check for camera drift and recalibrate if needed.

        Args:
            current_image: Current camera snapshot

        Returns:
            Tuple of (recalibration_performed, drift_result)
        """
        # Check for drift
        drift_result = self.drift_detector.detect_drift(current_image)

        if not drift_result.has_drifted:
            logger.info(f"No camera drift detected: {drift_result.message}")
            return False, drift_result

        logger.warning(f"Camera drift detected: {drift_result.message}")

        # Attempt recalibration
        success = self._recalibrate(current_image, drift_result)

        if success:
            # Update reference image to new position
            self.drift_detector.save_reference(current_image)
            self._last_calibration = datetime.now()
            logger.info("Recalibration completed successfully")
        else:
            logger.error("Recalibration failed")

        return success, drift_result

    def _recalibrate(
        self, current_image: np.ndarray, drift_result: DriftResult
    ) -> bool:
        """
        Perform recalibration using transformation or vehicle detection.

        Args:
            current_image: Current camera image
            drift_result: Detected drift information

        Returns:
            True if recalibration succeeded
        """
        # Load current spots config
        if not self.spots_config_path.exists():
            logger.error(f"Spots config not found: {self.spots_config_path}")
            return False

        with open(self.spots_config_path) as f:
            config = json.load(f)

        spots = config.get("spots", [])
        if not spots:
            logger.error("No spots configured")
            return False

        # Try transformation-based recalibration first
        matrix = self.drift_detector.get_transformation_matrix(current_image)

        if matrix is not None:
            new_spots = self._transform_spots(spots, matrix)
            if new_spots:
                config["spots"] = new_spots
                config["last_calibration"] = datetime.now().isoformat()
                config["calibration_method"] = "transformation"

                # Save updated config
                self._save_config(config)
                logger.info("Recalibrated spots using transformation matrix")
                return True

        # Fallback: use vehicle detection to adjust spots
        if self.vehicle_detector is not None:
            new_spots = self._recalibrate_with_detection(
                current_image, spots
            )
            if new_spots:
                config["spots"] = new_spots
                config["last_calibration"] = datetime.now().isoformat()
                config["calibration_method"] = "vehicle_detection"

                self._save_config(config)
                logger.info("Recalibrated spots using vehicle detection")
                return True

        logger.warning("Could not recalibrate automatically")
        return False

    def _transform_spots(
        self, spots: list[dict], matrix: np.ndarray
    ) -> Optional[list[dict]]:
        """
        Transform spot coordinates using affine transformation matrix.

        Args:
            spots: List of spot configurations
            matrix: 2x3 affine transformation matrix

        Returns:
            List of transformed spots, or None if failed
        """
        try:
            new_spots = []

            for spot in spots:
                polygon = spot.get("polygon", [])
                if not polygon:
                    continue

                # Transform each point in the polygon
                points = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
                transformed = cv2.transform(points, matrix)
                new_polygon = transformed.reshape(-1, 2).astype(int).tolist()

                new_spot = spot.copy()
                new_spot["polygon"] = new_polygon
                new_spots.append(new_spot)

            return new_spots

        except Exception as e:
            logger.error(f"Failed to transform spots: {e}")
            return None

    def _recalibrate_with_detection(
        self, current_image: np.ndarray, spots: list[dict]
    ) -> Optional[list[dict]]:
        """
        Recalibrate spots by detecting current vehicles and adjusting.

        This works by:
        1. Detecting vehicles in current image
        2. Matching detected vehicles to existing spots
        3. Adjusting spot positions to center on detected vehicles

        Args:
            current_image: Current camera image
            spots: Current spot configurations

        Returns:
            Adjusted spots, or None if failed
        """
        if self.vehicle_detector is None:
            return None

        detections = self.vehicle_detector.detect_vehicles(current_image)

        if len(detections) < len(spots):
            logger.warning(
                f"Only {len(detections)} vehicles detected for {len(spots)} spots. "
                "Cannot reliably recalibrate."
            )
            # Still try to match what we can
            if not detections:
                return None

        # Sort detections by x-coordinate (left to right)
        detections_sorted = sorted(detections, key=lambda d: d.bbox[0])

        # Sort spots by their leftmost x-coordinate
        spots_sorted = sorted(
            enumerate(spots),
            key=lambda x: min(p[0] for p in x[1].get("polygon", [[0, 0]])),
        )

        new_spots = [s.copy() for s in spots]

        # Match detections to spots (left-to-right order)
        for i, (spot_idx, spot) in enumerate(spots_sorted):
            if i >= len(detections_sorted):
                break

            detection = detections_sorted[i]
            bbox = detection.bbox  # (x1, y1, x2, y2)

            # Create new polygon centered on detection with some padding
            padding_x = int((bbox[2] - bbox[0]) * 0.1)
            padding_y = int((bbox[3] - bbox[1]) * 0.1)

            new_polygon = [
                [bbox[0] - padding_x, bbox[1] - padding_y],
                [bbox[2] + padding_x, bbox[1] - padding_y],
                [bbox[2] + padding_x, bbox[3] + padding_y],
                [bbox[0] - padding_x, bbox[3] + padding_y],
            ]

            new_spots[spot_idx]["polygon"] = new_polygon

            logger.debug(
                f"Adjusted spot '{spot.get('name', spot_idx)}' to match "
                f"detected {detection.class_name} at {bbox}"
            )

        return new_spots

    def _save_config(self, config: dict) -> None:
        """Save configuration to file."""
        # Create backup of original
        backup_path = self.spots_config_path.with_suffix(".json.bak")
        if self.spots_config_path.exists():
            import shutil
            shutil.copy(self.spots_config_path, backup_path)

        with open(self.spots_config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved updated spots config to {self.spots_config_path}")

    def get_last_calibration(self) -> Optional[datetime]:
        """Get timestamp of last calibration."""
        return self._last_calibration

    def force_recalibration(self, current_image: np.ndarray) -> bool:
        """
        Force recalibration regardless of drift detection.

        Useful after manual camera adjustment.

        Args:
            current_image: Current camera snapshot

        Returns:
            True if recalibration succeeded
        """
        logger.info("Forcing recalibration...")

        # Re-detect vehicles and recalibrate
        if self.vehicle_detector is not None:
            if not self.spots_config_path.exists():
                return False

            with open(self.spots_config_path) as f:
                config = json.load(f)

            spots = config.get("spots", [])
            new_spots = self._recalibrate_with_detection(current_image, spots)

            if new_spots:
                config["spots"] = new_spots
                config["last_calibration"] = datetime.now().isoformat()
                config["calibration_method"] = "forced_vehicle_detection"
                self._save_config(config)

                # Update reference
                self.drift_detector.save_reference(current_image)
                self._last_calibration = datetime.now()
                return True

        return False
