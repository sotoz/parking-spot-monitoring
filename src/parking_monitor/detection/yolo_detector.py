"""YOLOv8-based vehicle detection."""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Represents a detected vehicle."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    center: tuple[int, int]
    area: int


class VehicleDetector:
    """
    YOLOv8-based vehicle detector.

    Uses a pre-trained YOLO model to detect vehicles in images.
    Filters results to only include vehicle classes (car, motorcycle, bus, truck).
    """

    # COCO class IDs for vehicles
    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        """
        Initialize the vehicle detector.

        Args:
            model_path: Path to YOLO model weights (will download if not found)
            confidence_threshold: Minimum confidence for detections
            device: Device to run on ('cpu', 'cuda', or None for auto)
        """
        logger.info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        logger.info("YOLO model loaded successfully")

    def detect_vehicles(self, image: np.ndarray) -> list[Detection]:
        """
        Detect vehicles in an image.

        Args:
            image: BGR image as numpy array (OpenCV format)

        Returns:
            List of Detection objects for vehicles found
        """
        # Run inference
        results = self.model(
            image,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )[0]

        detections = []

        for box in results.boxes:
            class_id = int(box.cls[0])

            # Filter for vehicle classes only
            if class_id not in self.VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            detection = Detection(
                class_id=class_id,
                class_name=self.VEHICLE_CLASSES[class_id],
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                center=((x1 + x2) // 2, (y1 + y2) // 2),
                area=(x2 - x1) * (y2 - y1),
            )
            detections.append(detection)

            logger.debug(
                f"Detected {detection.class_name} at {detection.bbox} "
                f"(confidence: {confidence:.2f})"
            )

        logger.debug(f"Found {len(detections)} vehicle(s)")
        return detections

    def detect_from_bytes(self, image_bytes: bytes) -> list[Detection]:
        """
        Detect vehicles from JPEG image bytes.

        Args:
            image_bytes: JPEG image as bytes

        Returns:
            List of Detection objects for vehicles found
        """
        # Decode JPEG to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image bytes")

        return self.detect_vehicles(image)

    def annotate_image(
        self,
        image: np.ndarray,
        detections: list[Detection],
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw detection bounding boxes on an image.

        Args:
            image: BGR image as numpy array
            detections: List of detections to draw
            color: BGR color for bounding boxes
            thickness: Line thickness

        Returns:
            Annotated image copy
        """
        annotated = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            label = f"{det.class_name} {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Background for label
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1,
            )

            # Label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        return annotated
