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
    Includes low-light image enhancement for better night detection.
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
        enhance_low_light: bool = True,
    ):
        """
        Initialize the vehicle detector.

        Args:
            model_path: Path to YOLO model weights (will download if not found)
            confidence_threshold: Minimum confidence for detections
            device: Device to run on ('cpu', 'cuda', or None for auto)
            enhance_low_light: Apply CLAHE enhancement for better night detection
        """
        logger.info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.enhance_low_light = enhance_low_light

        # CLAHE for low-light enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        logger.info(f"YOLO model loaded successfully (low-light enhancement: {enhance_low_light})")

    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image for better low-light detection using CLAHE.

        Args:
            image: BGR image as numpy array

        Returns:
            Enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Split channels
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel (luminance)
        l_enhanced = self.clahe.apply(l)

        # Merge channels back
        lab_enhanced = cv2.merge([l_enhanced, a, b])

        # Convert back to BGR
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        return enhanced

    def _is_low_light(self, image: np.ndarray, threshold: int = 80) -> bool:
        """
        Check if image is low-light based on average brightness.

        Args:
            image: BGR image as numpy array
            threshold: Brightness threshold (0-255)

        Returns:
            True if image is considered low-light
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        return avg_brightness < threshold

    def detect_vehicles(self, image: np.ndarray) -> list[Detection]:
        """
        Detect vehicles in an image.

        Args:
            image: BGR image as numpy array (OpenCV format)

        Returns:
            List of Detection objects for vehicles found
        """
        # Apply low-light enhancement if enabled and image is dark
        processed_image = image
        if self.enhance_low_light and self._is_low_light(image):
            logger.debug("Low-light detected, applying image enhancement")
            processed_image = self._enhance_image(image)

        # Run inference
        results = self.model(
            processed_image,
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
