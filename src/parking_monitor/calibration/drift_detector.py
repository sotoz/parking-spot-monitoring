"""Camera drift detection using feature matching."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Result of camera drift detection."""

    has_drifted: bool
    shift_x: float  # Horizontal shift in pixels
    shift_y: float  # Vertical shift in pixels
    confidence: float  # Confidence in the detection (0-1)
    match_count: int  # Number of feature matches found
    message: str


class CameraDriftDetector:
    """
    Detects if camera has moved by comparing current image to a reference.

    Uses ORB feature detection and matching to find corresponding points
    between images, then estimates the transformation.
    """

    def __init__(
        self,
        reference_path: str | Path,
        drift_threshold_pixels: float = 20.0,
        min_match_count: int = 10,
        match_ratio: float = 0.75,
    ):
        """
        Initialize the drift detector.

        Args:
            reference_path: Path to store/load reference image
            drift_threshold_pixels: Minimum shift to consider as drift
            min_match_count: Minimum matches needed for reliable detection
            match_ratio: Lowe's ratio test threshold for good matches
        """
        self.reference_path = Path(reference_path)
        self.drift_threshold = drift_threshold_pixels
        self.min_match_count = min_match_count
        self.match_ratio = match_ratio

        # Initialize ORB detector
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Reference image data
        self._reference_image: Optional[np.ndarray] = None
        self._reference_keypoints = None
        self._reference_descriptors = None

    def save_reference(self, image: np.ndarray) -> None:
        """
        Save current image as the reference for drift detection.

        Args:
            image: BGR image as numpy array
        """
        self._reference_image = image.copy()

        # Compute features for reference
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self._reference_keypoints, self._reference_descriptors = self.orb.detectAndCompute(
            gray, None
        )

        # Save to disk
        self.reference_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.reference_path), image)

        logger.info(
            f"Saved reference image with {len(self._reference_keypoints)} keypoints"
        )

    def load_reference(self) -> bool:
        """
        Load reference image from disk.

        Returns:
            True if reference was loaded successfully
        """
        if not self.reference_path.exists():
            logger.warning(f"Reference image not found: {self.reference_path}")
            return False

        self._reference_image = cv2.imread(str(self.reference_path))
        if self._reference_image is None:
            logger.error(f"Failed to load reference image: {self.reference_path}")
            return False

        # Compute features
        gray = cv2.cvtColor(self._reference_image, cv2.COLOR_BGR2GRAY)
        self._reference_keypoints, self._reference_descriptors = self.orb.detectAndCompute(
            gray, None
        )

        logger.info(
            f"Loaded reference image with {len(self._reference_keypoints)} keypoints"
        )
        return True

    def has_reference(self) -> bool:
        """Check if a reference image is loaded."""
        return self._reference_descriptors is not None

    def detect_drift(self, current_image: np.ndarray) -> DriftResult:
        """
        Detect if camera has drifted from reference position.

        Args:
            current_image: Current camera image (BGR numpy array)

        Returns:
            DriftResult with drift information
        """
        if not self.has_reference():
            if not self.load_reference():
                return DriftResult(
                    has_drifted=False,
                    shift_x=0,
                    shift_y=0,
                    confidence=0,
                    match_count=0,
                    message="No reference image available",
                )

        # Detect features in current image
        gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if descriptors is None or len(descriptors) < self.min_match_count:
            return DriftResult(
                has_drifted=False,
                shift_x=0,
                shift_y=0,
                confidence=0,
                match_count=0,
                message="Not enough features in current image",
            )

        # Match features using KNN
        try:
            matches = self.matcher.knnMatch(
                self._reference_descriptors, descriptors, k=2
            )
        except cv2.error as e:
            logger.error(f"Feature matching failed: {e}")
            return DriftResult(
                has_drifted=False,
                shift_x=0,
                shift_y=0,
                confidence=0,
                match_count=0,
                message=f"Feature matching failed: {e}",
            )

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.match_ratio * n.distance:
                    good_matches.append(m)

        if len(good_matches) < self.min_match_count:
            return DriftResult(
                has_drifted=True,  # Assume drift if we can't match
                shift_x=0,
                shift_y=0,
                confidence=0,
                match_count=len(good_matches),
                message=f"Only {len(good_matches)} matches found (need {self.min_match_count}). Camera may have moved significantly.",
            )

        # Extract matched point coordinates
        ref_pts = np.float32(
            [self._reference_keypoints[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        cur_pts = np.float32(
            [keypoints[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        # Estimate transformation using RANSAC
        matrix, mask = cv2.estimateAffinePartial2D(
            ref_pts, cur_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0
        )

        if matrix is None:
            return DriftResult(
                has_drifted=True,
                shift_x=0,
                shift_y=0,
                confidence=0,
                match_count=len(good_matches),
                message="Could not estimate transformation",
            )

        # Extract translation from transformation matrix
        shift_x = matrix[0, 2]
        shift_y = matrix[1, 2]

        # Calculate confidence based on inlier ratio
        inliers = mask.sum() if mask is not None else 0
        confidence = inliers / len(good_matches) if good_matches else 0

        # Check if drift exceeds threshold
        total_shift = np.sqrt(shift_x**2 + shift_y**2)
        has_drifted = total_shift > self.drift_threshold

        if has_drifted:
            message = f"Camera drift detected: {total_shift:.1f}px (threshold: {self.drift_threshold}px)"
        else:
            message = f"Camera stable: {total_shift:.1f}px shift"

        logger.info(message)

        return DriftResult(
            has_drifted=has_drifted,
            shift_x=shift_x,
            shift_y=shift_y,
            confidence=confidence,
            match_count=len(good_matches),
            message=message,
        )

    def get_transformation_matrix(
        self, current_image: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Get the full transformation matrix from reference to current.

        Args:
            current_image: Current camera image

        Returns:
            2x3 affine transformation matrix, or None if failed
        """
        if not self.has_reference():
            return None

        gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if descriptors is None:
            return None

        matches = self.matcher.knnMatch(
            self._reference_descriptors, descriptors, k=2
        )

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.match_ratio * n.distance:
                    good_matches.append(m)

        if len(good_matches) < self.min_match_count:
            return None

        ref_pts = np.float32(
            [self._reference_keypoints[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        cur_pts = np.float32(
            [keypoints[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        matrix, _ = cv2.estimateAffinePartial2D(
            ref_pts, cur_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0
        )

        return matrix
