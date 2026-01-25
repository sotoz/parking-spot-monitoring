"""Interactive GUI tool for defining parking spot regions."""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class SpotDefinition:
    """Parking spot definition for storage."""

    id: str
    name: str
    polygon: list[tuple[int, int]]


class ParkingSpotConfigurator:
    """
    Interactive GUI tool for defining parking spot regions.

    Usage:
        1. Run the tool with a snapshot from your camera
        2. Click and drag to draw rectangles for each parking spot
        3. Press 'n' to name the spot and confirm
        4. Press 's' to save configuration
        5. Press 'u' to undo last spot
        6. Press 'q' to quit
    """

    WINDOW_NAME = "Parking Spot Configuration"

    def __init__(self, image: np.ndarray):
        """
        Initialize the configurator.

        Args:
            image: Camera snapshot as BGR numpy array
        """
        self.original_image = image.copy()
        self.display_image = image.copy()
        self.spots: list[SpotDefinition] = []

        # Drawing state
        self.drawing = False
        self.start_point: Optional[tuple[int, int]] = None
        self.end_point: Optional[tuple[int, int]] = None
        self.spot_counter = 1

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing rectangles."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                self._update_display()

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            self._update_display()

    def _update_display(self):
        """Redraw the display with all spots and current drawing."""
        self.display_image = self.original_image.copy()

        # Draw existing spots
        for spot in self.spots:
            pts = np.array(spot.polygon, np.int32)
            cv2.polylines(self.display_image, [pts], True, (0, 255, 0), 2)

            # Semi-transparent fill
            overlay = self.display_image.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.2, self.display_image, 0.8, 0, self.display_image)

            # Draw label
            center = np.mean(pts, axis=0).astype(int)
            label_size, _ = cv2.getTextSize(
                spot.name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(
                self.display_image,
                (center[0] - 5, center[1] - label_size[1] - 10),
                (center[0] + label_size[0] + 5, center[1] + 5),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                self.display_image,
                spot.name,
                (center[0], center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # Draw current rectangle being drawn
        if self.start_point and self.end_point:
            cv2.rectangle(
                self.display_image,
                self.start_point,
                self.end_point,
                (0, 0, 255),  # Red for in-progress
                2,
            )

        # Draw instructions
        instructions = [
            "Click & drag: Draw rectangle",
            "N: Name & add spot",
            "U: Undo last",
            "S: Save config",
            "Q: Quit",
        ]
        y_offset = 30
        for instruction in instructions:
            cv2.putText(
                self.display_image,
                instruction,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            y_offset += 20

    def _add_current_spot(self):
        """Add the currently drawn rectangle as a parking spot."""
        if not self.start_point or not self.end_point:
            print("No rectangle drawn. Draw a rectangle first.")
            return

        x1, y1 = self.start_point
        x2, y2 = self.end_point

        # Normalize coordinates (ensure x1 < x2, y1 < y2)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Skip tiny rectangles (likely accidental clicks)
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            print("Rectangle too small, ignored.")
            return

        # Create polygon from rectangle
        polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        # Get spot name from user
        default_name = f"Spot {self.spot_counter}"
        print(f"\nEnter name for this spot (default: '{default_name}'): ", end="")
        name = input().strip() or default_name

        spot = SpotDefinition(
            id=f"spot_{self.spot_counter}",
            name=name,
            polygon=polygon,
        )

        self.spots.append(spot)
        self.spot_counter += 1

        # Reset drawing state
        self.start_point = None
        self.end_point = None

        self._update_display()
        print(f"Added: {spot.name}")

    def _undo_last_spot(self):
        """Remove the last added spot."""
        if self.spots:
            removed = self.spots.pop()
            print(f"Removed: {removed.name}")
            self._update_display()
        else:
            print("No spots to undo")

    def save_config(self, filepath: str):
        """Save spot configuration to JSON file."""
        config = {
            "spots": [asdict(s) for s in self.spots],
            "image_dimensions": {
                "width": self.original_image.shape[1],
                "height": self.original_image.shape[0],
            },
        }

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Configuration saved to: {filepath}")

    def run(self) -> list[SpotDefinition]:
        """
        Run the interactive configuration GUI.

        Returns:
            List of defined parking spots
        """
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 1280, 720)
        cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)

        print("\n" + "=" * 50)
        print("Parking Spot Configuration Tool")
        print("=" * 50)
        print("\nInstructions:")
        print("  - Click and drag to draw a rectangle for each parking spot")
        print("  - Press 'n' to name and add the current rectangle")
        print("  - Press 'u' to undo the last spot")
        print("  - Press 's' to save configuration")
        print("  - Press 'q' to quit")
        print("=" * 50)

        self._update_display()

        while True:
            cv2.imshow(self.WINDOW_NAME, self.display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("n"):
                self._add_current_spot()
            elif key == ord("u"):
                self._undo_last_spot()
            elif key == ord("s"):
                self.save_config("config/spots.json")

        cv2.destroyAllWindows()
        return self.spots


def run_configurator(
    image_path: str,
    output_path: str = "config/spots.json",
) -> list[SpotDefinition]:
    """
    Run the parking spot configurator with an image file.

    Args:
        image_path: Path to camera snapshot image
        output_path: Path to save spot configuration

    Returns:
        List of defined parking spots
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    configurator = ParkingSpotConfigurator(image)
    spots = configurator.run()

    if spots:
        configurator.save_config(output_path)

    return spots


def main():
    """CLI entry point for configuration GUI."""
    if len(sys.argv) < 2:
        print("Usage: python -m config_tool.gui <snapshot_path>")
        print("\nExample:")
        print("  python -m config_tool.gui config/reference_snapshot.jpg")
        print("\nTo capture a snapshot first, run:")
        print("  python -m config_tool.snapshot_capture")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "config/spots.json"

    try:
        spots = run_configurator(image_path, output_path)
        print(f"\nDefined {len(spots)} parking spot(s)")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
