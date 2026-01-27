"""Prometheus metrics for parking spot monitoring."""

from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, generate_latest

# Create a custom registry to avoid conflicts with default metrics
REGISTRY = CollectorRegistry()

# Detection confidence histogram (0.0 to 1.0 in buckets)
DETECTION_CONFIDENCE = Histogram(
    "parking_detection_confidence",
    "Confidence score of vehicle detections",
    ["spot_id", "spot_name"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=REGISTRY,
)

# Detection latency histogram (in seconds)
DETECTION_LATENCY = Histogram(
    "parking_detection_latency_seconds",
    "Time taken to process detection for all spots",
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 10.0),
    registry=REGISTRY,
)

# Spot state changes counter with time of day label
SPOT_STATE_CHANGES = Counter(
    "parking_spot_state_changes_total",
    "Total number of parking spot state changes",
    ["spot_id", "spot_name", "change_type", "hour_of_day"],
    registry=REGISTRY,
)

# Current spot status gauge
SPOT_STATUS = Gauge(
    "parking_spot_occupied",
    "Current status of parking spot (1=occupied, 0=available)",
    ["spot_id", "spot_name"],
    registry=REGISTRY,
)

# Total spots gauges
TOTAL_SPOTS = Gauge(
    "parking_spots_total",
    "Total number of parking spots",
    registry=REGISTRY,
)

AVAILABLE_SPOTS = Gauge(
    "parking_spots_available",
    "Number of available parking spots",
    registry=REGISTRY,
)

OCCUPIED_SPOTS = Gauge(
    "parking_spots_occupied",
    "Number of occupied parking spots",
    registry=REGISTRY,
)

# Detection cycles counter
DETECTION_CYCLES = Counter(
    "parking_detection_cycles_total",
    "Total number of detection cycles run",
    registry=REGISTRY,
)

# Low light enhancement counter
LOW_LIGHT_ENHANCEMENTS = Counter(
    "parking_low_light_enhancements_total",
    "Number of times low-light enhancement was applied",
    registry=REGISTRY,
)


def record_detection_confidence(spot_id: str, spot_name: str, confidence: float) -> None:
    """Record detection confidence for a spot."""
    DETECTION_CONFIDENCE.labels(spot_id=spot_id, spot_name=spot_name).observe(confidence)


def record_detection_latency(latency_seconds: float) -> None:
    """Record detection cycle latency."""
    DETECTION_LATENCY.observe(latency_seconds)


def record_spot_change(spot_id: str, spot_name: str, became_occupied: bool, hour: int) -> None:
    """Record a spot state change."""
    change_type = "became_occupied" if became_occupied else "became_available"
    SPOT_STATE_CHANGES.labels(
        spot_id=spot_id,
        spot_name=spot_name,
        change_type=change_type,
        hour_of_day=str(hour).zfill(2),
    ).inc()


def update_spot_status(spot_id: str, spot_name: str, is_occupied: bool) -> None:
    """Update current spot status gauge."""
    SPOT_STATUS.labels(spot_id=spot_id, spot_name=spot_name).set(1 if is_occupied else 0)


def update_spot_counts(total: int, available: int, occupied: int) -> None:
    """Update overall spot count gauges."""
    TOTAL_SPOTS.set(total)
    AVAILABLE_SPOTS.set(available)
    OCCUPIED_SPOTS.set(occupied)


def increment_detection_cycles() -> None:
    """Increment detection cycle counter."""
    DETECTION_CYCLES.inc()


def increment_low_light_enhancements() -> None:
    """Increment low-light enhancement counter."""
    LOW_LIGHT_ENHANCEMENTS.inc()


def get_metrics() -> bytes:
    """Generate Prometheus metrics output."""
    return generate_latest(REGISTRY)
