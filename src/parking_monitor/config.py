"""Configuration models and loading utilities."""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, field_validator


class CameraConfig(BaseModel):
    """Camera connection configuration."""

    host: str  # UDM Pro IP address
    port: int = 443
    username: str
    password: str
    verify_ssl: bool = False
    camera_id: str  # Camera name or ID
    snapshot_width: int = 1920
    snapshot_height: int = 1080

    @field_validator("password", mode="before")
    @classmethod
    def resolve_env_var(cls, v: str) -> str:
        """Resolve environment variable references like ${VAR_NAME}."""
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            env_var = v[2:-1]
            return os.environ.get(env_var, "")
        return v


class DetectionConfig(BaseModel):
    """Detection engine configuration."""

    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    min_overlap: float = 0.3  # Minimum overlap to consider spot occupied
    interval_seconds: int = 10  # Detection interval
    hysteresis_count: int = 2  # Consecutive detections before state change


class APIConfig(BaseModel):
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000


class CalibrationConfig(BaseModel):
    """Auto-calibration configuration."""

    enabled: bool = True
    check_hour: int = 3  # Hour of day to run calibration check (0-23)
    drift_threshold_pixels: float = 20.0  # Minimum shift to trigger recalibration
    reference_image_path: str = "config/reference_calibration.jpg"


class AppConfig(BaseModel):
    """Main application configuration."""

    camera: CameraConfig
    detection: DetectionConfig = DetectionConfig()
    api: APIConfig = APIConfig()
    calibration: CalibrationConfig = CalibrationConfig()


def load_config(path: str | Path) -> AppConfig:
    """
    Load configuration from a YAML file.

    Args:
        path: Path to the configuration file

    Returns:
        Validated AppConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return AppConfig(**data)


def get_config_path() -> Path:
    """Get the default configuration file path."""
    # Check for config in current directory first
    local_config = Path("config/config.yaml")
    if local_config.exists():
        return local_config

    # Check for config in parent directory (for Docker)
    parent_config = Path("/app/config/config.yaml")
    if parent_config.exists():
        return parent_config

    return local_config  # Return default even if doesn't exist
