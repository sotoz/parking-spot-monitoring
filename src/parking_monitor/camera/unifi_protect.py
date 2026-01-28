"""UniFi Protect API client for camera snapshot access."""

import io
import logging
from typing import Optional

import cv2
import numpy as np
from uiprotect import ProtectApiClient
from uiprotect.data import Camera

logger = logging.getLogger(__name__)


class UniFiProtectClient:
    """
    Wrapper for UniFi Protect API to fetch camera snapshots.

    This client connects to UniFi Protect running on a UDM Pro
    and provides methods to capture snapshots from cameras.
    """

    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        port: int = 443,
        verify_ssl: bool = False,
    ):
        """
        Initialize the UniFi Protect client.

        Args:
            host: IP address or hostname of the UDM Pro
            username: Local access username (not Ubiquiti SSO)
            password: Password for the local user
            port: HTTPS port (default 443)
            verify_ssl: Whether to verify SSL certificate
        """
        self.host = host
        self.username = username
        self.password = password
        self.port = port
        self.verify_ssl = verify_ssl
        self._client: Optional[ProtectApiClient] = None
        self._camera_cache: dict[str, Camera] = {}
        self._connected = False

    async def connect(self) -> None:
        """
        Initialize connection to UniFi Protect.

        This must be called before any other methods.
        """
        logger.info(f"Connecting to UniFi Protect at {self.host}:{self.port}")

        self._client = ProtectApiClient(
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            verify_ssl=self.verify_ssl,
        )

        await self._client.update()

        # Cache camera objects by both ID and name for easy lookup
        for camera in self._client.bootstrap.cameras.values():
            self._camera_cache[camera.id] = camera
            self._camera_cache[camera.name.lower()] = camera
            logger.debug(f"Found camera: {camera.name} (ID: {camera.id})")

        self._connected = True
        logger.info(
            f"Connected to UniFi Protect. Found {len(self._client.bootstrap.cameras)} camera(s)"
        )

    async def get_snapshot(
        self,
        camera_id: str,
        width: int = 1920,
        height: int = 1080,
    ) -> bytes:
        """
        Fetch a snapshot from the specified camera.

        Args:
            camera_id: Camera ID or name (case-insensitive)
            width: Desired image width
            height: Desired image height

        Returns:
            JPEG image bytes

        Raises:
            ValueError: If camera is not found
            RuntimeError: If not connected
        """
        if not self._connected or self._client is None:
            await self.connect()

        # Try exact match first, then case-insensitive name match
        camera = self._camera_cache.get(camera_id)
        if camera is None:
            camera = self._camera_cache.get(camera_id.lower())

        if camera is None:
            available = [c.name for c in self._client.bootstrap.cameras.values()]
            raise ValueError(
                f"Camera '{camera_id}' not found. Available cameras: {available}"
            )

        # Try to get high-quality snapshot from RTSP stream (channel 0)
        # The standard get_snapshot() returns low-quality preview
        if hasattr(camera, 'channels') and camera.channels and len(camera.channels) > 0:
            high_channel = camera.channels[0]  # Channel 0 is usually highest quality
            if hasattr(high_channel, 'is_rtsp_enabled') and high_channel.is_rtsp_enabled:
                logger.debug(
                    f"Using RTSP stream for snapshot: {high_channel.width}x{high_channel.height}"
                )
                try:
                    # Get RTSP URL - uiprotect should provide this
                    rtsp_url = None
                    if hasattr(camera, 'rtsp_url'):
                        rtsp_url = camera.rtsp_url
                    elif hasattr(high_channel, 'rtsp_url'):
                        rtsp_url = high_channel.rtsp_url
                    elif hasattr(camera, 'get_rtsps_url'):
                        rtsp_url = camera.get_rtsps_url()

                    if rtsp_url:
                        # Grab a frame from RTSP using OpenCV with retry logic
                        # HEVC streams can have codec errors when catching mid-sequence
                        import time

                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                cap = cv2.VideoCapture(rtsp_url)
                                if not cap.isOpened():
                                    logger.warning(f"Failed to open RTSP stream (attempt {attempt + 1}/{max_retries})")
                                    if attempt < max_retries - 1:
                                        time.sleep(0.5)
                                    continue

                                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering

                                # Skip first few frames to get past codec initialization
                                for _ in range(3):
                                    cap.grab()

                                ret, frame = cap.read()
                                cap.release()

                                if ret and frame is not None:
                                    # Resize if needed
                                    if width and height:
                                        frame = cv2.resize(frame, (width, height))

                                    # Encode to JPEG
                                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                                    logger.debug(f"Captured RTSP frame: {len(buffer)} bytes (attempt {attempt + 1})")
                                    return buffer.tobytes()
                                else:
                                    logger.debug(f"Failed to read frame (attempt {attempt + 1}/{max_retries})")
                                    if attempt < max_retries - 1:
                                        time.sleep(0.5)

                            except Exception as e:
                                logger.debug(f"RTSP capture error (attempt {attempt + 1}/{max_retries}): {e}")
                                if attempt < max_retries - 1:
                                    time.sleep(0.5)
                                continue

                        logger.warning("Failed to capture frame from RTSP after retries, falling back to API")

                except Exception as e:
                    logger.warning(f"RTSP capture failed: {e}, falling back to API snapshot")

        # Fallback to standard API snapshot (low quality)
        logger.debug(f"Fetching API snapshot from camera '{camera.name}' at {width}x{height}")
        snapshot = await camera.get_snapshot(width=width, height=height)

        if snapshot:
            logger.debug(f"Received snapshot: {len(snapshot)} bytes")

        return snapshot

    async def list_cameras(self) -> list[dict]:
        """
        List all available cameras.

        Returns:
            List of camera info dictionaries
        """
        if not self._connected or self._client is None:
            await self.connect()

        return [
            {
                "id": cam.id,
                "name": cam.name,
                "type": cam.type,
                "is_connected": cam.is_connected,
            }
            for cam in self._client.bootstrap.cameras.values()
        ]

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected

    async def close(self) -> None:
        """Close the API connection."""
        if self._client:
            # Try different close methods depending on uiprotect version
            if hasattr(self._client, "async_disconnect"):
                await self._client.async_disconnect()
            elif hasattr(self._client, "close"):
                await self._client.close()
            self._connected = False
            logger.info("Disconnected from UniFi Protect")

    async def __aenter__(self) -> "UniFiProtectClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
