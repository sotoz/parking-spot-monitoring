"""Capture a snapshot from the camera for configuration."""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


async def capture_snapshot_for_config(
    host: str,
    username: str,
    password: str,
    camera_id: str,
    output_path: str = "config/reference_snapshot.jpg",
    port: int = 443,
    verify_ssl: bool = False,
) -> str:
    """
    Capture a snapshot from the camera for use in the configuration tool.

    Args:
        host: UDM Pro IP address
        username: Local access username
        password: Password
        camera_id: Camera ID or name
        output_path: Where to save the snapshot
        port: HTTPS port
        verify_ssl: Whether to verify SSL

    Returns:
        Path to saved snapshot
    """
    from parking_monitor.camera.unifi_protect import UniFiProtectClient

    print(f"Connecting to UniFi Protect at {host}...")

    client = UniFiProtectClient(
        host=host,
        username=username,
        password=password,
        port=port,
        verify_ssl=verify_ssl,
    )

    try:
        await client.connect()

        # List available cameras
        cameras = await client.list_cameras()
        print("\nAvailable cameras:")
        for cam in cameras:
            print(f"  - {cam['name']} (ID: {cam['id']})")

        print(f"\nFetching snapshot from camera '{camera_id}'...")
        snapshot = await client.get_snapshot(camera_id, width=1920, height=1080)

        # Ensure output directory exists
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "wb") as f:
            f.write(snapshot)

        print(f"Snapshot saved to: {output}")
        return str(output)

    finally:
        await client.close()


def main():
    """CLI entry point for capturing configuration snapshot."""
    print("=== Parking Spot Monitor - Snapshot Capture ===\n")

    # Get credentials from environment or prompt
    host = os.getenv("UFP_ADDRESS") or input("UDM Pro IP address: ")
    username = os.getenv("UFP_USERNAME") or input("Username: ")
    password = os.getenv("UFP_PASSWORD")

    if not password:
        import getpass
        password = getpass.getpass("Password: ")

    camera_id = input("Camera name or ID: ")

    try:
        asyncio.run(
            capture_snapshot_for_config(
                host=host,
                username=username,
                password=password,
                camera_id=camera_id,
            )
        )
        print("\nSnapshot captured successfully!")
        print("Now run: python -m config_tool.gui config/reference_snapshot.jpg")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
