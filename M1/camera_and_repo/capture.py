#!/usr/bin/env python3
"""
Camera Capture Utility
Captures an initial frame from the active camera (or synthetic fallback)
and saves it to the evidence and data directories.
"""

import os
import sys
import logging

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.camera.camera_manager import CameraManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CaptureUtility")


def main():
    logger.info("Initializing Camera Capture Utility...")
    cam = CameraManager(device_index=0, width=640, height=480, auto_fallback_simulation=True)

    destinations = [
        "data/test/initial_capture.jpg",
        "evidence/photographs/camera_setup/initial_capture.jpg",
    ]

    for dest in destinations:
        success = cam.save_snapshot(dest)
        if success:
            logger.info(f"Frame saved to: {dest}")
        else:
            logger.error(f"Failed to save frame to: {dest}")

    cam.release()
    logger.info("Capture complete.")


if __name__ == "__main__":
    main()
