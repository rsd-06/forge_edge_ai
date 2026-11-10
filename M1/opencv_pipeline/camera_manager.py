"""
Camera Manager module for Edge AI Vision System.
Handles hardware USB webcam acquisition via V4L2 with an automated
synthetic frame generator fallback when physical hardware is not detected.
"""

import os
import time
import logging
from typing import Tuple, Optional
import cv2
import numpy as np

logger = logging.getLogger("CameraManager")


class CameraManager:
    """
    Manages camera acquisition, resolution configuration, proper shutdown,
    and automatic simulation fallback.
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        auto_fallback_simulation: bool = True,
        test_image_dir: str = "data/test",
    ):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.auto_fallback_simulation = auto_fallback_simulation
        self.test_image_dir = test_image_dir

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_simulation = False
        self.is_opened = False
        self.frame_count = 0
        self.start_time = time.time()

        self._initialize_camera()

    def _initialize_camera(self) -> None:
        """Attempts to open physical hardware camera; falls back to simulation if needed."""
        logger.info(f"Probing video device index: {self.device_index}")
        
        # Try opening physical camera
        self.cap = cv2.VideoCapture(self.device_index)
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Verify actual capture
            ret, test_frame = self.cap.read()
            if ret and test_frame is not None and test_frame.shape[0] > 0:
                self.is_opened = True
                self.is_simulation = False
                logger.info(
                    f"Physical camera initialized successfully on /dev/video{self.device_index} "
                    f"({self.width}x{self.height} @ {self.fps} FPS)"
                )
                return
            else:
                logger.warning("Camera device opened but failed to read initial frame.")
                self.cap.release()
                self.cap = None

        # Camera failed or absent
        if self.auto_fallback_simulation:
            logger.warning(
                f"[SIMULATION FALLBACK] No physical camera recognized on /dev/video{self.device_index}. "
                "Engaging synthetic industrial simulation stream."
            )
            self.is_simulation = True
            self.is_opened = True
        else:
            self.is_opened = False
            self.is_simulation = False
            logger.error(f"Failed to open camera on /dev/video{self.device_index} and fallback disabled.")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads a frame from either physical webcam or synthetic generator.
        Returns: (success: bool, frame: np.ndarray)
        """
        if not self.is_opened:
            return False, None

        if not self.is_simulation and self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.frame_count += 1
                return True, frame
            else:
                logger.warning("Physical camera read failed. Falling back to simulation.")
                self.is_simulation = True

        # Generate synthetic frame
        self.frame_count += 1
        frame = self._generate_synthetic_frame()
        return True, frame

    def _generate_synthetic_frame(self) -> np.ndarray:
        """
        Generates an industrial inspection frame featuring a simulated conveyor belt
        and metallic industrial enclosure under inspection.
        """
        # Create base conveyor background (dark industrial grey with belt texture)
        img = np.full((self.height, self.width, 3), 45, dtype=np.uint8)

        # Draw conveyor belt surface (middle 60% of vertical space)
        belt_top = int(self.height * 0.20)
        belt_bottom = int(self.height * 0.85)
        cv2.rectangle(img, (0, belt_top), (self.width, belt_bottom), (65, 65, 65), -1)

        # Draw conveyor guide rails
        cv2.line(img, (0, belt_top), (self.width, belt_top), (160, 160, 160), 4)
        cv2.line(img, (0, belt_bottom), (self.width, belt_bottom), (160, 160, 160), 4)

        # Simulate object motion across conveyor (moves horizontally over 120 frames)
        cycle_len = 120
        phase = (self.frame_count % cycle_len) / cycle_len
        box_w, box_h = 240, 180
        center_x = int(50 + phase * (self.width - 100))
        center_y = int((belt_top + belt_bottom) / 2)

        x1 = max(10, center_x - box_w // 2)
        y1 = max(belt_top + 10, center_y - box_h // 2)
        x2 = min(self.width - 10, x1 + box_w)
        y2 = min(belt_bottom - 10, y1 + box_h)

        # Draw simulated metal enclosure (chamfered box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (180, 185, 190), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (230, 235, 240), 2)  # border

        # Inner cavity / mounting holes
        cv2.rectangle(img, (x1 + 25, y1 + 25), (x2 - 25, y2 - 25), (130, 135, 140), -1)
        for hx, hy in [
            (x1 + 15, y1 + 15),
            (x2 - 15, y1 + 15),
            (x1 + 15, y2 - 15),
            (x2 - 15, y2 - 15),
        ]:
            cv2.circle(img, (hx, hy), 4, (40, 40, 40), -1)

        # Text banner indicating SIMULATION MODE
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, 32), (0, 0, 140), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
        status_text = (
            f"[SIMULATION STREAM - WEBCAM ABSENT] Frame: {self.frame_count:05d} | {timestamp_str}"
        )
        cv2.putText(
            img,
            status_text,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return img

    def save_snapshot(self, output_path: str) -> bool:
        """Captures a single frame and writes it to disk."""
        ret, frame = self.read()
        if ret and frame is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            success = cv2.imwrite(output_path, frame)
            if success:
                logger.info(f"Snapshot written successfully to {output_path}")
            return success
        logger.error("Failed to capture snapshot.")
        return False

    def release(self) -> None:
        """Releases physical camera resource and cleans up."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_opened = False
        logger.info("Camera manager released cleanly.")
