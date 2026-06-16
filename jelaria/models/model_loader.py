"""Load YOLO detection models used by the pipeline."""

import os

from ultralytics import YOLO

from config import settings


def setup_torch_env():
    """Apply PyTorch environment flags before loading models."""
    if settings.TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


def load_vehicle_model():
    """
    Load the YOLOv8 COCO model for vehicle detection.

    Returns:
        YOLO: Pre-trained model (downloads weights on first run if needed).
    """
    setup_torch_env()
    return YOLO(settings.COCO_MODEL_PATH)


def load_license_plate_model():
    """
    Load the custom YOLO model trained for license plate detection.

    Returns:
        YOLO: Custom license plate detector.
    """
    setup_torch_env()
    return YOLO(settings.LICENSE_PLATE_MODEL_PATH)
