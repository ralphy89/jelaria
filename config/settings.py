"""
Central configuration for the JeLarIA license plate recognition module.

Edit these values to adapt the pipeline to your camera, models, or deployment
environment (e.g. NVIDIA Jetson Orin Nano).
"""

import os

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Video source: path to a file, or use 0 for the default webcam
VIDEO_SOURCE = os.path.join(PROJECT_ROOT, "sample.mp4")
# VIDEO_SOURCE = "rtsp://192.168.43.110/stream"
# YOLO model weights
COCO_MODEL_PATH = "yolov8n.pt"
LICENSE_PLATE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "license_plate_detector.pt")

# Output CSV file (written progressively during processing)
OUTPUT_CSV_PATH = os.path.join(PROJECT_ROOT, "output", "results.csv")

# ---------------------------------------------------------------------------
# Camera / backend
# ---------------------------------------------------------------------------
CAMERA_ID = "1"
NODE_ID = "JETSON_NANO_NODE_1"
SERVER_URL = f"https://tracage-vehicule.onrender.com/api/detections/batch/"
SEND_TO_SERVER = True  # Set True to POST results to the backend
SERVER_TIMEOUT_SECONDS = 3

# Batch upload: when CSV has more than MIN_ROWS, send unsent rows in batches
SERVER_BATCH_SIZE = 2
SERVER_BATCH_MIN_ROWS = 2
STATUS_PENDING = "PLATE_RECOGNIZED"
STATUS_SENT = "SENT"

# ---------------------------------------------------------------------------
# Detection settings
# ---------------------------------------------------------------------------
# COCO class IDs for vehicles: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASS_IDS = [2, 3, 5, 7]

# ---------------------------------------------------------------------------
# OCR and quality thresholds
# ---------------------------------------------------------------------------
OCR_CONFIDENCE_THRESHOLD = 0.70

# EasyOCR: set gpu=True on Jetson Orin Nano when CUDA is available
OCR_USE_GPU = False
OCR_LANGUAGES = ["en"]

# ---------------------------------------------------------------------------
# PyTorch / Jetson compatibility
# ---------------------------------------------------------------------------
# Required on some PyTorch versions when loading YOLO weights
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = True
