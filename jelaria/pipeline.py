"""
Main video processing pipeline for JeLarIA.

Orchestrates: vehicle detection → tracking → plate detection → OCR → output.
"""

from datetime import datetime
import cv2

from config import settings
from jelaria.association.plate_vehicle import assign_plate_to_vehicle
from jelaria.detection.plate_detector import detect_license_plates
from jelaria.detection.vehicle_detector import detect_vehicles
from jelaria.filters.confidence_filter import ConfidenceFilter
from jelaria.models.model_loader import load_license_plate_model, load_vehicle_model
from jelaria.ocr.plate_cropper import crop_plate, preprocess_plate
from jelaria.ocr.plate_reader import read_license_plate
from jelaria.output.batch_sync import sync_csv_to_server
from jelaria.output.csv_writer import ResultsCsvWriter
from jelaria.output.json_payload import create_detection_payload
from jelaria.tracking.vehicle_tracker import VehicleTracker


def run_pipeline(
    video_source=None,
    output_csv=None,
    send_results=None,
):
    """
    Process a video file or camera stream frame by frame.

    Args:
        video_source: Path to video or camera index (0 = webcam). Uses config default.
        output_csv: Path to output CSV. Uses config default.
        send_results: Whether to POST to backend. Uses config default if None.
    """
    video_source = video_source if video_source is not None else settings.VIDEO_SOURCE
    output_csv = output_csv if output_csv is not None else settings.OUTPUT_CSV_PATH
    send_results = settings.SEND_TO_SERVER if send_results is None else send_results

    # --- Step 1: Load models ---
    print("Loading models...")
    vehicle_model = load_vehicle_model()
    plate_model = load_license_plate_model()

    # --- Step 2: Open video source ---
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {video_source}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracker = VehicleTracker()
    confidence_filter = ConfidenceFilter()

    # --- Step 3: Prepare CSV output (one row per car_id, updated progressively) ---
    csv_writer = ResultsCsvWriter(output_csv)
    print(f"Results will be saved to: {output_csv}")

    frame_nmr = -1

    # --- Step 4: Process each frame ---
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_nmr += 1
        if frame_count > 0 and frame_nmr >= frame_count:
            break

        print(f"Processing frame {frame_nmr}/{frame_count if frame_count > 0 else '?'}")

        # --- Step 5: Detect vehicles ---
        vehicle_boxes = detect_vehicles(vehicle_model, frame)

        # No vehicles → skip this frame entirely
        if not vehicle_boxes:
            continue

        # --- Step 6: Track vehicles with SORT ---
        track_ids = tracker.update(vehicle_boxes)

        if len(track_ids) == 0:
            continue

        # --- Step 7: Detect license plates ---
        plate_detections = detect_license_plates(plate_model, frame)

        for plate in plate_detections:
            x1, y1, x2, y2, plate_score, _class_id = plate

            # --- Step 8: Associate plate with a tracked vehicle ---
            xcar1, ycar1, xcar2, ycar2, car_id = assign_plate_to_vehicle(plate, track_ids)

            # Plate not inside any vehicle → ignore
            if car_id == -1:
                continue

            # --- Step 9: Crop and preprocess the plate region ---
            plate_crop = crop_plate(frame, x1, y1, x2, y2)
            if plate_crop.size == 0:
                continue

            plate_preprocessed = preprocess_plate(plate_crop)

            # --- Step 10: OCR reading ---
            plate_text, ocr_score = read_license_plate(plate_preprocessed)

            # OCR failed → do not save
            if plate_text is None or ocr_score is None:
                continue

            # --- Step 11: Confidence filter + duplicate check ---
            if not confidence_filter.should_save(car_id, ocr_score):
                continue

            car_bbox = [xcar1, ycar1, xcar2, ycar2]
            plate_bbox = [x1, y1, x2, y2]
            status = settings.STATUS_PENDING

            # --- Step 12: Upsert CSV row (replace if same car_id with better score) ---
            csv_writer.upsert(
                frame_nmr,
                car_id,
                car_bbox,
                plate_bbox,
                plate_score,
                plate_text,
                ocr_score,
                status,
            )

            confidence_filter.register_save(car_id, plate_text, ocr_score)

            # --- Step 13: Build JSON payload ---
            payload = create_detection_payload(
                settings.CAMERA_ID,
                datetime.now().isoformat(),
                frame_nmr,
                car_id,
                car_bbox,
                plate_bbox,
                plate_score,
                plate_text,
                ocr_score,
                status,
            )

            print(f"Plate saved: {payload}")

    cap.release()
    print(f"Processing complete. {csv_writer.row_count} unique vehicle(s) recognized.")
    print(f"CSV saved to: {output_csv}")

    # --- Step 14: Batch upload unsent rows when CSV exceeds threshold ---
    if send_results:
        sync_csv_to_server(output_csv)
