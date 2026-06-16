"""Build JSON payloads for backend communication."""


def create_detection_payload(
    camera_id,
    timestamp,
    frame_nmr,
    car_id,
    car_bbox,
    license_plate_bbox,
    license_plate_bbox_score,
    license_number,
    license_number_score,
    status="PLATE_RECOGNIZED",
):
    """
    Build the standard JSON payload for one recognized plate.

    Example output:
        {
            "frame_nmr": 25,
            "car_id": 3,
            "car_bbox": [100, 200, 400, 500],
            "license_plate_bbox": [180, 430, 300, 470],
            "license_plate_bbox_score": 0.89,
            "license_number": "AA12345",
            "license_number_score": 0.82,
            "status": "PLATE_RECOGNIZED"
        }
    """
    return {
        "camera_id": camera_id,
        "timestamp": timestamp,
        "frame_nmr": int(frame_nmr),
        "car_id": int(car_id),
        "car_bbox": [float(v) for v in car_bbox],
        "license_plate_bbox": [float(v) for v in license_plate_bbox],
        "license_plate_bbox_score": float(license_plate_bbox_score),
        "license_number": license_number,
        "license_number_score": float(license_number_score),
        "status": status,
    }
