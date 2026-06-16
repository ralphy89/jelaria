"""Vehicle detection module."""

from config import settings


def detect_vehicles(model, frame, class_ids=None):
    """
    Run YOLO on a frame and return bounding boxes for vehicle classes only.

    Args:
        model: Loaded YOLO COCO model.
        frame: BGR image (numpy array).
        class_ids: List of COCO class IDs to keep. Defaults to config value.

    Returns:
        list: Each item is [x1, y1, x2, y2, confidence_score].
    """
    if class_ids is None:
        class_ids = settings.VEHICLE_CLASS_IDS

    detections = model(frame)[0]
    vehicle_boxes = []

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in class_ids:
            vehicle_boxes.append([x1, y1, x2, y2, score])

    return vehicle_boxes
