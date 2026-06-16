"""License plate detection module."""

def detect_license_plates(model, frame):
    """
    Run the custom YOLO model on a frame to find license plates.

    Args:
        model: Loaded license plate YOLO model.
        frame: BGR image (numpy array).

    Returns:
        list: Each item is [x1, y1, x2, y2, confidence_score, class_id].
    """
    results = model(frame)[0]
    plates = []

    for detection in results.boxes.data.tolist():
        plates.append(detection)

    return plates
