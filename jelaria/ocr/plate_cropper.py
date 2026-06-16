"""Crop and preprocess license plate regions for OCR."""

import cv2


def crop_plate(frame, x1, y1, x2, y2):
    """
    Extract the license plate region from the full frame.

    Args:
        frame: Full BGR image.
        x1, y1, x2, y2: Plate bounding box coordinates.

    Returns:
        numpy.ndarray: Cropped plate image (BGR).
    """
    return frame[int(y1):int(y2), int(x1):int(x2), :]


def preprocess_plate(plate_crop):
    """
    Convert the plate crop to a binary image suitable for EasyOCR.

    Steps:
        1. Grayscale conversion
        2. Binary threshold (inverted) to highlight characters

    Args:
        plate_crop: BGR cropped plate image.

    Returns:
        numpy.ndarray: Preprocessed single-channel image.
    """
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)
    return thresh
