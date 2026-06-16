"""Read license plate text using EasyOCR."""

import easyocr

from config import settings
from jelaria.ocr.plate_validator import format_license, license_complies_format

# Single shared reader instance (loaded once at import time)
_reader = None


def _get_reader():
    """Lazy-load the EasyOCR reader to avoid reloading on every call."""
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(settings.OCR_LANGUAGES, gpu=settings.OCR_USE_GPU)
    return _reader


def read_license_plate(plate_image):
    """
    Run OCR on a preprocessed plate image and return the best valid reading.

    Args:
        plate_image: Preprocessed plate image (grayscale/binary).

    Returns:
        tuple: (text, score) or (None, None) if OCR fails or format is invalid.
    """
    reader = _get_reader()
    detections = reader.readtext(plate_image)

    best_text = None
    best_score = 0.0

    for _bbox, text, score in detections:
        cleaned = format_license(text)
        if license_complies_format(cleaned) and score > best_score:
            best_text = cleaned
            best_score = score

    if best_text is not None:
        return best_text, best_score

    return None, None
