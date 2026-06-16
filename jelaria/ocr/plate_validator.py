"""Validate and clean OCR text output."""

import re


def format_license(text):
    """
    Clean raw OCR text: uppercase, remove spaces and punctuation.

    Args:
        text: Raw string from EasyOCR.

    Returns:
        str or None: Cleaned text, or None if input is None.
    """
    if text is None:
        return None

    text = text.upper()
    for char in (" ", ".", ",", "_", "-"):
        text = text.replace(char, "")

    return text


def license_complies_format(text):
    """
    Check whether the text looks like a valid license plate.

    Rules (prototype-friendly):
        - Only letters and digits
        - Length between 5 and 10 characters

    Args:
        text: Cleaned plate string.

    Returns:
        bool: True if the format is acceptable.
    """
    if text is None:
        return False

    text = text.upper().replace(" ", "").replace("-", "")

    if not re.match(r"^[A-Z0-9]+$", text):
        return False

    if len(text) < 5 or len(text) > 10:
        return False

    return True
