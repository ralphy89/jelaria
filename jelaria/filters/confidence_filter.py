"""Filter OCR results by confidence and avoid duplicate saves per vehicle."""

from config import settings


class ConfidenceFilter:
    """
    Track the best OCR result per car_id.

    A detection is accepted when:
        - OCR confidence >= threshold (default 0.70)
        - The car_id has never been saved, OR the new score is higher
    """

    def __init__(self, threshold=None):
        self.threshold = threshold if threshold is not None else settings.OCR_CONFIDENCE_THRESHOLD
        self._saved_cars = {}

    def should_save(self, car_id, ocr_score):
        """
        Decide whether this detection should be written to CSV / sent to server.

        Args:
            car_id: SORT track ID of the vehicle.
            ocr_score: EasyOCR confidence for the plate text.

        Returns:
            bool: True if the result should be saved.
        """
        if ocr_score is None or ocr_score < self.threshold:
            return False

        if car_id not in self._saved_cars:
            return True

        return ocr_score > self._saved_cars[car_id]["score"]

    def register_save(self, car_id, plate_text, ocr_score):
        """Record the best result for a car_id after a successful save."""
        self._saved_cars[car_id] = {"plate": plate_text, "score": ocr_score}

    @property
    def saved_cars(self):
        """Read-only view of saved car records."""
        return dict(self._saved_cars)
