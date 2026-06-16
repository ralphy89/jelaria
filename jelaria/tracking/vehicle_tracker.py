"""Vehicle tracking using the SORT algorithm."""

import numpy as np
from sort.sort import Sort


class VehicleTracker:
    """Thin wrapper around SORT for multi-object vehicle tracking."""

    def __init__(self):
        self._tracker = Sort()

    def update(self, detections):
        """
        Update tracks with new vehicle detections for the current frame.

        Args:
            detections: List of [x1, y1, x2, y2, score] boxes.

        Returns:
            numpy.ndarray: Tracks as [x1, y1, x2, y2, track_id] rows.
        """
        if not detections:
            return np.empty((0, 5))

        return self._tracker.update(np.asarray(detections))
