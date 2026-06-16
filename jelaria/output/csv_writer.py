"""Progressive CSV output — one row per car_id, best OCR score wins."""

import ast
import csv
import os

from config import settings


CSV_COLUMNS = [
    "frame_nmr",
    "car_id",
    "car_bbox",
    "license_plate_bbox",
    "license_plate_bbox_score",
    "license_number",
    "license_number_score",
    "status",
]


def _format_bbox(bbox):
    """Convert bbox values to plain floats for clean CSV output."""
    return [float(v) for v in bbox]


def _parse_bbox(value):
    """Parse a bbox stored as a list string or list."""
    if isinstance(value, list):
        return [float(v) for v in value]
    return [float(v) for v in ast.literal_eval(value)]


def read_results_csv(output_path):
    """
    Read all rows from the results CSV.

    Returns:
        dict: car_id (int) -> row dict keyed by column name.
    """
    rows = {}
    if not os.path.exists(output_path):
        return rows

    with open(output_path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            car_id = int(float(row["car_id"]))
            rows[car_id] = row

    return rows


def write_results_csv(output_path, rows):
    """Write all rows to CSV, sorted by car_id."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_COLUMNS)
        for car_id in sorted(rows.keys()):
            row = rows[car_id]
            writer.writerow([row[col] for col in CSV_COLUMNS])


def mark_rows_sent(output_path, car_ids):
    """Update status to SENT for the given car_ids and rewrite the CSV."""
    rows = read_results_csv(output_path)
    for car_id in car_ids:
        car_id = int(car_id)
        if car_id in rows:
            rows[car_id]["status"] = settings.STATUS_SENT
    write_results_csv(output_path, rows)


class ResultsCsvWriter:
    """
    Maintain a CSV with exactly one row per car_id.

    When a better OCR score is found for the same vehicle, the existing row
    is replaced (not duplicated). The file is rewritten after each update so
    results are saved progressively during processing.
    """

    def __init__(self, output_path):
        self.output_path = output_path
        self._rows = {}  # car_id (int) -> row values list
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self._load_existing()
        self._flush()

    def _load_existing(self):
        """Load existing CSV so SENT status is preserved between runs."""
        if not os.path.exists(self.output_path):
            return

        with open(self.output_path, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader, None)  # skip header
            for row in reader:
                if row:
                    car_id = int(float(row[1]))
                    self._rows[car_id] = row

    def upsert(
        self,
        frame_nmr,
        car_id,
        car_bbox,
        license_plate_bbox,
        license_plate_bbox_score,
        license_number,
        license_number_score,
        status=None,
    ):
        """
        Insert or replace the row for this car_id, then rewrite the CSV.

        Only the highest-scoring detection should call this (enforced upstream
        by ConfidenceFilter). Resets status to PLATE_RECOGNIZED on update.
        """
        if status is None:
            status = settings.STATUS_PENDING

        car_id = int(car_id)
        self._rows[car_id] = [
            int(frame_nmr),
            car_id,
            _format_bbox(car_bbox),
            _format_bbox(license_plate_bbox),
            float(license_plate_bbox_score),
            license_number,
            float(license_number_score),
            status,
        ]
        self._flush()

    def mark_sent(self, car_ids):
        """Mark rows as SENT so they are skipped by future batch uploads."""
        for car_id in car_ids:
            car_id = int(car_id)
            if car_id in self._rows:
                self._rows[car_id][7] = settings.STATUS_SENT
        self._flush()

    def _flush(self):
        """Write all rows to disk (sorted by car_id for stable output)."""
        with open(self.output_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(CSV_COLUMNS)
            for car_id in sorted(self._rows.keys()):
                writer.writerow(self._rows[car_id])

    @property
    def row_count(self):
        """Number of unique vehicles saved."""
        return len(self._rows)


def init_csv(output_path):
    """Create an empty CSV file with column headers only."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_COLUMNS)
