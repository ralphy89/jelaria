"""
Read results.csv and send unsent rows to the server in batches.

When the CSV contains more than SERVER_BATCH_MIN_ROWS data rows, pending rows
(status != SENT) are uploaded in groups of SERVER_BATCH_SIZE. After each
successful batch, those rows are marked SENT in the CSV.
"""

from datetime import datetime
from config import settings
from jelaria.output.csv_writer import mark_rows_sent, read_results_csv, _parse_bbox
from jelaria.output.json_payload import create_detection_payload
from jelaria.output.server_client import send_batch_to_server


def _row_to_payload(row):
    """Convert a CSV row dict into the standard JSON payload."""
    return create_detection_payload(
        camera_id=settings.CAMERA_ID,
        timestamp=datetime.now().isoformat(),
        frame_nmr=row["frame_nmr"],
        car_id=row["car_id"],
        car_bbox=_parse_bbox(row["car_bbox"]),
        license_plate_bbox=_parse_bbox(row["license_plate_bbox"]),
        license_plate_bbox_score=row["license_plate_bbox_score"],
        license_number=row["license_number"],
        license_number_score=row["license_number_score"],
        status=row["status"],
    )


def _get_pending_rows(rows):
    """Return rows that have not been sent yet, sorted by car_id."""
    pending = [
        row for row in rows.values()
        if row.get("status", settings.STATUS_PENDING) != settings.STATUS_SENT
    ]
    pending.sort(key=lambda r: int(float(r["car_id"])))
    return pending


def sync_csv_to_server(
    csv_path=settings.OUTPUT_CSV_PATH,
    batch_size=settings.SERVER_BATCH_SIZE,
    min_rows=settings.SERVER_BATCH_MIN_ROWS,
    server_url=settings.SERVER_URL,
):
    """
    Upload unsent CSV rows in batches when total row count exceeds min_rows.

    Args:
        csv_path: Path to results.csv (defaults to config).
        batch_size: Rows per HTTP request (defaults to config).
        min_rows: Minimum total rows required to start batch upload.
        server_url: Override backend URL.

    Returns:
        dict: Summary with keys sent, failed, skipped.
    """
    csv_path = csv_path or settings.OUTPUT_CSV_PATH
    batch_size = batch_size if batch_size is not None else settings.SERVER_BATCH_SIZE
    min_rows = min_rows if min_rows is not None else settings.SERVER_BATCH_MIN_ROWS

    rows = read_results_csv(csv_path)
    total_rows = len(rows)

    if total_rows <= min_rows:
        print(
            f"CSV has {total_rows} row(s) — batch upload requires more than {min_rows}. Skipping."
        )
        return {"sent": 0, "failed": 0, "skipped": total_rows}

    pending = _get_pending_rows(rows)
    if not pending:
        print("All rows already marked SENT. Nothing to upload.")
        return {"sent": 0, "failed": 0, "skipped": total_rows}

    print(
        f"Uploading {len(pending)} pending row(s) from {total_rows} total "
        f"in batches of {batch_size}..."
    )

    sent_count = 0
    failed_count = 0

    for start in range(0, len(pending), batch_size):
        batch_rows = pending[start : start + batch_size]
        payloads = [_row_to_payload(row) for row in batch_rows]

        if send_batch_to_server(payloads, server_url=server_url):
            car_ids = [int(float(row["car_id"])) for row in batch_rows]
            mark_rows_sent(csv_path, car_ids)
            sent_count += len(batch_rows)
        else:
            failed_count += len(batch_rows)
            # Stop on failure so unsent rows remain for the next attempt
            break

    print(f"Batch sync done: {sent_count} sent, {failed_count} failed.")
    return {"sent": sent_count, "failed": failed_count, "skipped": total_rows - len(pending)}


if __name__ == "__main__":
    sync_csv_to_server()
