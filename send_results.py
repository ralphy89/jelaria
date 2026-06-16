"""
Upload unsent rows from output/results.csv to the backend in batches.

Usage:
    python send_results.py
"""

from jelaria.output.batch_sync import sync_csv_to_server


if __name__ == "__main__":
    sync_csv_to_server()
