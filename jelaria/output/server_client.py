"""Optional HTTP POST to a backend server."""

import requests

from config import settings


def send_to_server(payload, server_url=None, timeout=None):
    """
    Send a single detection payload to the backend server.

    Never raises: if the server is down or returns an error, the pipeline
    continues normally.

    Returns:
        bool: True on HTTP 200/201, False otherwise.
    """
    url = server_url or settings.SERVER_URL
    timeout = timeout if timeout is not None else settings.SERVER_TIMEOUT_SECONDS

    try:
        response = requests.post(url, json=payload, timeout=timeout)

        if response.status_code in (200, 201):
            print("Result sent to server successfully.")
            return True

        print(f"Server error: HTTP {response.status_code}")
        return False

    except requests.exceptions.RequestException as exc:
        print(f"Server unavailable: {exc}")
        return False


def send_batch_to_server(payloads, server_url=None, timeout=None):
    """
    Send a batch of detection payloads in one HTTP request.

    Body format: {"detections": [payload1, payload2, ...]}

    Never raises. Returns False if the server is unavailable or rejects the batch.
    """
    if not payloads:
        return True

    url = server_url or settings.SERVER_URL
    timeout = timeout if timeout is not None else settings.SERVER_TIMEOUT_SECONDS

    try:
        response = requests.post(
            url,
            json={"detections": payloads},
            timeout=timeout,
        )

        if response.status_code in (200, 201):
            car_ids = [p["car_id"] for p in payloads]
            print(f"Batch sent successfully ({len(payloads)} row(s), car_ids={car_ids}).")
            return True

        print(f"Server error: HTTP {response.status_code}")
        return False

    except requests.exceptions.RequestException as exc:
        print(f"Server unavailable: {exc}")
        return False
