"""HTTP client for the local structured image-analysis API."""

import requests


ANALYSIS_ENDPOINT = "http://localhost:8000/analyze-image"


def request_image_analysis(image_bytes: bytes, image_name: str, image_type: str):
    response = requests.post(
        ANALYSIS_ENDPOINT,
        files={"file": (image_name, image_bytes, image_type)},
        timeout=(5, 120),
    )
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "The image-analysis API returned an unreadable response."
        ) from exc

    if response.status_code != 200:
        detail = (
            payload.get("detail", response.text)
            if isinstance(payload, dict)
            else response.text
        )
        raise RuntimeError(f"Image analysis failed: {detail}")
    if not isinstance(payload, dict):
        raise RuntimeError("The image-analysis API returned an unexpected response.")
    return payload
