"""Offline tests for the uploaded-image and structured Gemini analysis path."""

import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import os

from PIL import Image

import api


def png_bytes():
    image = Image.new("RGB", (48, 32), color=(220, 210, 190))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class ImageAnalysisTests(unittest.TestCase):
    def test_gemini_response_is_parsed_into_structured_objects(self):
        generated = {
            "description": "A study desk with a laptop.",
            "objects": [
                {
                    "name": "laptop",
                    "position": "upper left",
                    "visual_notes": "Open screen and keyboard are visible.",
                    "certainty": "clear",
                }
            ],
            "summary": "One clearly visible laptop.",
        }
        calls = []

        class FakeModels:
            def generate_content(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(text=json.dumps(generated))

        fake_client = SimpleNamespace(models=FakeModels())
        result = api.analyze_image_with_llm(
            png_bytes(),
            api_key="test-key",
            client_factory=lambda key: fake_client,
        )

        self.assertEqual(result["provider"], "gemini")
        self.assertEqual(result["objects"][0]["name"], "laptop")
        self.assertEqual(result["objects"][0]["position"], "upper left")
        self.assertEqual(calls[0]["config"].response_mime_type, "application/json")
        self.assertEqual(len(calls), 1)

    def test_unavailable_primary_model_uses_current_fallback(self):
        generated = {
            "description": "A desk.",
            "objects": [],
            "summary": "No identifiable personal items.",
        }
        models = []

        class FakeModels:
            def generate_content(self, **kwargs):
                models.append(kwargs["model"])
                if len(models) < 3:
                    raise RuntimeError("temporary model availability issue")
                return SimpleNamespace(text=json.dumps(generated))

        fake_client = SimpleNamespace(models=FakeModels())
        with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-3.8-flash"}):
            result = api.analyze_image_with_llm(
                png_bytes(),
                api_key="test-key",
                client_factory=lambda key: fake_client,
            )

        self.assertEqual(
            models,
            ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"],
        )
        self.assertEqual(result["model"], "gemini-3.5-flash-lite")

    def test_missing_key_is_reported_without_fabricated_objects(self):
        with patch.object(api, "_configured_api_key", return_value=None):
            result = api.analyze_image_with_llm(png_bytes(), api_key="")

        self.assertEqual(result["provider"], "unavailable")
        self.assertEqual(result["objects"], [])
        self.assertIn("LLM_API_KEY", result["summary"])

    def test_invalid_image_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not a readable image"):
            api._open_image(b"not an image")

    def test_oversized_image_is_rejected_before_decode(self):
        oversized = b"0" * (api.MAX_IMAGE_BYTES + 1)
        with self.assertRaisesRegex(ValueError, "under 10 MB"):
            api._open_image(oversized)


if __name__ == "__main__":
    unittest.main()
