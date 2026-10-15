"""Offline tests for the uploaded-image and structured Gemini analysis path."""

import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import os

from PIL import Image

import api
from vision_client import request_image_analysis


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

    @patch("vision_client.requests.post")
    def test_image_client_uploads_named_image_and_returns_structured_payload(self, post):
        expected = {
            "description": "A desk.",
            "objects": [],
            "summary": "No clear objects.",
            "provider": "gemini",
            "model": "gemini-3.6-flash",
        }
        response = SimpleNamespace(status_code=200, json=lambda: expected)
        post.return_value = response

        result = request_image_analysis(b"png bytes", "desk.png", "image/png")

        self.assertEqual(result, expected)
        self.assertEqual(post.call_args.args[0], "http://localhost:8000/analyze-image")
        self.assertEqual(
            post.call_args.kwargs["files"]["file"],
            ("desk.png", b"png bytes", "image/png"),
        )
        self.assertEqual(post.call_args.kwargs["timeout"], (5, 120))

    @patch("vision_client.requests.post")
    def test_image_client_surfaces_api_errors(self, post):
        response = SimpleNamespace(
            status_code=502,
            text="bad gateway",
            json=lambda: {"detail": "Gemini is temporarily unavailable."},
        )
        post.return_value = response

        with self.assertRaisesRegex(RuntimeError, "temporarily unavailable"):
            request_image_analysis(b"png bytes", "desk.png", "image/png")

if __name__ == "__main__":
    unittest.main()
