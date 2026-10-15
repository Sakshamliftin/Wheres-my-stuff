"""
api.py — FastAPI Backend

Endpoints:
  POST /analyze-image    — Return structured visible-object analysis for an image
  POST /query            — Natural-language "where is my …" query against the DB
  GET  /zones            — Retrieve current zone configuration
  POST /zones            — Update zone polygon configuration
  GET  /classes          — List distinct object classes tracked in the DB
  GET  /summary          — Observation summary per class
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
from typing import Literal
from pydantic import BaseModel, Field, ValidationError
import io
from PIL import Image, ImageOps, UnidentifiedImageError

from query_engine import answer_query
from zones import ZoneManager
from db import get_distinct_tracked_classes, get_observation_summary, init_db

app = FastAPI(title="Where Is My Stuff? — API", version="2.0")


# ── Models ───────────────────────────────────────────────────────────────────
class VisibleObject(BaseModel):
    name: str = Field(description="The visible item's short, common name")
    position: Literal[
        "upper left", "upper center", "upper right",
        "middle left", "center", "middle right",
        "lower left", "lower center", "lower right", "unclear",
    ] = Field(description="Approximate position within the image")
    visual_notes: str = Field(
        description="Brief visible details that support the identification"
    )
    certainty: Literal["clear", "partial", "uncertain"] = Field(
        description="How clearly the item can be identified from this image"
    )


class GeminiImageAnalysis(BaseModel):
    description: str
    objects: list[VisibleObject]
    summary: str


class AnalysisResult(GeminiImageAnalysis):
    provider: Literal["gemini", "unavailable"] = "gemini"
    model: str | None = None


class QueryRequest(BaseModel):
    question: str


class ZoneUpdateRequest(BaseModel):
    zones: list[dict]


# ── Vision LLM ───────────────────────────────────────────────────────────────
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
_MODEL_FALLBACKS = ("gemini-3.6-flash", "gemini-3.5-flash-lite")


def _configured_api_key() -> str | None:
    api_key = os.environ.get("LLM_API_KEY")
    if api_key and api_key != "your_gemini_api_key_here":
        return api_key

    # Read .env directly so edits take effect without restarting the API.
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("LLM_API_KEY="):
                    value = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if value and value != "your_gemini_api_key_here":
                        os.environ["LLM_API_KEY"] = value
                        return value
                    break
    return None


def _open_image(image_bytes: bytes) -> Image.Image:
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Image is too large. Please upload an image under 10 MB.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            if source.format not in SUPPORTED_IMAGE_FORMATS:
                raise ValueError("Please upload a JPEG, PNG, or WebP image.")
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image dimensions are too large to analyze safely.")
            return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The uploaded file is not a readable image.") from exc


def analyze_image_with_llm(image_bytes, *, api_key=None, client_factory=None):
    """Return a validated, structured description of one uploaded image."""
    image = _open_image(image_bytes)
    if api_key is None:
        api_key = _configured_api_key()

    if not api_key or api_key == "your_gemini_api_key_here":
        return AnalysisResult(
            description="Image received, but Gemini is not configured yet.",
            objects=[],
            summary="Set LLM_API_KEY to enable image analysis.",
            provider="unavailable",
        ).model_dump()

    try:
        from google.genai import types

        if client_factory is None:
            from google import genai
            client_factory = lambda key: genai.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    timeout=30_000,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )

        client = client_factory(api_key)
        prompt = (
            "Analyze this room or classroom image. Return only objects that are "
            "visibly present; do not infer hidden items or invent details. "
            "For each item, provide its common name, approximate image position, "
            "brief visible evidence, and certainty as clear, partial, or uncertain. "
            "Use an empty objects list when no item can be identified. Keep the "
            "scene description and summary concise."
        )
        model_name = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

        def generate(model):
            return client.models.generate_content(
                model=model,
                contents=[prompt, image],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeminiImageAnalysis,
                ),
            )

        models_to_try = [model_name] + [
            fallback for fallback in _MODEL_FALLBACKS
            if fallback != model_name
        ]
        last_error = None
        for candidate_model in models_to_try:
            try:
                response = generate(candidate_model)
                used_model = candidate_model
                break
            except Exception as exc:
                last_error = exc
        else:
            raise RuntimeError(
                "The configured Gemini model and its fallbacks are unavailable. "
                "Please try again later or check the model access for this API key."
            ) from last_error

        # Parse and validate even when the provider's response is already JSON.
        try:
            parsed = GeminiImageAnalysis.model_validate_json(response.text)
        except ValidationError as exc:
            raise RuntimeError(
                "Gemini returned a response that did not match the expected schema."
            ) from exc
        return AnalysisResult(
            **parsed.model_dump(), provider="gemini", model=used_model
        ).model_dump()
    except Exception as exc:
        raise RuntimeError(f"Gemini image analysis failed: {exc}") from exc


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/analyze-image", response_model=AnalysisResult)
async def analyze_image(file: UploadFile = File(...)):
    """Validate and analyze a user-uploaded image or saved video frame."""
    contents = await file.read(MAX_IMAGE_BYTES + 1)
    try:
        return analyze_image_with_llm(contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/query")
async def query_objects(req: QueryRequest):
    """Answer a natural-language question about tracked objects."""
    result = answer_query(req.question)
    return JSONResponse(content=result)


@app.get("/zones")
async def get_zones():
    """Return the current zone polygon configuration."""
    zm = ZoneManager()
    return JSONResponse(content={"zones": zm.to_dict_list()})


@app.post("/zones")
async def update_zones(req: ZoneUpdateRequest):
    """Update zone polygon configuration and persist to disk."""
    zm = ZoneManager()
    zm.update_zones(req.zones)
    return JSONResponse(content={"status": "ok", "zones": zm.to_dict_list()})


@app.get("/classes")
async def list_classes():
    """Return distinct object classes tracked in the current database."""
    init_db()
    classes = get_distinct_tracked_classes()
    return JSONResponse(content={"classes": classes})


@app.get("/summary")
async def summary():
    """Return per-class observation summary."""
    init_db()
    data = get_observation_summary()
    return JSONResponse(content={"summary": data})


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
