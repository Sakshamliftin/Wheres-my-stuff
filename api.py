"""
api.py — FastAPI Backend

Endpoints:
  POST /analyze-image    — Send an image frame to the Vision LLM (Gemini 3.8 Flash)
  POST /query            — Natural-language "where is my …" query against the DB
  GET  /zones            — Retrieve current zone configuration
  POST /zones            — Update zone polygon configuration
  GET  /classes          — List distinct object classes tracked in the DB
  GET  /summary          — Observation summary per class
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
from pydantic import BaseModel
import io
from PIL import Image

from query_engine import answer_query
from zones import ZoneManager
from db import get_distinct_tracked_classes, get_observation_summary, init_db

app = FastAPI(title="Where Is My Stuff? — API", version="2.0")


# ── Models ───────────────────────────────────────────────────────────────────
class AnalysisResult(BaseModel):
    description: str
    objects: list[str]
    summary: str


class QueryRequest(BaseModel):
    question: str


class ZoneUpdateRequest(BaseModel):
    zones: list[dict]


# ── Vision LLM ───────────────────────────────────────────────────────────────
def analyze_image_with_llm(image_bytes):
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        # Check .env file directly so edits take effect immediately without restarting
        env_file = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_file):
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("LLM_API_KEY="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val and val != "your_gemini_api_key_here":
                            api_key = val
                            os.environ["LLM_API_KEY"] = val
                        break

    if not api_key or api_key == "your_gemini_api_key_here":
        return {
            "description": "[MOCK] A view of a room with a desk and some objects.",
            "objects": ["bottle", "laptop", "book (mock)"],
            "summary": "Mock summary because LLM_API_KEY is not set in .env."
        }

    try:
        import google.genai as genai
        client = genai.Client(api_key=api_key)

        image = Image.open(io.BytesIO(image_bytes))

        prompt = (
            "Briefly describe this scene and identify the visible objects. "
            "Return a short description."
        )

        model_name = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, image]
            )
        except Exception as model_err:
            # Fall back to the previously supported vision model if 3.8 is unavailable.
            if "gemini-3.8" in model_name:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[prompt, image]
                )
            else:
                raise model_err

        text = response.text

        return {
            "description": text,
            "objects": ["Extracted from Gemini response"],
            "summary": "Successfully analyzed with Gemini"
        }
    except Exception as e:
        return {
            "description": f"Error analyzing image: {str(e)}",
            "objects": [],
            "summary": "API call failed."
        }


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/analyze-image", response_model=AnalysisResult)
async def analyze_image(file: UploadFile = File(...)):
    """Send a frame image to the Vision LLM for scene description."""
    contents = await file.read()
    result = analyze_image_with_llm(contents)
    return result


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
