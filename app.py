"""
app.py — Streamlit Frontend (Task 5 — Multi-Tab Dashboard)

Tabs:
  1. Video Processing & Zone Config
  2. "Where Is My Stuff?" — NL Query Search
  3. Object Movement History
  4. AI Scene Analysis (Gemini)
"""

import streamlit as st
import os
import threading
import requests
import cv2
import pandas as pd
from datetime import datetime, timedelta
from PIL import Image
from pipeline import process_stream, process_video
from db import (
    init_db,
    get_all_observations,
    get_distinct_tracked_classes,
    get_observation_summary,
    get_latest_observation,
    get_object_history,
    ensure_observation_images,
)
from query_engine import answer_query
from vision_client import request_image_analysis
from zones import (
    ZoneManager,
    canvas_polygon_points,
    default_zones,
    hex_to_bgr,
    hex_to_rgba,
)

_CANVAS_MAX_WIDTH = 900
_ZONE_FRAME_PATH = os.path.join("uploads", "_zone_freeze.jpg")
_LIVE_PREVIEW = "live_preview.jpg"
_LIVE_PREVIEW_LOCK = threading.Lock()
_LIVE = {
    "stop": False,
    "thread": None,
    "error": None,
    "preview_ready": False,
    "frame_number": None,
}


def _live_frame(annotated_frame, frame_number):
    """Save a preview for the dashboard. Return True when the user pressed Stop."""
    if frame_number % 3 == 1:
        tmp = _LIVE_PREVIEW + ".tmp.jpg"
        height, width = annotated_frame.shape[:2]
        longest_side = max(height, width)
        preview = annotated_frame
        if longest_side > 1280:
            scale = 1280 / longest_side
            preview = cv2.resize(
                annotated_frame,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        cv2.imwrite(
            tmp,
            preview,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )
        with _LIVE_PREVIEW_LOCK:
            os.replace(tmp, _LIVE_PREVIEW)
            _LIVE["preview_ready"] = True
            _LIVE["frame_number"] = frame_number
    return _LIVE["stop"]


def _live_worker(source: str, detection_imgsz: int):
    try:
        process_stream(
            source,
            frame_callback=_live_frame,
            detection_imgsz=detection_imgsz,
        )
    except Exception as exc:
        _LIVE["error"] = str(exc)
    finally:
        _LIVE["stop"] = True


def _live_running() -> bool:
    thread = _LIVE["thread"]
    return thread is not None and thread.is_alive()


def _read_live_preview():
    """Read one consistent preview and its source frame number."""
    with _LIVE_PREVIEW_LOCK:
        if not _LIVE["preview_ready"] or not os.path.exists(_LIVE_PREVIEW):
            return None
        with open(_LIVE_PREVIEW, "rb") as image_file:
            return image_file.read(), _LIVE["frame_number"]


def _render_image_analysis_result(result: dict):
    if result.get("provider") == "unavailable":
        st.warning(result.get("summary", "Gemini is not configured."))
    else:
        st.success("Image analyzed")
        st.caption(
            f"Provider: Gemini · Model: {result.get('model', 'configured model')}"
        )

    with st.container(border=True):
        st.subheader("Scene")
        st.write(result.get("description", "No description returned."))
        st.markdown("**Visible objects**")
        objects = result.get("objects", [])
        if objects:
            object_rows = [
                {
                    "Object": item.get("name", "Unknown"),
                    "Approximate position": item.get("position", "unclear"),
                    "Visual evidence": item.get("visual_notes", ""),
                    "Certainty": item.get("certainty", "uncertain"),
                }
                for item in objects
            ]
            st.dataframe(
                pd.DataFrame(object_rows),
                width="stretch",
                hide_index=True,
                alt="Structured list of objects visible in the selected image",
            )
        else:
            st.info("No clear objects were identified in this image.")
        if result.get("summary"):
            st.caption(result["summary"])


def _first_frame(video_path: str):
    """Return the first frame of a video, or None if it cannot be read."""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


def _preview_bgr(frame_bgr, zones: list[dict]):
    """Draw the current draft zones on a copy of the freeze-frame."""
    preview = frame_bgr.copy()
    if zones:
        mgr = ZoneManager()
        mgr.zones = zones
        mgr.draw_zones(preview, alpha=0.35)
    return preview

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Where Is My Stuff?", layout="wide")


@st.fragment(run_every=timedelta(seconds=1))
def _live_panel():
    """Refresh the preview and the latest zone of each object while the camera runs."""
    if _LIVE["error"]:
        st.error(_LIVE["error"])
    live_preview = _read_live_preview()
    has_live_frame = live_preview is not None
    if live_preview:
        st.image(live_preview[0], caption="Live camera frame", width=480,
                 alt="Latest annotated camera frame")
    elif _live_running():
        st.info("Waiting for the first frame from the camera.")

    if st.button(
        "Analyze latest frame with Gemini",
        key="analyze_live_frame",
        disabled=not has_live_frame,
    ):
        st.session_state.pop("live_image_analysis", None)
        st.session_state.pop("live_image_analysis_error", None)
        try:
            live_preview = _read_live_preview()
            if live_preview is None:
                raise RuntimeError("No live camera frame is ready to analyze.")
            image_bytes, frame_number = live_preview
            result = request_image_analysis(
                image_bytes,
                f"live-frame-{frame_number}.jpg",
                "image/jpeg",
            )
            st.session_state["live_image_analysis"] = {
                "frame_number": frame_number,
                "result": result,
            }
        except (OSError, requests.RequestException, RuntimeError, ValueError) as exc:
            st.session_state["live_image_analysis_error"] = str(exc)

    if st.session_state.get("live_image_analysis_error"):
        st.error(st.session_state["live_image_analysis_error"])
    live_analysis = st.session_state.get("live_image_analysis")
    if live_analysis:
        st.caption(f"Gemini analysis of live frame {live_analysis['frame_number']}")
        _render_image_analysis_result(live_analysis["result"])

    summary = get_observation_summary()
    if summary:
        st.markdown("**Latest zone**")
        cols = st.columns(min(len(summary), 4))
        for i, item in enumerate(summary):
            with cols[i % len(cols)]:
                st.metric(item["object_class"].title(), item["last_zone"])
    elif _live_running():
        st.caption("No objects detected yet.")


with st.container(border=True):
    st.caption(":material/home_pin: ROOM MEMORY")
    st.title("Where Is My Stuff?")
    st.caption("Track everyday items and find their last-seen location.")

os.makedirs("uploads", exist_ok=True)
os.makedirs("frames", exist_ok=True)
os.makedirs("crops", exist_ok=True)

# Ensure DB exists
init_db()

if "draft_zones" not in st.session_state:
    st.session_state.draft_zones = []
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0
if "zone_video_name" not in st.session_state:
    st.session_state.zone_video_name = None

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    ":material/video_library: Video & zones",
    ":material/search: Find an item",
    ":material/history: Movement history",
    ":material/auto_awesome: Scene analysis",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Video Processing & Zone Configuration
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Video Processing & Zone Configuration")

    col_upload, col_zones = st.columns([3, 2])

    with col_zones:
        st.subheader("Saved Zones")
        zm = ZoneManager()
        if zm.zones and os.path.exists(zm.config_path):
            st.caption("Loaded from `zones.json`.")
        else:
            st.caption("No saved file yet. Defaults are the four quadrants.")

        for i, zone in enumerate(zm.zones):
            st.markdown(f"**{i + 1}. {zone['name']}** — "
                        f"{len(zone['points'])} vertices")

    with col_upload:
        st.subheader("Upload & Process Video")
        uploaded_file = st.file_uploader("Upload Room Video",
                                         type=["mp4", "avi", "mov"])

        if uploaded_file is not None:
            video_path = os.path.join("uploads", uploaded_file.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ Uploaded **{uploaded_file.name}**")

            if st.session_state.zone_video_name != uploaded_file.name:
                freeze = _first_frame(video_path)
                if freeze is None:
                    st.error("Could not read a frame from this video.")
                else:
                    cv2.imwrite(_ZONE_FRAME_PATH, freeze)
                    h, w = freeze.shape[:2]
                    st.session_state.zone_video_name = uploaded_file.name
                    st.session_state.zone_frame_size = (w, h)
                    st.session_state.draft_zones = []
                    st.session_state.canvas_key = 0

            if st.button("▶️ Start Processing", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(current, total):
                    if total > 0:
                        progress_bar.progress(current / total)
                    status_text.text(
                        f"Processing frame {current}/{total}")

                with st.spinner("Running detection and tracking pipeline…"):
                    output_video_path = "output.mp4"
                    if os.path.exists(output_video_path):
                        os.remove(output_video_path)

                    output_video_path = process_video(
                        video_path,
                        output_video_path=output_video_path,
                        progress_callback=update_progress,
                    )

                st.success("✅ Processing Complete!")

                st.subheader("Annotated Video")
                st.caption(
                    "Tracked detections show stable GID labels. "
                    "UNTRACKED boxes are still rendered for visibility."
                )
                video_check = cv2.VideoCapture(output_video_path)
                fourcc = int(video_check.get(cv2.CAP_PROP_FOURCC))
                video_check.release()
                video_codec = "".join(
                    chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)
                ).lower()
                if video_codec not in {"avc1", "h264"}:
                    st.info(
                        "This run used a codec some browsers cannot play. "
                        "Download the annotated video to watch it."
                    )
                try:
                    st.video(output_video_path)
                except Exception:
                    st.warning("Video cannot play in-browser (codec). "
                               "Download below.")

                with open(output_video_path, "rb") as f:
                    st.download_button("⬇️ Download Annotated Video", f,
                                       file_name="annotated_output.mp4")

    st.markdown("---")
    st.subheader("Live camera")
    st.caption("Webcam index `0`, or an RTSP address. "
               "Sightings from the last 30 minutes are kept.")
    camera_source = st.text_input("Camera", value="0")
    live_detection_imgsz = st.selectbox(
        "Live detection detail",
        options=[640, 800, 1024],
        index=1,
        format_func=lambda size: {
            640: "Fast (640 px)",
            800: "Balanced (800 px)",
            1024: "Detailed (1024 px)",
        }[size],
        help="Smaller sizes run faster but may miss small or partly hidden items.",
        key="live_detection_imgsz",
    )
    start_col, stop_col = st.columns(2)
    with start_col:
        if st.button("Start camera", type="primary", disabled=_live_running()):
            _LIVE["stop"] = False
            _LIVE["error"] = None
            with _LIVE_PREVIEW_LOCK:
                _LIVE["preview_ready"] = False
                _LIVE["frame_number"] = None
            st.session_state.pop("live_image_analysis", None)
            st.session_state.pop("live_image_analysis_error", None)
            source = camera_source.strip() or "0"
            _LIVE["thread"] = threading.Thread(
                target=_live_worker,
                args=(source, live_detection_imgsz),
                daemon=True,
            )
            _LIVE["thread"].start()
            st.rerun()
    with stop_col:
        if st.button("Stop camera", disabled=not _live_running()):
            _LIVE["stop"] = True
    if _live_running():
        st.success("Camera is running.")
    else:
        st.caption("Camera is stopped.")
    _live_panel()

    # ── Draw zones on the uploaded freeze-frame ──────────────────────────
    st.markdown("---")
    st.subheader("Draw Zones")
    if st.session_state.pop("zone_save_ok", False):
        st.success("Saved to `zones.json`. Processing will use these polygons.")
    if (st.session_state.zone_video_name is None
            or not os.path.exists(_ZONE_FRAME_PATH)):
        st.info("Upload a video to draw zones on its first frame. "
                "Until then, processing uses `zones.json` or the default "
                "quadrants.")
    else:
        freeze = cv2.imread(_ZONE_FRAME_PATH)
        frame_w, frame_h = st.session_state.zone_frame_size
        scale = min(1.0, _CANVAS_MAX_WIDTH / max(frame_w, 1))
        disp_w = max(1, int(frame_w * scale))
        disp_h = max(1, int(frame_h * scale))
        x_scale = frame_w / disp_w
        y_scale = frame_h / disp_h

        preview = _preview_bgr(freeze, st.session_state.draft_zones)
        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        background = Image.fromarray(rgb).resize((disp_w, disp_h))

        draw_col, list_col = st.columns([3, 2])
        with list_col:
            st.markdown("**Zones for this video**")
            zone_name = st.text_input("Zone name", value="Desk")
            zone_color = st.color_picker("Zone colour", value="#00FF00")
            st.caption(
                "Left-click to place vertices, then right-click to close the "
                "polygon. Click the first toolbar icon below the canvas "
                "(Send to Streamlit), then click Add polygon as zone. "
                "Double-click removes the last vertex."
            )

            if st.session_state.draft_zones:
                for i, zone in enumerate(st.session_state.draft_zones):
                    c1, c2 = st.columns([4, 1])
                    c1.markdown(
                        f"**{i + 1}. {zone['name']}** — "
                        f"{len(zone['points'])} vertices"
                    )
                    if c2.button("✕", key=f"drop_zone_{i}"):
                        st.session_state.draft_zones.pop(i)
                        st.rerun()
            else:
                st.caption("No zones drawn yet.")

            if st.button("Use quadrant defaults"):
                st.session_state.draft_zones = default_zones(frame_w, frame_h)
                st.session_state.canvas_key += 1
                st.rerun()

            if st.button("Save zones", type="primary"):
                if not st.session_state.draft_zones:
                    st.warning("Draw at least one zone before saving.")
                else:
                    ZoneManager().update_zones(st.session_state.draft_zones)
                    st.session_state.zone_save_ok = True
                    st.rerun()

        with draw_col:
            try:
                from streamlit_drawable_canvas import st_canvas
            except ImportError:
                st.error("Install `streamlit-drawable-canvas-fix` "
                         "(`pip install -r requirements.txt`) to draw zones.")
                st_canvas = None

            canvas_result = None
            if st_canvas is not None:
                canvas_result = st_canvas(
                    fill_color=hex_to_rgba(zone_color, 0.35),
                    stroke_width=2,
                    stroke_color=zone_color,
                    background_image=background,
                    update_streamlit=True,
                    height=disp_h,
                    width=disp_w,
                    drawing_mode="polygon",
                    display_toolbar=True,
                    key=f"zone_canvas_{st.session_state.canvas_key}",
                )

            if st.button("Add polygon as zone", type="primary"):
                objects = []
                if canvas_result is not None and canvas_result.json_data:
                    objects = canvas_result.json_data.get("objects") or []
                polygons = [
                    canvas_polygon_points(obj, x_scale, y_scale)
                    for obj in objects
                    if obj.get("type") == "polygon"
                ]
                polygons = [pts for pts in polygons if len(pts) >= 3]
                name = zone_name.strip()
                if not name:
                    st.warning("Give the zone a name.")
                elif not polygons:
                    st.warning(
                        "No submitted polygon found. Right-click to close "
                        "the shape, click the first canvas toolbar icon "
                        "(Send to Streamlit), then click Add polygon as zone."
                    )
                else:
                    points = polygons[-1]
                    points = [
                        [min(max(0, x), frame_w - 1),
                         min(max(0, y), frame_h - 1)]
                        for x, y in points
                    ]
                    st.session_state.draft_zones.append({
                        "name": name,
                        "color": hex_to_bgr(zone_color),
                        "points": points,
                    })
                    st.session_state.canvas_key += 1
                    st.rerun()

    # Observations table
    st.markdown("---")
    st.subheader("📊 Observation Database")
    try:
        obs = get_all_observations()
        if obs:
            df = pd.DataFrame(obs)
            if "tracking_id" in df.columns:
                df["detection_state"] = df["tracking_id"].apply(
                    lambda tid: "tracked" if pd.notna(tid) and int(tid) >= 0 else "untracked"
                )
            # Clean up display
            display_cols = [
                "frame_number",
                "object_class",
                "detection_state",
                "global_track_id",
                "tracking_id",
                "zone",
                "confidence",
                "timestamp",
            ]
            available = [c for c in display_cols if c in df.columns]
            st.dataframe(df[available], width="stretch", height=300,
                         hide_index=True)
            st.caption(f"Total observations: {len(df)}")
        else:
            st.info("No observations found. Process a video first.")
    except Exception:
        st.info("Database not initialised or no observations yet.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Natural Language Query ("Where Is My Stuff?")
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Where Is My Stuff?")
    st.markdown("Ask a question about any object tracked in the video.")

    # Ensure session state for search exists
    if "query_input_text" not in st.session_state:
        st.session_state.query_input_text = ""
    if "trigger_search" not in st.session_state:
        st.session_state.trigger_search = False

    def _select_chip(query_text):
        st.session_state.query_input_text = query_text
        st.session_state.trigger_search = True

    # Example chips
    example_queries = [
        "Where are my keys?",
        "Where did I leave my wallet?",
        "Where are my spectacles?",
        "Where are my earphones?",
        "Where is my watch?",
        "Where is my cream bottle?",
        "Where is my bag?",
        "What objects do you see?",
    ]
    st.caption("💡 Try one of these:")
    chip_row1 = st.columns(4)
    for i, eq in enumerate(example_queries[:4]):
        with chip_row1[i]:
            st.button(eq, key=f"chip_{i}", on_click=_select_chip, args=(eq,), width="stretch")

    chip_row2 = st.columns(4)
    for i, eq in enumerate(example_queries[4:], start=4):
        with chip_row2[i - 4]:
            st.button(eq, key=f"chip_{i}", on_click=_select_chip, args=(eq,), width="stretch")

    query_input = st.text_input(
        "Your question:",
        key="query_input_text",
        placeholder="e.g. Where are my glasses?",
    )

    search_clicked = st.button("🔎 Search", type="primary")
    auto_search = st.session_state.get("trigger_search", False)
    st.session_state.trigger_search = False

    if (search_clicked or auto_search) and query_input:
        with st.spinner("Searching…"):
            result = answer_query(query_input)

        latest = result.get("latest")
        if result["success"] and latest:
            # Regenerate from this exact sighting so the images match its frame.
            latest = ensure_observation_images(latest, force_extract=True)

            st.success("Found")
            seen_at = latest.get("timestamp") or "Unknown"
            try:
                seen_at = datetime.fromisoformat(str(seen_at)).strftime(
                    "%d %b %Y, %I:%M %p"
                )
            except (TypeError, ValueError):
                pass

            with st.container(border=True):
                st.markdown("#### Last seen")
                item_col, zone_col, time_col = st.columns(3)
                item_col.metric(
                    "Item",
                    (result.get("object_class") or latest.get("object_class") or "Object").title(),
                )
                zone_col.metric("Location", latest.get("zone") or "Unknown")
                time_col.metric("Time", seen_at)

                image_cols = st.columns(2)
                crop_path = latest.get("crop_path")
                if crop_path and os.path.exists(crop_path):
                    image_cols[0].image(
                        crop_path,
                        caption="Detected item",
                        width=320,
                        alt="Crop of the item from its last-seen frame",
                    )
                frame_path = latest.get("frame_path")
                if frame_path and os.path.exists(frame_path):
                    image_cols[1].image(
                        frame_path,
                        caption="Last-seen frame",
                        width=480,
                        alt="Room frame showing the item's last-seen location",
                    )
        elif result["success"]:
            # For the all-objects question, show only each class's latest sighting.
            st.success("Found")
            summary = get_observation_summary()
            if summary:
                last_seen = pd.DataFrame([
                    {
                        "Item": row["object_class"].title(),
                        "Last seen location": row["last_zone"],
                        "Last seen frame": row["last_seen_frame"],
                    }
                    for row in summary
                ])
                st.dataframe(last_seen, width="stretch", hide_index=True)
            else:
                st.info("No observations have been recorded yet.")
        else:
            requested_item = result.get("object_class")
            if requested_item and requested_item.casefold() == "bag":
                st.warning(
                    "Bag is supported, but no bag was detected in this video. "
                    "There is no last-seen image to show."
                )
            elif requested_item:
                st.warning(
                    f"No last-seen record for **{requested_item.title()}** in this video."
                )
            else:
                st.info("No tracked items yet. Process a video first.")

    # Show tracked classes for reference
    st.markdown("---")
    classes = get_distinct_tracked_classes()
    if classes:
        st.caption(f"**Tracked objects in DB:** {', '.join(classes)}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Object Movement History
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📋 Object Movement History")
    st.markdown("See how objects moved between zones over time.")

    classes = get_distinct_tracked_classes()
    if not classes:
        st.info("No data yet. Process a video first.")
    else:
        # Summary cards
        st.subheader("Summary")
        summary = get_observation_summary()
        if summary:
            card_cols = st.columns(min(len(summary), 4))
            for i, s in enumerate(summary):
                with card_cols[i % len(card_cols)]:
                    st.metric(
                        label=s["object_class"].title(),
                        value=s["last_zone"],
                        delta=f"{s['total_detections']} detections",
                    )

        st.markdown("---")
        st.subheader("Zone Transition Timeline")
        selected_class = st.selectbox(
            "Select object class:", classes)

        if selected_class:
            history = get_object_history(object_class=selected_class)
            if history:
                st.markdown(f"**{selected_class.title()}** — "
                            f"{len(history)} zone transition(s):")

                for i, h in enumerate(history):
                    icon = "🟢" if i == 0 else "➡️"
                    st.markdown(
                        f"{icon} **{h['zone']}** — frame "
                        f"{h['frame_number']} "
                        f"(confidence {h['confidence']:.1%})"
                    )
                    # Show frame if available
                    fp = h.get("frame_path")
                    if fp and os.path.exists(fp):
                        st.image(fp, width=320,
                                 caption=f"Frame {h['frame_number']}",
                                 alt=f"Scene at frame {h['frame_number']}")
            else:
                st.info(f"No transitions found for **{selected_class}**.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — AI Scene Analysis (Gemini)
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("🤖 AI Scene Analysis")
    st.markdown(
        "Upload a room or classroom image, or choose a saved video frame, "
        "to get a concise scene description and a structured visible-object list."
    )

    frame_files = sorted(
        [f for f in os.listdir("frames") if f.endswith(".jpg")]
    ) if os.path.exists("frames") else []

    image_source_options = ["Upload an image"]
    if frame_files:
        image_source_options.append("Choose a processed video frame")
    image_source = st.selectbox("Image source", image_source_options)

    image_bytes = None
    image_name = None
    image_type = None
    if image_source == "Upload an image":
        uploaded_image = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "webp"],
            key="scene_image_upload",
            help="JPEG, PNG, or WebP; maximum 10 MB.",
        )
        if uploaded_image is not None:
            image_bytes = uploaded_image.getvalue()
            image_name = uploaded_image.name
            image_type = uploaded_image.type or "application/octet-stream"
            st.image(
                uploaded_image,
                caption=image_name,
                width=450,
                alt="Image selected for scene analysis",
            )
    else:
        selected_frame = st.selectbox("Select a saved frame", frame_files)
        frame_path = os.path.join("frames", selected_frame)
        st.image(
            frame_path,
            caption=selected_frame,
            width=450,
            alt=f"Selected room frame {selected_frame}",
        )
        with open(frame_path, "rb") as frame_file:
            image_bytes = frame_file.read()
        image_name = selected_frame
        image_type = "image/jpeg"

    if st.button(
        "🧠 Analyze image",
        type="primary",
        disabled=image_bytes is None,
    ):
        with st.spinner("Analyzing image…"):
            try:
                result = request_image_analysis(
                    image_bytes, image_name, image_type
                )
                _render_image_analysis_result(result)
            except requests.RequestException as exc:
                st.error(
                    f"Could not reach the image-analysis API: {exc}. "
                    "Check that the FastAPI server is running on port 8000."
                )
            except RuntimeError as exc:
                st.error(str(exc))
            except ValueError as exc:
                st.error(f"The API returned an unreadable response: {exc}")
