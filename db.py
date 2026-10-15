"""
db.py — SQLite Database Layer for Object Observations

Schema (Task 5 — extended):
  id              INTEGER PRIMARY KEY
  timestamp       TEXT
  frame_number    INTEGER
  object_class    TEXT
  tracking_id     INTEGER       ← raw ByteTrack ID
  confidence      REAL
  bounding_box    TEXT           ← JSON {"x1","y1","x2","y2"}
  zone            TEXT
  global_track_id INTEGER       ← stable ReID-stitched ID  (NEW)
  crop_path       TEXT           ← path to cropped object image  (NEW)
  frame_path      TEXT           ← path to saved keyframe  (NEW)
"""

import datetime
import sqlite3
import os

DB_PATH = "observations.db"


def _get_conn():
    return sqlite3.connect(DB_PATH)


# ── Schema ───────────────────────────────────────────────────────────────────
def init_db():
    """Create the observations table if it doesn't exist (idempotent)."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            frame_number INTEGER,
            object_class TEXT,
            tracking_id INTEGER,
            confidence REAL,
            bounding_box TEXT,
            zone TEXT,
            global_track_id INTEGER,
            crop_path TEXT,
            frame_path TEXT
        )
    """)
    conn.commit()
    conn.close()


def window_cutoff(seconds: float) -> str:
    """Timestamp string matching how observations are stored."""
    cutoff = datetime.datetime.now() - datetime.timedelta(seconds=seconds)
    return str(cutoff)


def prune_older_than(seconds: float) -> int:
    """Delete sightings older than ``seconds``. Returns how many rows were removed."""
    init_db()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM observations WHERE timestamp < ?",
        (window_cutoff(seconds),),
    )
    removed = cursor.rowcount
    conn.commit()
    conn.close()
    return removed


def max_frame_number() -> int:
    """Highest frame number already stored, or 0 when the log is empty."""
    init_db()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT COALESCE(MAX(frame_number), 0) FROM observations")
    value = cursor.fetchone()[0]
    conn.close()
    return int(value or 0)


def reset_db():
    """Drop and recreate the observations table (used before a new file run)."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS observations")
    conn.commit()
    conn.close()
    init_db()


# ── Insert ───────────────────────────────────────────────────────────────────
def insert_observation(timestamp, frame_number, object_class, tracking_id,
                       confidence, bounding_box, zone,
                       global_track_id=None, crop_path=None, frame_path=None):
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO observations
            (timestamp, frame_number, object_class, tracking_id, confidence,
             bounding_box, zone, global_track_id, crop_path, frame_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, frame_number, object_class, tracking_id, confidence,
          bounding_box, zone, global_track_id, crop_path, frame_path))
    conn.commit()
    conn.close()


# ── Basic Queries ────────────────────────────────────────────────────────────
_COLUMNS = [
    "id", "timestamp", "frame_number", "object_class", "tracking_id",
    "confidence", "bounding_box", "zone", "global_track_id", "crop_path",
    "frame_path",
]


def _rows_to_dicts(rows):
    return [dict(zip(_COLUMNS, row)) for row in rows]


def get_all_observations():
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM observations ORDER BY frame_number ASC")
    rows = cursor.fetchall()
    conn.close()
    return _rows_to_dicts(rows)


def get_observations_for_frame(frame_number):
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM observations WHERE frame_number = ?",
                   (frame_number,))
    rows = cursor.fetchall()
    conn.close()
    return _rows_to_dicts(rows)


import json


def ensure_observation_images(obs: dict, force_extract: bool = False) -> dict:
    """Ensure that the observation has valid, existing crop_path and frame_path."""
    if not obs:
        return obs
    crop_path = obs.get("crop_path")
    frame_path = obs.get("frame_path")
    if (not force_extract and crop_path and os.path.exists(crop_path)
            and frame_path and os.path.exists(frame_path)):
        return obs

    obj_class = obs.get("object_class")
    # 1. Check if another observation of this class already has saved images
    if obj_class and not force_extract:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT crop_path, frame_path FROM observations
            WHERE LOWER(object_class) = LOWER(?)
              AND crop_path IS NOT NULL AND frame_path IS NOT NULL
            ORDER BY id DESC LIMIT 1
        """, (obj_class,))
        row = cur.fetchone()
        conn.close()
        if row and row[0] and os.path.exists(row[0]) and row[1] and os.path.exists(row[1]):
            obs["crop_path"] = row[0]
            obs["frame_path"] = row[1]
            return obs

    # 2. Extract directly from video if missing
    frame_number = obs.get("frame_number", 1)
    video_path = None
    run_stats_path = os.path.join("memory", "run_stats.json")
    if os.path.exists(run_stats_path):
        try:
            with open(run_stats_path, "r", encoding="utf-8") as f:
                source_video = json.load(f).get("source_video")
            if source_video and os.path.exists(source_video):
                video_path = source_video
        except (OSError, ValueError, TypeError):
            pass

    if not video_path and os.path.exists("uploads"):
        mp4s = [os.path.join("uploads", f) for f in os.listdir("uploads") if f.endswith(".mp4")]
        if mp4s:
            mp4s.sort(key=os.path.getmtime, reverse=True)
            video_path = mp4s[0]

    if not video_path and os.path.exists("input.mp4"):
        video_path = "input.mp4"
    if not video_path and os.path.exists("output.mp4"):
        video_path = "output.mp4"

    if video_path and os.path.exists(video_path):
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number - 1))
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                os.makedirs("crops", exist_ok=True)
                os.makedirs("frames", exist_ok=True)

                annotated = frame.copy()
                bbox_str = obs.get("bounding_box")
                if bbox_str:
                    try:
                        b = json.loads(bbox_str) if isinstance(bbox_str, str) else bbox_str
                        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
                        h, w = frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        if x2 > x1 and y2 > y1:
                            crop_img = frame[y1:y2, x1:x2]
                            if crop_img.size > 0:
                                cp = os.path.join("crops", f"crop_f{frame_number:04d}_{obj_class}.jpg")
                                cv2.imwrite(cp, crop_img)
                                obs["crop_path"] = cp

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        label = f"{obj_class.title()} - {obs.get('zone', '')}"
                        cv2.putText(annotated, label, (x1, max(25, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    except Exception:
                        pass

                fp = os.path.join("frames", f"frame_{frame_number:04d}.jpg")
                cv2.imwrite(fp, annotated)
                obs["frame_path"] = fp

                if obs.get("id") and (obs.get("crop_path") or obs.get("frame_path")):
                    try:
                        conn = _get_conn()
                        cur = conn.cursor()
                        cur.execute(
                            "UPDATE observations SET crop_path = ?, frame_path = ? WHERE id = ?",
                            (obs.get("crop_path"), obs.get("frame_path"), obs["id"])
                        )
                        conn.commit()
                        conn.close()
                    except Exception:
                        pass
        except Exception:
            pass

    return obs


def get_latest_observation(object_class: str):
    """Return the most recent observation matching *object_class* (case-insensitive)."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM observations
        WHERE LOWER(object_class) = LOWER(?)
        ORDER BY id DESC
        LIMIT 1
    """, (object_class,))
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return None
    obs = _rows_to_dicts(rows)[0]
    return ensure_observation_images(obs)


def get_object_history(object_class: str = None, global_track_id: int = None):
    """
    Return chronological zone-transition history.

    Supply *object_class* for a class-level query or *global_track_id* for a
    specific tracked instance.  Returns a list of dicts with zone-change events.
    """
    conn = _get_conn()
    cursor = conn.cursor()
    if global_track_id is not None:
        cursor.execute("""
            SELECT * FROM observations
            WHERE global_track_id = ?
            ORDER BY id ASC
        """, (global_track_id,))
    elif object_class is not None:
        cursor.execute("""
            SELECT * FROM observations
            WHERE LOWER(object_class) = LOWER(?)
            ORDER BY id ASC
        """, (object_class,))
    else:
        conn.close()
        return []
    rows = cursor.fetchall()
    conn.close()

    all_obs = _rows_to_dicts(rows)
    if not all_obs:
        return []

    # Collapse consecutive same-zone entries into transitions
    transitions = []
    prev_zone = None
    for obs in all_obs:
        if obs["zone"] != prev_zone:
            transitions.append(obs)
            prev_zone = obs["zone"]
    return transitions


def get_distinct_tracked_classes():
    """Return sorted list of distinct object class names in the database."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT object_class FROM observations ORDER BY object_class")
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_observation_summary():
    """Return per-class summary: class, count, last zone, last frame."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        WITH ranked AS (
            SELECT object_class,
                   frame_number,
                   zone,
                   confidence,
                   COUNT(*) OVER (
                       PARTITION BY LOWER(object_class)
                   ) AS total_detections,
                   ROW_NUMBER() OVER (
                       PARTITION BY LOWER(object_class)
                       ORDER BY id DESC
                   ) AS recency_rank
            FROM observations
        )
        SELECT object_class,
               total_detections,
               frame_number,
               zone,
               confidence
        FROM ranked
        WHERE recency_rank = 1
        ORDER BY total_detections DESC, object_class
    """)
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "object_class": cls,
            "total_detections": count,
            "last_seen_frame": frame_number,
            "last_zone": zone or "Unknown",
            "last_confidence": confidence or 0,
        }
        for cls, count, frame_number, zone, confidence in rows
    ]


# ── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    print("Database initialized with extended schema.")
