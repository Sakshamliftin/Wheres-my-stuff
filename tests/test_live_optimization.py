"""Tests for live observation throttling without opening a camera."""

import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import pipeline
from pipeline import should_log_live_observation


class LiveObservationLogTests(unittest.TestCase):
    def test_logs_first_seen_then_throttles_same_zone(self):
        state = {}
        key = ("laptop", "global", 17)

        self.assertTrue(should_log_live_observation(state, key, "Desk", 10.0))
        self.assertFalse(should_log_live_observation(state, key, "Desk", 11.9))
        self.assertEqual(state[key], ("Desk", 10.0))

    def test_logs_periodically_and_debounces_zone_changes(self):
        state = {}
        key = ("laptop", "global", 17)
        should_log_live_observation(state, key, "Desk", 10.0)

        self.assertTrue(should_log_live_observation(state, key, "Desk", 12.0))
        self.assertFalse(should_log_live_observation(state, key, "Shelf", 12.2))
        self.assertTrue(should_log_live_observation(state, key, "Shelf", 12.5))
        self.assertEqual(state[key], ("Shelf", 12.5))

    def test_tracks_have_independent_throttling(self):
        state = {}
        first = ("laptop", "global", 17)
        second = ("laptop", "global", 18)
        self.assertTrue(should_log_live_observation(state, first, "Desk", 10.0))
        self.assertTrue(should_log_live_observation(state, second, "Desk", 10.1))

    def test_live_pipeline_throttles_database_rows_and_passes_size(self):
        class Tensor:
            def __init__(self, values):
                self.values = values

            def cpu(self):
                return self

            def int(self):
                return self

            def tolist(self):
                return list(self.values)

            def numpy(self):
                return np.asarray(self.values)

        class Boxes:
            def __init__(self):
                self.xyxy = Tensor([[10, 10, 30, 30]])
                self.cls = Tensor([0])
                self.conf = Tensor([0.9])
                self.id = Tensor([5])

            def __len__(self):
                return 1

        class FakeModel:
            names = {0: "laptop"}

            def __init__(self):
                self.img_sizes = []

            def track(self, frame, **kwargs):
                self.img_sizes.append(kwargs["imgsz"])
                return [SimpleNamespace(boxes=Boxes())]

        class FakeCapture:
            def __init__(self, frame):
                self.frame = frame
                self.remaining = 4

            def get(self, property_id):
                if property_id == pipeline.cv2.CAP_PROP_FPS:
                    return 30
                if property_id == pipeline.cv2.CAP_PROP_FRAME_COUNT:
                    return 5
                return 0

            def read(self):
                if self.remaining <= 0:
                    return False, None
                self.remaining -= 1
                return True, self.frame.copy()

            def release(self):
                pass

        class FakeWriter:
            def write(self, frame):
                pass

            def release(self):
                pass

        class FakeZones:
            def __init__(self):
                self.zones = iter(["Desk", "Desk", "Shelf", "Shelf", "Shelf"])

            def classify_point(self, x, y):
                return next(self.zones)

            def draw_zones(self, frame, alpha):
                pass

        class FakeTracker:
            backend = "test"
            stitch_count = 0

            def update(self, *args, **kwargs):
                return 7

            def end_of_frame(self, *args, **kwargs):
                pass

            def embedding_for(self, global_id):
                return None

        class FakeMemory:
            def __init__(self, directory):
                self.directory = directory

            def load(self):
                return self

            def save(self):
                pass

            def drop_older_than(self, cutoff):
                pass

        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as temp_dir, ExitStack() as stack:
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            model = FakeModel()
            memory = FakeMemory(os.path.join(temp_dir, "memory"))
            cap = FakeCapture(frame)
            inserted = []
            clocks = iter([10.0, 10.1, 10.2, 10.6, 12.1])

            stack.enter_context(patch.object(pipeline, "load_detector", return_value=model))
            stack.enter_context(patch.object(pipeline, "init_db"))
            stack.enter_context(patch.object(pipeline, "max_frame_number", return_value=0))
            stack.enter_context(patch.object(pipeline, "_prune_window"))
            stack.enter_context(patch.object(pipeline, "ObservationMemory", return_value=memory))
            stack.enter_context(patch.object(pipeline, "GlobalTracker", return_value=FakeTracker()))
            stack.enter_context(patch.object(pipeline, "ZoneManager", return_value=FakeZones()))
            stack.enter_context(patch.object(pipeline, "_open_capture", return_value=(cap, frame.copy())))
            stack.enter_context(patch.object(pipeline, "_open_video_writer", return_value=(FakeWriter(), "mp4v")))
            stack.enter_context(
                patch.object(
                    pipeline,
                    "insert_observation",
                    side_effect=lambda *args, **kwargs: inserted.append(
                        (args[1], kwargs.get("crop_path"), kwargs.get("frame_path"))
                    ),
                )
            )
            stack.enter_context(
                patch.object(pipeline, "time", SimpleNamespace(monotonic=lambda: next(clocks)))
            )

            pipeline._run_pipeline(
                "mock-live-source",
                output_video_path=os.path.join(temp_dir, "live.mp4"),
                frames_dir=os.path.join(temp_dir, "frames"),
                crops_dir=os.path.join(temp_dir, "crops"),
                max_frames=5,
                reset=False,
                window_seconds=1800,
                detection_imgsz=640,
            )

            self.assertEqual([row[0] for row in inserted], [1, 4])
            self.assertTrue(all(row[1] and row[2] for row in inserted))
            self.assertEqual(model.img_sizes, [640] * 5)
            with open(os.path.join(memory.directory, "run_stats.json"), encoding="utf-8") as stats_file:
                self.assertEqual(json.load(stats_file)["detection_imgsz"], 640)


if __name__ == "__main__":
    unittest.main()
