"""Tests for turning drawable-canvas point markers into zone vertices."""

import unittest

from zones import canvas_point_position


class CanvasPointPositionTests(unittest.TestCase):
    def test_maps_circle_center_back_to_source_frame_coordinates(self):
        marker = {
            "type": "circle",
            "left": 10,
            "top": 25,
            "width": 12,
            "height": 12,
            "scaleX": 1.5,
            "scaleY": 2,
            "originX": "left",
            "originY": "center",
        }

        self.assertEqual(canvas_point_position(marker, 2, 3), [38, 75])

    def test_ignores_non_points_and_zero_sized_markers(self):
        self.assertIsNone(
            canvas_point_position({"type": "polygon", "points": []})
        )
        self.assertIsNone(
            canvas_point_position(
                {"type": "circle", "width": 0, "height": 10}
            )
        )


if __name__ == "__main__":
    unittest.main()
