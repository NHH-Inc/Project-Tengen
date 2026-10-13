"""The score bar is found wherever the broadcast puts it, and whichever side red is on.

Championship and most regionals: top of the frame, red on the left or the right. Space Coast
Showdown 2026: across the bottom. Missing it silently costs the match clock, so every phase.
"""

import unittest

import numpy as np

from ingest.scoreboard import _bar_columns, find_cells

RED, BLUE = (0, 0, 220), (220, 60, 0)  # BGR


def frame(bar_top, red_left=True, height=720, width=1280, bar_height=60):
    image = np.full((height, width, 3), 40, np.uint8)
    left, right = (RED, BLUE) if red_left else (BLUE, RED)
    image[bar_top:bar_top + bar_height, 200:620] = left
    image[bar_top:bar_top + bar_height, 660:1080] = right
    image[bar_top:bar_top + bar_height, 620:660] = (240, 240, 240)  # the timer's white box
    return image


class FindCellsTests(unittest.TestCase):
    def test_a_bar_at_the_top(self):
        cells = find_cells(frame(30))
        self.assertIsNotNone(cells)
        self.assertLess(cells["red"][2], 100)

    def test_a_bar_at_the_bottom(self):
        cells = find_cells(frame(640))
        self.assertIsNotNone(cells, "a bottom score bar must be found")
        self.assertGreaterEqual(cells["red"][2], 600)
        self.assertGreaterEqual(cells["blue"][2], 600)

    def test_blue_on_the_left_spans_both_cells(self):
        cells = find_cells(frame(30, red_left=False))
        columns = _bar_columns(cells)
        self.assertLessEqual(columns.start, 205)
        self.assertGreaterEqual(columns.stop, 1075)

    def test_no_bar_is_none(self):
        self.assertIsNone(find_cells(np.full((720, 1280, 3), 40, np.uint8)))


if __name__ == "__main__":
    unittest.main()
