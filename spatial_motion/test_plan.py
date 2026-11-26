"""Regression checks for the exact written SpatialMotion collection plan."""
import unittest
from collections import Counter

import numpy as np
from scipy.spatial.distance import pdist

from .plan import PLANES, SPLITS, START_INDICES, STARTS, STEP_M, motions


class PlanTest(unittest.TestCase):
    def test_split_quotas_and_spans(self):
        expected = {
            "train": (396, 396, 792, 1584, 2112, 1584, 528),
            "validation": (44, 44, 132, 264, 352, 264, 88),
            "test": (44, 44, 132, 264, 352, 264, 88, 176),
        }
        self.assertEqual(len(PLANES), 11)
        self.assertEqual(len(STARTS), 100)
        for split in SPLITS:
            counts = Counter()
            for motion in motions(split):
                counts[motion.kind] += len(START_INDICES[split])
                lengths = np.linalg.norm(np.diff(motion.path, axis=0), axis=1)
                np.testing.assert_array_less(lengths, STEP_M + 1e-7)
                span = pdist(motion.path).max()
                # Discrete circle samples may miss the exact antipode.
                self.assertGreaterEqual(span, 0.10 - 0.00005)
                self.assertLessEqual(span, 0.20 + 1e-7)
            self.assertEqual(tuple(counts.values()), expected[split])
        self.assertEqual(sum(sum(counts) for counts in expected.values()), 9944)

    def test_arc_uses_diameter_and_sweep(self):
        arcs = [motion for motion in motions("train") if motion.kind == "arc"]
        self.assertEqual({a.parameters["diameter_cm"] for a in arcs}, {16, 20})
        self.assertEqual({a.parameters["sweep_deg"] for a in arcs}, {90, 120, 180})
        for arc in arcs:
            expected = arc.parameters["diameter_cm"] / 100 * np.sin(
                np.deg2rad(arc.parameters["sweep_deg"] / 2))
            self.assertAlmostEqual(np.linalg.norm(arc.path[-1]), expected, places=6)


if __name__ == "__main__":
    unittest.main()
