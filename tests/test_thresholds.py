import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_sft import select_threshold


class ThresholdTests(unittest.TestCase):
    def test_recall_constraint_prevents_overly_high_threshold(self):
        result = select_threshold([0, 0, 1, 1], [0.1, 0.7, 0.6, 0.9], target=1.0)
        self.assertEqual(result["threshold"], 0.6)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["fp"], 1)

    def test_no_positive_labels_rejects_recall_selection(self):
        with self.assertRaises(ValueError):
            select_threshold([0, 0], [0.1, 0.2])

    def test_nonfinite_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            select_threshold([0, 1], [0.2, float("nan")])


if __name__ == "__main__":
    unittest.main()
