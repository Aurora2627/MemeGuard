import unittest
import numpy as np
from memeguard.data import validate_row, pair_digest
from memeguard.metrics import binary_metrics
from memeguard.runtime import instruction


class GuardTests(unittest.TestCase):
    def row(self):
        return {
            "id": "example",
            "label": 0,
            "text": "example caption",
            "role": "capacity_train",
            "split": "train",
        }

    def test_missing_label_is_rejected(self):
        row = self.row()
        del row["label"]
        with self.assertRaises(ValueError):
            validate_row(row)

    def test_test_rows_cannot_be_training_pilot(self):
        row = self.row()
        row["split"] = "test_seen"
        with self.assertRaises(ValueError):
            validate_row(row)

    def test_pair_hash_normalizes_caption_whitespace(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        self.assertEqual(pair_digest(image, "A  B"), pair_digest(image, "a\nb"))

    def test_same_image_different_caption_is_not_same_pair(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        self.assertNotEqual(pair_digest(image, "first"), pair_digest(image, "second"))

    def test_known_confusion_matrix(self):
        metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.8, 0.7, 0.2])
        self.assertEqual([metrics[k] for k in ("tp", "fp", "fn", "tn")], [1, 1, 1, 1])
        self.assertEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["auroc"], 0.5)

    def test_one_class_auc_is_missing(self):
        self.assertIsNone(binary_metrics([0, 0], [0.1, 0.2])["auroc"])

    def test_nonfinite_prediction_rejected(self):
        with self.assertRaises(ValueError):
            binary_metrics([0, 1], [0.2, float("nan")])

    def test_label_never_changes_prompt(self):
        first = self.row()
        second = dict(first, label=1)
        self.assertEqual(instruction(first), instruction(second))


if __name__ == "__main__":
    unittest.main()
