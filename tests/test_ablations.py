import unittest
from memeguard.ablations import messages_for, rotated_images


class AblationTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(id="a", image="a.png", text="example", label=0)

    def test_label_never_changes_messages(self):
        for mode in ("joint", "text_only", "image_only", "shuffled_image"):
            self.assertEqual(
                messages_for(self.row, mode),
                messages_for(dict(self.row, label=1), mode),
            )

    def test_text_only_has_no_image(self):
        content = messages_for(self.row, "text_only")[0]["content"]
        self.assertEqual([x["type"] for x in content], ["text"])
        self.assertTrue(content[0]["text"].endswith("example"))

    def test_image_only_omits_caption(self):
        content = messages_for(self.row, "image_only")[0]["content"]
        self.assertEqual(content[0]["type"], "image")
        self.assertFalse(content[-1]["text"].endswith("example"))

    def test_shuffle_has_no_self_matches(self):
        rows = [dict(self.row, id=str(i), image=str(i) + ".png") for i in range(3)]
        mapping = rotated_images(rows)
        self.assertTrue(all(mapping[r["id"]] != r["image"] for r in rows))
        self.assertEqual(mapping, rotated_images(list(reversed(rows))))

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            messages_for(self.row, "unknown")
