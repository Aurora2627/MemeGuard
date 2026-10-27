"""Final evaluation must reject altered assets and mismatched prediction resumes."""

from pathlib import Path
import sys, tempfile, unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import final_test as final
from prepare_final import paired_rows
from memeguard.common import save, sha


class FinalProtocolTests(unittest.TestCase):
    def test_prediction_resume_identity(self):
        row = dict(id="a", label=1, image_sha256="hash")
        final.validate_prefix([row], [row])
        for wrong in (
            dict(row, id="b"),
            dict(row, label=0),
            dict(row, image_sha256="changed"),
        ):
            with self.assertRaises(ValueError):
                final.validate_prefix([wrong], [row])
        with self.assertRaises(ValueError):
            final.validate_prefix([row, row], [row])

    def test_pairing_rejects_changed_membership(self):
        row = dict(id="a", label=1, image_sha256="hash")
        with self.assertRaises(ValueError):
            list(paired_rows([row], [dict(row, id="b")]))
        with self.assertRaises(ValueError):
            list(paired_rows([row], [dict(row, label=0)]))

    def test_changed_asset_or_freeze_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            asset = root / "asset"
            asset.write_text("original")
            frozen = root / "freeze.json"
            save(frozen, dict(hashes={str(asset): sha(asset)}))
            save(root / "freeze-receipt.json", dict(sha256=sha(frozen)))
            with patch.object(final, "OUT", root):
                final.verify_freeze(frozen)
                asset.write_text("changed")
                with self.assertRaises(ValueError):
                    final.verify_freeze(frozen)
                asset.write_text("original")
                save(frozen, dict(hashes={str(asset): sha(asset)}, changed=True))
                with self.assertRaises(ValueError):
                    final.verify_freeze(frozen)


if __name__ == "__main__":
    unittest.main()
