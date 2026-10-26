import tempfile, unittest
from pathlib import Path
import torch
from memeguard.checkpoints import save_checkpoint, read_checkpoint, restore_state


class Toy(torch.nn.Linear):
    def save_pretrained(self, path, **kwargs):
        torch.save(self.state_dict(), Path(path) / "model.pt")


class CheckpointTests(unittest.TestCase):
    def test_adapter_only_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "Adapter-only"):
                read_checkpoint(d, {}, "x")

    def test_roundtrip_preserves_optimizer_scheduler_and_rng(self):
        torch.manual_seed(17)
        model = Toy(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        def update(m, o, s):
            o.zero_grad()
            loss = m(torch.rand(3, 2)).square().mean()
            loss.backward()
            o.step()
            s.step()

        update(model, optimizer, scheduler)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "checkpoint"
            save_checkpoint(
                path, model, optimizer, scheduler, {"seed": 17}, "manifest", 8, 1, [0.2]
            )
            update(model, optimizer, scheduler)
            expected = {k: v.clone() for k, v in model.state_dict().items()}
            restored = Toy(2, 1)
            restored.load_state_dict(torch.load(path / "model.pt", weights_only=True))
            opt = torch.optim.AdamW(restored.parameters(), lr=0.01)
            sch = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.9)
            state = read_checkpoint(path, {"seed": 17}, "manifest")
            restore_state(state, opt, sch)
            update(restored, opt, sch)
            for k, v in restored.state_dict().items():
                self.assertTrue(torch.equal(v, expected[k]))
            self.assertEqual(scheduler.state_dict(), sch.state_dict())
            with self.assertRaisesRegex(ValueError, "mismatch"):
                read_checkpoint(path, {"seed": 18}, "manifest")

    def test_corrupt_state_rejected(self):
        model = Toy(1, 1)
        o = torch.optim.AdamW(model.parameters())
        s = torch.optim.lr_scheduler.StepLR(o, 1)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "checkpoint"
            save_checkpoint(path, model, o, s, {}, "m", 0, 0, [])
            with (path / "training-state.pt").open("ab") as f:
                f.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_checkpoint(path, {}, "m")
