import unittest
import torch
from memeguard.objectives import binary_label_loss


class BinaryObjectiveTests(unittest.TestCase):
    def test_matches_inference_probability(self):
        x = torch.tensor([[9.0, 1.0, -3.0, 2.0]], requires_grad=True)
        loss = binary_label_loss(x, [1, 3], 1)
        self.assertTrue(
            torch.allclose(loss, -torch.softmax(x[:, [1, 3]], dim=-1)[0, 1].log())
        )
        loss.backward()
        self.assertEqual(x.grad[0, 0].item(), 0.0)
        self.assertLess(x.grad[0, 3].item(), 0.0)
        self.assertGreater(x.grad[0, 1].item(), 0.0)

    def test_ignores_other_vocabulary_logits(self):
        a = torch.tensor([[0.0, 1.0, 2.0]])
        b = torch.tensor([[100.0, 1.0, 2.0]])
        self.assertEqual(
            binary_label_loss(a, [1, 2], 0).item(),
            binary_label_loss(b, [1, 2], 0).item(),
        )

    def test_invalid_labels(self):
        with self.assertRaises(ValueError):
            binary_label_loss(torch.zeros(1, 3), [1, 2], 2)
