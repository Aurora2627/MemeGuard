"""Training losses aligned with binary label-token inference."""

import torch
import torch.nn.functional as F


def binary_label_loss(next_token_logits, token_ids, label):
    """Cross entropy over only the two label logits; no EOS supervision."""
    if len(token_ids) != 2 or len(set(token_ids)) != 2 or label not in (0, 1):
        raise ValueError("Expected two distinct token IDs and a binary target")
    logits = next_token_logits[:, token_ids].float()
    target = torch.full(
        (logits.shape[0],), label, dtype=torch.long, device=logits.device
    )
    return F.cross_entropy(logits, target)
