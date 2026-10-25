"""Input interventions for development-only modality diagnostics."""

from PIL import Image
from .runtime import PROMPT, encode

MODES = ("joint", "text_only", "image_only", "shuffled_image")


def messages_for(row, mode):
    if mode not in MODES:
        raise ValueError("Unknown input mode")
    # Keep the task wording constant: only available input content changes.
    text = PROMPT + ("" if mode == "image_only" else row["text"])
    content = [] if mode == "text_only" else [{"type": "image"}]
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def rotated_images(rows):
    """Deterministic non-self image assignment; independent of labels."""
    ordered = sorted(rows, key=lambda row: str(row["id"]))
    if len(ordered) < 2 or len({r["id"] for r in ordered}) != len(ordered):
        raise ValueError("Need at least two unique examples")
    return {
        row["id"]: ordered[(i + 1) % len(ordered)]["image"]
        for i, row in enumerate(ordered)
    }


def encode_ablation(processor, row, mode, replacement=None):
    if mode == "joint":
        return encode(processor, row)
    messages = messages_for(row, mode)
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    kwargs = dict(text=[prompt], return_tensors="pt")
    if mode != "text_only":
        path = replacement if mode == "shuffled_image" else row["image"]
        if path is None:
            raise ValueError("Missing replacement image")
        with Image.open(path) as source:
            kwargs["images"] = [source.convert("RGB")]
    inputs = processor(**kwargs)
    if inputs["input_ids"].shape[-1] > 768:
        raise ValueError("Input exceeds reviewed capacity")
    return {key: value.to("cuda") for key, value in inputs.items()}
