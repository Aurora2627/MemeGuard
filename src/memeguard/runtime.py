"""Native PyTorch/Transformers inference on the project's isolated CUDA runtime."""

from pathlib import Path
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
)
from .common import ROOT, read, sha

PROMPT = (
    "Classify whether this meme is hateful. Consider the image and its caption together. "
    "Hateful content attacks or demeans people based on protected characteristics; "
    "general offensiveness or profanity alone is not sufficient. "
    "Return 1 for hateful or 0 for benign, and nothing else. Caption: "
)


def instruction(row):
    # Ground-truth labels never enter the model input.
    return PROMPT + row["text"]


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    free, total = torch.cuda.mem_get_info()
    if free < 6 * 1024**3:
        raise RuntimeError("Less than 6 GiB free GPU memory; check other experiments")
    model_info = read(ROOT / "configs/sources.lock.json")["model"]
    directory = Path(model_info["local_root"])
    identity = read(directory / "identity.json")
    if identity["revision"] != model_info["revision"]:
        raise ValueError("Model revision mismatch")
    for record in identity["files"]:
        if sha(directory / record["file"]) != record["sha256"]:
            raise ValueError("Changed model asset")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(
        str(directory),
        local_files_only=True,
        min_pixels=4 * 28 * 28,
        max_pixels=128 * 28 * 28,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(directory),
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
    )
    return model, processor, identity


def encode(processor, row, answer=None):
    with Image.open(row["image"]) as source:
        image = source.convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction(row)}],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    if answer is not None:
        prompt_length = inputs["input_ids"].shape[-1]
        messages.append({"role": "assistant", "content": str(answer)})
        full = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        complete = processor(text=[full], images=[image], return_tensors="pt")
        if not torch.equal(
            complete["input_ids"][:, :prompt_length], inputs["input_ids"]
        ):
            raise ValueError("Training prompt prefix differs from evaluation input")
        labels = complete["input_ids"].clone()
        labels[:, :prompt_length] = -100
        if (labels != -100).sum() < 1:
            raise ValueError("No supervised answer tokens")
        complete["labels"] = labels
        inputs = complete
    if inputs["input_ids"].shape[-1] > 768:
        raise ValueError("Input exceeds reviewed capacity profile")
    return {key: value.to("cuda") for key, value in inputs.items()}


def label_ids(processor):
    ids = [
        processor.tokenizer.encode(value, add_special_tokens=False)
        for value in ("0", "1")
    ]
    if any(len(value) != 1 for value in ids) or ids[0] == ids[1]:
        raise ValueError("Binary label tokens are not distinct single tokens")
    return [value[0] for value in ids]
