"""Inspect development errors, then freeze the pre-specified final comparison."""

from pathlib import Path
from collections import Counter
import html, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from memeguard.common import read, save, sha, now

DATA = Path("D:/career-projects/data/MemeGuard")
RUNS = Path("D:/career-projects/runs/MemeGuard")
SCALE = RUNS / "20261026-scale-resume112-v1"
OUT = RUNS / "20261027-final-test-v1"
REPORT = ROOT / "reports/final-test-v1"


def paired_rows(old, new):
    previous = {r["id"]: r for r in old}
    if (
        len(previous) != len(old)
        or len({r["id"] for r in new}) != len(new)
        or set(previous) != {r["id"] for r in new}
    ):
        raise ValueError("Prediction membership mismatch")
    for row in new:
        before = previous[row["id"]]
        if (row["label"], row["image_sha256"]) != (
            before["label"],
            before["image_sha256"],
        ):
            raise ValueError("Prediction identity mismatch")
        yield before, row


def analyze():
    old = read(RUNS / "20261024-validation-v1/sft/predictions.json")
    new = read(SCALE / "development-predictions.json")
    old_t = read(ROOT / "configs/thresholds-v1.json")["sft"]["threshold"]
    new_t = read(SCALE / "frozen-threshold.json")["selection"]["threshold"]
    manifest = {r["id"]: r for r in read(DATA / "validation-v1.json")}
    counts = Counter()
    changed = []
    new_fn = []
    for a, b in paired_rows(old, new):
        label = b["label"]
        before = int(a["p_hateful_conditional"] >= old_t)
        after = int(b["p_hateful_conditional"] >= new_t)
        counts[f"label{label}:{before}->{after}"] += 1
        if before != after:
            item = dict(
                id=b["id"],
                label=label,
                old_score=a["p_hateful_conditional"],
                new_score=b["p_hateful_conditional"],
                old_prediction=before,
                new_prediction=after,
            )
            changed.append(item)
            if label == 1 and after == 0:
                new_fn.append(item)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    summary = dict(
        n=len(new),
        old_threshold=old_t,
        new_threshold=new_t,
        transitions=dict(counts),
        new_false_negatives=len(new_fn),
        changed_predictions=len(changed),
        semantic_review="pending visual inspection; transitions alone do not identify causes",
    )
    save(REPORT / "development-transitions.json", summary)
    save(OUT / "changed-development-cases.json", changed)
    cards = []
    for item in sorted(new_fn, key=lambda x: x["new_score"]):
        row = manifest[item["id"]]
        cards.append(
            "<article><h3>"
            + html.escape(str(item))
            + '</h3><img style="max-width:550px;max-height:500px" src="'
            + Path(row["image"]).as_uri()
            + '"><p>'
            + html.escape(row["text"])
            + "</p></article>"
        )
    (OUT / "new-false-negatives.html").write_text(
        '<meta charset="utf-8"><h1>New development false negatives</h1>'
        + "".join(cards),
        encoding="utf-8",
    )
    from PIL import Image, ImageOps, ImageDraw, ImageFont

    for page in range((len(new_fn) + 3) // 4):
        canvas = Image.new("RGB", (1200, 1000), "white")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
        for slot, item in enumerate(
            sorted(new_fn, key=lambda x: x["new_score"])[page * 4 : page * 4 + 4]
        ):
            row = manifest[item["id"]]
            x = (slot % 2) * 600
            y = (slot // 2) * 500
            with Image.open(row["image"]) as im:
                thumbnail = ImageOps.contain(im.convert("RGB"), (570, 370))
            canvas.paste(thumbnail, (x + (600 - thumbnail.width) // 2, y + 35))
            draw.text(
                (x + 10, y + 5),
                f"ID {item['id']} old={item['old_score']:.3f} new={item['new_score']:.3f}",
                fill="black",
                font=font,
            )
            import textwrap

            draw.multiline_text(
                (x + 10, y + 410),
                "\n".join(textwrap.wrap(row["text"], 55))[:320],
                fill="black",
                font=font,
            )
        canvas.save(OUT / f"new-fn-page-{page+1}.jpg")
    print(json.dumps(summary, indent=2), flush=True)


def freeze():
    path = REPORT / "freeze.json"
    if path.exists():
        print("Existing freeze retained", sha(path))
        return
    if read(SCALE / "status.json")["status"] != "complete":
        raise ValueError("Scaling run incomplete")
    adapter = SCALE / "training/adapter"
    files = [
        ROOT / "configs/sources.lock.json",
        ROOT / "configs/thresholds-v1.json",
        ROOT / "configs/sft-v3-restart.json",
        SCALE / "frozen-threshold.json",
        adapter / "adapter_config.json",
        adapter / "adapter_model.safetensors",
        DATA / "scale-v1.json",
        DATA / "scale-and-all-dev-v1.json",
        ROOT / "scripts/final_test.py",
        ROOT / "scripts/prepare_final.py",
        *sorted((ROOT / "src/memeguard").glob("*.py")),
    ]
    lock = read(ROOT / "configs/sources.lock.json")
    files.append(Path(lock["model"]["local_root"]) / "identity.json")
    record = dict(
        created=now(),
        selection="Highest reused-development AUROC among baseline, SFT-v1, SFT-v2 and scaling-v1, as pre-specified",
        candidate_auroc=dict(
            baseline=0.6584958783234646,
            sft_v1=0.6821374666202252,
            sft_v2=0.67805932893266,
            scaling_v1=0.7279983745500986,
        ),
        adapter=str(adapter),
        thresholds=dict(
            baseline=read(ROOT / "configs/thresholds-v1.json")["baseline"]["threshold"],
            selected=read(SCALE / "frozen-threshold.json")["selection"]["threshold"],
        ),
        hashes={str(p): sha(p) for p in files},
        test_split="test_seen",
        expected_test_rows=1000,
        protocol="Acquire full pinned split; audit exact pair leakage and image reuse against 4096 train + 500 dev; evaluate baseline then selected once. No post-test tuning.",
        limitations="Upstream annotations were previously downloaded for schema/count audit. Test images and model predictions have not been used for development. Pretraining exposure is unknown; near-duplicate detection is not exhaustive.",
    )
    with path.open("x", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    save(
        OUT / "freeze-receipt.json",
        dict(freeze=str(path), sha256=sha(path), created=now()),
    )
    print("FINAL_PROTOCOL_FROZEN", sha(path), flush=True)


if __name__ == "__main__":
    analyze()
    freeze()
