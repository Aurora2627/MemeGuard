"""Build local review inventory; semantic causes remain unassigned until inspection."""

from pathlib import Path
import sys, json, csv, html, hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from memeguard.common import read, save


def main():
    run = Path("D:/career-projects/runs/MemeGuard/20261025-ablations-v1")
    output = Path("D:/career-projects/runs/MemeGuard/20261026-error-review-v1")
    output.mkdir(parents=True, exist_ok=False)
    rows = {
        r["id"]: r for r in read("D:/career-projects/data/MemeGuard/validation-v1.json")
    }
    predictions = {
        m: {r["id"]: r for r in read(run / (m + "-development-predictions.json"))}
        for m in ("joint", "text_only", "image_only")
    }
    thresholds = {
        m: read(run / (m + "-threshold.json"))["selection"]["threshold"]
        for m in predictions
    }
    errors = []
    counts = {"fp": 0, "fn": 0}
    joint_correct_single_wrong = 0
    for ident, row in rows.items():
        verdict = {
            m: int(predictions[m][ident]["p_hateful_conditional"] >= thresholds[m])
            for m in predictions
        }
        if verdict["joint"] == row["label"]:
            joint_correct_single_wrong += int(
                verdict["text_only"] != row["label"]
                and verdict["image_only"] != row["label"]
            )
            continue
        category = "fp" if verdict["joint"] == 1 else "fn"
        counts[category] += 1
        errors.append(
            dict(
                id=ident,
                error_type=category,
                label=row["label"],
                score=predictions["joint"][ident]["p_hateful_conditional"],
                image=row["image"],
                text=row["text"],
                cause="unreviewed",
                review_notes="",
            )
        )
    selected = []
    for category in ("fp", "fn"):
        subset = sorted(
            (r for r in errors if r["error_type"] == category),
            key=lambda r: hashlib.sha256(("review-v1:" + r["id"]).encode()).hexdigest(),
        )
        selected.extend(subset[:24])
    with (output / "review.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    cards = []
    for r in selected:
        uri = Path(r["image"]).as_uri()
        cards.append(
            "<article><h2>"
            + html.escape(r["id"] + " " + r["error_type"])
            + '</h2><img src="'
            + html.escape(uri, quote=True)
            + '"><p>'
            + html.escape(r["text"])
            + "</p><p>label="
            + str(r["label"])
            + " score="
            + str(round(r["score"], 4))
            + "</p><p>Cause: unreviewed</p></article>"
        )
    (output / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>MemeGuard error review</title><style>body{font-family:sans-serif;max-width:1000px;margin:auto}article{border:1px solid #ccc;padding:16px;margin:16px}img{max-width:450px;max-height:400px}</style><h1>Local development error review</h1><p>Semantic causes are unreviewed. Suggested categories: general offense vs protected-group attack; irony/context; OCR; image-text relation; ambiguous label. Do not infer causes from score alone.</p>'
        + "".join(cards),
        encoding="utf-8",
    )
    save(
        ROOT / "reports/error-review-v1/inventory.json",
        dict(
            scope="Reused development data, first SFT adapter, frozen threshold",
            threshold=thresholds["joint"],
            counts=counts,
            review_selected=len(selected),
            joint_correct_both_single_modes_wrong=joint_correct_single_wrong,
            semantic_causes="unreviewed",
            private_review=str(output / "review.html"),
        ),
    )
    print(
        "ERROR_INVENTORY_READY",
        counts,
        "selected",
        len(selected),
        "joint_correct_both_single_wrong",
        joint_correct_single_wrong,
    )


if __name__ == "__main__":
    main()
