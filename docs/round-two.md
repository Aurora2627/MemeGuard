# Round two: inference-aligned binary objective

Only the objective changes from answer-token language modeling (including ending tokens) to cross entropy on the two label logits at the inference prompt boundary. The 1024 training rows, seed 17, 128 updates, optimizer, LoRA rank, learning rate and visual resolution stay fixed. Start from the same base model, not the previous adapter. No class balancing is added in this round, to isolate the objective effect.

Evaluate on the original 128 development rows to select a recall >= 0.90 threshold, then apply it unchanged to the remaining 372 development rows. Because results on those 372 rows informed this next experiment, they are now a reused development comparison, not an untouched holdout. Report AUROC, AUPRC, fixed-0.5 confusion counts and frozen-threshold confusion counts, including negative results. Official test remains untouched. Do not select checkpoints using these 372 labels.

The previous SFT improved 372-row AUROC from 0.65850 to 0.68214, but at frozen thresholds produced 156 false positives versus 155 for baseline at identical recall 0.90805. This motivates the experiment; it does not prove the objective caused the error pattern.
