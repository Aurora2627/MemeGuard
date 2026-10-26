# 版本历史

v0.1.0：实验基础与验证；完整实验成绩待后续版本记录。

v0.2.0：首轮结果与固定预算 SFT；第二轮成绩待后续版本记录。

v0.2.1：VS Code 环境关联配置。

v0.3.0：完整训练结果、阈值诊断与额外验证流程。


## v0.4.0 - 2026-10-24

Add controlled binary-label objective experiment, retaining answer-LM baseline; 16 unit tests passed. Archive 372-row validation results. Same 1024 training examples and 128 updates; automatic 128-row threshold selection and 372-row reused-development comparison. New training results pending.


## v0.5.0 - 2026-10-25

Archive completed binary-objective training results; add fixed-adapter modality ablations, label-leakage/input-removal tests, and a Chinese experiment report. Ablation outcomes pending.


## v0.5.1 - 2026-10-25

Complete four-mode input ablation; publish aggregate metrics and paired 2000-resample AUROC intervals. Verify exact joint-score reproduction. No stable joint-vs-image-only gain established.


## v0.6.0 - 2026-10-26

Add 4096-example data scaling pipeline (one epoch, 512 updates), original-prefix inclusion guard, expanded train/all-dev overlap audit, pre-specified model-selection protocol, and local error review inventory. Semantic causes remain unreviewed; scaling results pending.


## v0.6.1 - 2026-10-26

Add atomic complete training checkpoints and guarded resume; 24 tests passed including CPU next-update equivalence. Preserve interrupted scaling run; restart from base using cached data because historical adapters lack optimizer state.
