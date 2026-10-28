# 更新记录

## v0.10.0 - 2026-10-28

新增本地社区图文审核工作台：FastAPI 图文接入、SQLite 持久化队列、独立 CUDA worker、冻结版本与阈值分流、追加式人工复核、授权反馈留出集审计与现有 SFT 交接。提供主模型/基线切换回滚、任务租约恢复、心跳与运行指标、VS Code 调试配置和使用文档。54 项测试通过，真实 CUDA HTTP 与浏览器上传/复核流程验证完成。固定留出集审计 1,500 条，合成测试未进入训练；没有新增人工反馈训练或线上质量指标。v0.9.0 模型与离线实验结果保持不变。证据见 reports/service-v1。

## v0.9.0 - 2026-10-27

Complete the frozen 1,000-example test_seen evaluation. Baseline vs selected AUROC 0.654448 vs 0.728864; AUPRC 0.633633 vs 0.695780. At development-selected frozen thresholds, FP 395 to 302 and FN 41 to 67; recall decreases from 91.63% to 86.33%. Archive aggregate metrics, audit and hash receipts; add prediction identity/metric verification and paired 2,000-resample analysis. Publish final report with both operating points, development-only ablation scope and limitations. No test tuning or retraining.


## v0.1.0 — 2026-10-23

新增固定数据与模型来源、CUDA 开发流程、图文数据审计、128 条开发集基线脚本、两步 QLoRA 容量测试和 VS Code 调试入口。完成 144 张图片审计与 8 项单元测试。模型仍在下载，尚无评分或正式训练结果。

## v0.2.0 — 2026-10-24

记录真实基线与容量结果；新增 1024 样本固定预算 SFT、适配器评估、后台流程和检查点，新增训练/开发角色隔离测试。第二轮效果待运行后报告。

## v0.2.1 — 2026-10-24

记录 VS Code Python Environments 的项目环境关联，远程终端已激活 memeguard-cuda。10 项单元测试通过。

## v0.3.0 — 2026-10-24

保存 1024 样本完整 QLoRA 训练曲线、资源消耗和真实开发集对照：AUROC 上升而固定阈值召回下降。新增配对 bootstrap、召回约束阈值选择及额外 372 条验证流程；明确额外验证尚未完成。新增阈值边界测试。


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

## v0.6.2 - 2026-10-26
Add a saved resume-and-evaluate runner for the complete step-112 checkpoint. Validate protocol and training-state checksum before training; preserve the old run and record a recovery receipt. Prevent automatic Windows system sleep while the runner is active, without keeping the display on. Task Scheduler launch failed on this host; the supported launcher uses the VS Code remote terminal. Training/evaluation results are pending.

## v0.7.0 - 2026-10-26
Archive completed 4096-row / 512-update training and development evaluation after complete-state recovery. AUROC 0.7280 vs 0.6821 for 1024-row SFT; paired 2000-resample bootstrap delta CI [0.0217, 0.0704]. Frozen-threshold recall 84.48%, FP112/FN27: improved ranking but 90% recall target not met. Preserve aggregate evidence, runtime, limitations and analysis source. Official test not evaluated.

## v0.8.0 - 2026-10-27
Add paired development error transitions and visual review of 12 newly introduced false negatives. Freeze the final baseline-versus-scaling model, thresholds, sources and data identities before test acquisition. Add cached test acquisition, exact-pair overlap audit and prefix-validated interrupted inference recovery. 27 tests pass. Final test acquisition/evaluation is pending; no final test metrics claimed.

## v0.8.1 - 2026-10-27
Resume interrupted frozen final-test acquisition from cached images with a manually triggered Windows Scheduled Task under the current user (S4U, limited privileges). Verify progress beyond 625/1000 with the frozen evaluator unchanged. Preserve interruption facts; no SSH disconnect survival test or final test metrics claimed.
