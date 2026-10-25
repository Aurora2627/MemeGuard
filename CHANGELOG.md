# 更新记录

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
