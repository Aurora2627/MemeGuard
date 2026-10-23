# MemeGuard：面向图文内容审核的多模态风险识别

基于 Qwen2-VL-2B 的图文联合分类实验，面向社区帖子、商品宣传图等内容进入人工审核前的风险筛查。当前使用英文 Hateful Memes 数据开发，目标是验证图文理解、监督微调与检索增强的方法；尚不能代表中文电商违规识别或线上审核效果。

## 实现范围

- 固定模型、数据与参考代码版本，记录下载校验信息。
- 对开发样本进行图片解码、标签检查及跨划分图文对重复检查。
- 使用模型最后一个位置的 0/1 标签 logits 计算条件风险分数，统计 AUROC、AUPRC、F1、混淆矩阵、耗时和显存。
- 使用 NF4 4-bit、LoRA 与梯度检查点进行训练容量测试，检查梯度和参数实际更新。

当前首轮包含 128 条验证集零样本评估，以及 2 条训练样本的两步容量测试。两步容量测试不是完整训练，不能用来声称微调有效。正式 SFT、检索对照及独立测试尚未完成。

## 运行环境

Windows，Python 3.11.9，PyTorch 2.6.0 / CUDA 12.4，torchvision 0.21.0，Transformers 4.53.2，PEFT 0.17.1，bitsandbytes 0.48.1。目标显卡为 RTX 4060 Laptop 8GB；训练可行性以实际容量测试为准。依赖版本见 requirements-runtime.txt，PyTorch 使用官方 cu124 发行包。

VS Code 已提供 Python 调试入口。实验输出目录必须使用新目录，以保留历史结果。原始数据、权重与适配器位于项目外，不进入 Git。

## 目录

- src/memeguard：数据审计、指标与模型输入构造。
- scripts：资源准备、基线评估与训练容量测试。
- configs/sources.lock.json：模型、数据及参考仓库的固定版本。
- tests：数据边界、指标与提示词标签隔离检查。
- docs/experiment-plan.md：评估协议和后续实验。
- vendor/RGCL：参考实现的固定 Git 子模块，不自动运行其中脚本。

## 方法与来源

参考 [RA-HMD / RGCL](https://github.com/JingbiaoMei/RGCL) 与 [EMNLP 2025 论文](https://aclanthology.org/2025.emnlp-main.1215/)，本仓库开发独立的 Transformers/PEFT 实验流程。当前没有实现论文完整的联合损失与第二阶段检索训练，不声称复现论文结果。

模型来自 [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)，数据来自 [Jingbiao/RA-HMD](https://huggingface.co/datasets/Jingbiao/RA-HMD)。参考代码及数据分别遵循各自许可证，详见 NOTICE.md。
