# MemeGuard：面向图文内容审核的多模态风险识别

基于 Qwen2-VL-2B 的图文联合分类实验，面向社区帖子、商品宣传图等内容进入人工审核前的风险筛查。当前使用英文 Hateful Memes 数据开发，目标是验证图文理解、监督微调与检索增强的方法；尚不能代表中文电商违规识别或线上审核效果。

## 实现范围

- 固定模型、数据与参考代码版本，记录下载校验信息。
- 对开发样本进行图片解码、标签检查及跨划分图文对重复检查。
- 使用模型最后一个位置的 0/1 标签 logits 计算条件风险分数，统计 AUROC、AUPRC、F1、混淆矩阵、耗时和显存。
- 使用 NF4 4-bit、LoRA 与梯度检查点进行训练容量测试，检查梯度和参数实际更新。

首轮 128 条开发样本基线已完成：AUROC 0.6387、AUPRC 0.6616、F1 0.7263、准确率 0.5938；TP/FP/FN/TN 为 69/48/4/7，误报较高。两步 QLoRA 容量测试完成真实参数更新，峰值显存预留约 3.24 GiB，但不是完整训练成果。结果文件见 reports/first-phase。

第二轮固定 1024 条训练样本、1 epoch、rank=8、alpha=16、学习率 1e-4、梯度累积 8、seed=17，共 128 次优化器更新，视觉编码器冻结。第二轮已完成：1024 条训练样本、128 次优化器更新，训练约 35.9 分钟，峰值预留显存 3.27 GiB。在相同 128 条开发样本上，AUROC 0.6387→0.6841，AUPRC 0.6616→0.7091；固定 0.5 阈值时，误报 48→9，但漏报 4→39，F1 0.7263→0.5862，召回率 0.9452→0.4658，不能声称整体审核效果提升。

在原 128 条开发样本上以召回率至少 90% 为约束选择并冻结阈值：baseline=0.5312094092，SFT=0.1192029193。该步骤属于开发集阈值选择，不是独立泛化成绩。其余 372 条 dev_seen 样本用于后续对照，官方 test_seen 仍保留。完整训练结果见 reports/sft-v1，协议见 docs/experiment-plan.md。

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


## Second training round

The first adapter scored AUROC 0.6821 versus baseline 0.6585 on 372 additional development records, but at frozen thresholds both recalled 90.80%, with 156 versus 155 false positives. Thus no high-recall operational improvement was established. Round two completed: AUROC 0.6781, frozen-threshold recall 92.53%, 157 false positives and 13 false negatives on reused development data. These 372 examples now serve as reused development data, not untouched validation. See [protocol](docs/round-two.md).


## 完整实验报告与输入消融

[实验报告](docs/experiment-report.zh-CN.md)记录两轮训练结果、数据划分和四组输入消融协议。第二轮未明显降低高召回误报；消融实验用于定位模态依赖，结果待完成。
