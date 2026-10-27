# MemeGuard 图文内容风险识别

基于 Qwen2-VL-2B 的英文图文仇恨内容识别研究原型，面向社区内容初筛与人工复核。输入图片与配文，输出二分类条件分数、阈值判定和批量评估结果。

## 最终结果

2026-10-27 完成 4,096 条样本微调与 1,000 条官方 test_seen 最终评估，模型和阈值在测试图片获取及推理前冻结。

| 指标 | 原始模型 | 微调模型 |
|---|---:|---:|
| AUROC | 0.6544 | 0.7289 |
| AUPRC | 0.6336 | 0.6958 |
| F1 | 0.6732 | 0.6963 |
| 召回率 | 91.63% | 86.33% |
| FP / FN | 395 / 41 | 302 / 67 |

分类指标采用各自提前在开发集确定的阈值。误报减少 93 条，但漏报增加 26 条，不能声称保持高召回的误报优化。默认 0.5 阈值下微调模型召回率仅 43.27%，不可当成默认业务阈值。分数未经概率校准。

完整协议、两种阈值、配对置信区间与限制见[最终报告](docs/final-validation.zh-CN.md)，聚合证据见[结果文件](reports/final-test-v1)。

## 方法与实现

- PyTorch / Transformers / PEFT 训练流程，NF4 双重量化、BF16、SDPA、梯度累积和梯度检查点。
- 冻结视觉编码器，注意力 q/v 投影 LoRA，rank=8、alpha=16，1,089,536 个可训练参数。
- 4,096 条训练样本、1 epoch、512 次更新；完整状态检查点与协议验证，支持训练恢复。
- 500 条开发数据用于阈值及模型选择；最终测试 1,000 条，不用测试结果重新调参。
- 跨划分精确图像与配文重复审计、输入消融、漏报案例复核和预测续跑。

## 实验记录

- [早期训练与输入消融](docs/experiment-report.zh-CN.md)：1,024 条适配器的四组消融，完整图文优于仅图片的增益尚不稳定。
- [扩展训练](docs/scaling-results.zh-CN.md)：4,096 条、512 次更新和开发集比较。数据量与计算量同时增加，不能视为固定计算量消融。
- [最终报告](docs/final-validation.zh-CN.md)：冻结评估、1,000 条测试结果及开发漏报分析。
- [恢复说明](docs/recovery.md)：完整检查点与实际验证范围。
- [版本记录](CHANGELOG.md)：保留每轮实验及中断事实。

## 运行环境

Windows，Python 3.11.9，PyTorch 2.6.0 / CUDA 12.4，torchvision 0.21.0，Transformers 4.53.2，PEFT 0.17.1，bitsandbytes 0.48.1。使用 RTX 4060 Laptop 8GB 完成训练和推理，训练峰值 PyTorch 预留约 3.27 GiB。锁定依赖见 requirements-runtime.txt，PyTorch 使用官方 cu124 发行包。

VS Code 提供 Python 调试入口。原始数据、模型权重、适配器与逐条预测保存在项目外；新实验使用独立输出目录，保留历史结果。

## 代码入口

src/memeguard 包含数据审计、模型输入和指标计算；scripts 包含数据准备、训练、评估及报告分析。scripts/final_test.py 是冻结最终测试入口；scripts/analyze_final_test.py 复核已保存预测并生成聚合统计，不重新训练或调参。configs/sources.lock.json 固定模型、数据和参考代码版本。

## 来源与适用范围

模型来自 [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)，数据来自 [RA-HMD](https://huggingface.co/datasets/Jingbiao/RA-HMD)，参考 [RGCL](https://github.com/JingbiaoMei/RGCL) 与其[论文](https://aclanthology.org/2025.emnlp-main.1215/)。本仓库实现独立的 Transformers/PEFT 实验流程，未实现论文完整联合损失和第二阶段检索训练，不声称达到论文结果。

参考代码及数据遵循各自许可，见 NOTICE.md。未验证中文内容、线上吞吐或真实审核成本；精确重复审计不排除近重复或预训练暴露。
