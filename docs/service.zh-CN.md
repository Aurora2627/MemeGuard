# MemeGuard 图文审核工作台

## 场景和范围

面向社区英文图文投稿的人工审核辅助：图片与配文进入持久化队列，Qwen2-VL 为复核排序，审核者查看图文、记录判断和修订历史。经明确授权的人工标签经过固定开发/测试集隔离检查后，导出到已有 PyTorch 微调入口。

本系统是本地单用户工程原型。当前没有真实社区流量、多人鉴权、线上效果实验或自动处罚。中文、色情、暴力等其他类型未验证。风险分数是两个标签 token 的条件 softmax，未经概率校准；常规队列不表示安全。

## 实际运行环境

Windows 11，RTX 4060 Laptop 8 GiB，Python 3.11.9，PyTorch 2.6.0+cu124，Transformers 4.53.2，PEFT 0.17.1，bitsandbytes 0.48.1，FastAPI 0.162.0，Uvicorn 0.54.0，Pydantic 2.14.1，HTTPX 0.28.1。模型只从 Windows 本地加载，NF4 双重量化、BF16、SDPA；视觉编码器冻结，加载已训练 q/v LoRA 适配器。

服务依赖见 requirements-service.txt，训练依赖保持原有锁定文件。路径配置见 configs/service-models.json、configs/service-feedback.json，运行状态可通过 MEMEGUARD_STATE 指定，默认 D:/career-projects/services/MemeGuard。

## 启动和 VS Code 调试

在 Windows 的 VS Code Remote SSH 终端进入 D:/career-projects/MemeGuard，使用项目解释器：

~~~powershell
& D:/career-projects/envs/memeguard/Scripts/python.exe scripts/start_service.py
~~~

首次加载会校验基础权重与适配器，网页先显示加载中。Windows 本机打开 http://127.0.0.1:8787；Mac 通过 VS Code 的“端口”转发 8787 后访问该地址。只监听回环地址，未开放公网。

VS Code 调试列表提供“审核工作台 API”和“审核工作台 CUDA worker”。调试 API 前先停止已有 web 进程，避免端口冲突；调试 worker 前先停止已有 worker，独占锁会阻止两个进程同时加载模型。启动器只补启动缺失的组件，worker 心跳保留加载中、空闲、推理中、停止状态。

~~~powershell
& D:/career-projects/envs/memeguard/Scripts/python.exe scripts/stop_service_worker.py
~~~

以上请求 worker 完成本条任务后退出。确认 /api/metrics 中相应 worker 为 stopped 且 nvidia-smi 显存释放后，才能训练。停止请求不会清除队列；再次运行 start_service.py 会清除停止请求并启动 worker。首次加载中停止需等加载完成。

如需关闭 API，在运行它的调试终端按 Ctrl+C；后台启动的进程身份保存在 last-start.json，关闭前核对脚本路径，勿结束所有 Python 进程。

## 使用流程

1. 选择 PNG、JPEG 或 WebP，单图不超过 5 MiB，每批最多 10 张，填写英文配文。本批图片使用同一段配文。未勾选训练授权仍可进行推理和复核。
2. 提交后立即进入 SQLite 队列。后台单个 GPU worker 执行推理，页面每 6 秒刷新。可筛选优先复核、常规复核、等待或异常任务。
3. 选择内容，查看模型版本、冻结阈值、条件分数和原图。图文完全重复会沿用原记录，不更新首次提交的授权状态；不同配文视为不同内容。
4. 填写复核人、人工结论和依据。更正会追加新版本，不覆盖旧标签；并发修订冲突返回 409，刷新后重试。复核人是本地自报代号，不是经过认证的用户身份。
5. 切换主模型或基线只影响后续新任务。已入队任务保留当时的版本、指纹和阈值。重新推理创建新任务，旧结果仍保存。
6. 明确需要用于训练的数据，必须同时具备提交授权与复核授权。点击反馈导出，等待逐图校验固定开发/测试集。页面提供审计、数据清单和训练配置下载。

## 结构与可靠性

~~~text
浏览器 / HTTP 批量接口
          │
          ▼
FastAPI ── SQLite WAL ── 单 GPU worker ── 冻结 Qwen2-VL + LoRA
   │       │    │                │
   │       │    └─ 任务租约/心跳/失败记录/有界重试
   │       └─ 追加式复核与审计事件
   ▼
授权反馈 → 固定留出集隔离 → 不可变导出 → 原 PyTorch SFT 入口
~~~

- API 和模型进程分离，网页操作不会重复加载权重。
- BEGIN IMMEDIATE 原子认领，避免两个消费者拿同一任务；独占 worker 锁限制显存占用。
- worker 每 10 秒心跳并续租；任务租约 300 秒。租约失效后重新认领，最多 3 次；旧 worker 的迟到结果不能覆盖新任务。
- 请求幂等键防止网络重发，规范化图像与配文哈希防止完全重复提交。队列容量 1,000 条。
- 上传解码成 RGB PNG，去除原元数据，限制文件大小、像素和帧数。媒体按内容哈希保存。
- 写接口检查会话 token、Origin 与 Host；没有公网部署所需的登录、角色权限和 TLS。这些检查不构成多人服务鉴权。
- SQLite、图片、反馈和完整运行日志保存在仓库外。Git 只发布源码、配置、聚合证据和合成样例验证结果。

## 数据闭环的边界

导出只取每条内容的最新人工结论，禁止 demo / integration_test 来源进入训练；模型预测不生成训练标签。固定留出集文件及图片必须存在、与清单哈希一致，否则导出失败。按解码后像素检查相同图片，修改配文也不能绕过；当前尚不识别裁剪、压缩等近重复。

同一图片的不同配文被分到同一侧，稳定留出约 20% 为新反馈开发集，其他进入训练。训练入口最低要求 32 条训练数据且每类至少 5 条，这只是防止空任务的工程门槛，不是充分样本量证明。导出同时记录 manifest 与 training-config 的 SHA-256，修改后拒绝启动。

~~~powershell
& D:/career-projects/envs/memeguard/Scripts/python.exe scripts/train_service_feedback.py --export D:/career-projects/services/MemeGuard/exports/导出编号 --output D:/career-projects/runs/MemeGuard/新的训练目录 --validate-only
~~~

检查通过、释放 GPU 后移除 --validate-only 才会进入既有 PyTorch SFT。不得覆盖当前发布适配器，不自动提升新模型。需要独立的新开发集评估、阈值确定和新的未使用测试集，审核后才能添加发布版本。已经查看过的 1,000 条测试数据不能作为反复优化的新测试集。

目前没有采集真实人工反馈，因此没有声称新一轮反馈训练、线上准确率或收益。已有 4,096 条训练和 1,000 条最终测试结果属于 v0.9.0 离线实验；服务化没有改变这些指标。

## 接口和监控

| 入口 | 内容 |
|---|---|
| GET /healthz | 数据库可访问性，不表示模型就绪 |
| GET /readyz | GPU worker 最近心跳且已加载；未就绪返回 503 |
| GET /api/session | 本地会话写 token |
| POST /api/submissions | items 图文批次、idempotency_key；image_base64、text、consent、source |
| GET /api/submissions | 状态/分流筛选、分页 |
| GET /api/submissions/{id} | 所有任务与复核历史 |
| POST /api/submissions/{id}/retry | 使用当前模型新建任务 |
| POST /api/submissions/{id}/reviews | job_id、expected_version、verdict、reviewer、note、consent |
| POST /api/releases/active | 发布切换与回滚审计 |
| POST /api/feedback/export | 留出集审计与不可变导出 |
| GET /api/metrics、/metrics | JSON / Prometheus 格式监控 |

写请求头为 X-MemeGuard-Token。反馈导出返回的目录仅允许下载 audit.json、manifest.json、training-config.json。

监控包括队列状态和年龄、任务失败数、worker 心跳、近期 P50/P95、分数分布与复核分歧。推理耗时不含排队、首次模型加载和权重哈希校验。少量合成请求的分位数不能当作吞吐压测；优先复核造成采样偏差，复核分歧不能当作总体准确率。

## 验证和证据

~~~powershell
& D:/career-projects/envs/memeguard/Scripts/python.exe -m unittest discover -s tests -v
& D:/career-projects/envs/memeguard/Scripts/python.exe scripts/smoke_service.py --output D:/career-projects/runs/MemeGuard/新的检查目录
~~~

单元测试验证状态机、原子认领、过期租约、幂等、复核历史与冲突、冻结阈值、发布切换、授权和留出集隔离。真实 CUDA smoke 通过 HTTP 提交合成花园图片，分别运行主模型/基线、恢复发布并校验反馈隔离；不属于质量评估或真人盲审。聚合证据见 reports/service-v1。
