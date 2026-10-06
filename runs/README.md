# 运行产物目录索引

`runs/` 保存一次执行的请求、响应、计划、指标和检查点。正式训练输入放在 `data/`。本首页和 [catalog.json](catalog.json) 会进入 Git，实际运行产物只保留本地。

## 先看这些

| 位置 | 对应任务 | 阅读入口 |
|---|---|---|
| `evaluation/base-task2-case-neutral-v1/` | 客服任务未微调 Qwen3-0.6B 的固定验证与测试 | 基础测试 50.00%、macro-F1 0.3333；三模型对照、逐题概率和独立复核；[报告](../docs/reports/base-task2-results-20261004.md) |
| `archives/autodl-20261004-164052/` | 停租前的 AutoDL 非权重完整备份 | 294 个文件、48.18 MB 压缩包及解压副本，SHA-256 全通过；四份 TensorBoard、80 组曲线、环境与源码快照；[备份入口](archives/autodl-20261004-164052/README.md) |
| `evaluation/ood-choice-v1/` | 外部规则二选一、阅读逻辑四选一的三模型对照 | `execution.json`、六份逐题报告、`comparison.json` 与 `verified-analysis.json`；[评测指南](../docs/guides/ood-choice-evaluation.md) |
| `maintenance/ood-choice-20261004/` | 外部评测检查记录 | 两端 123 项回归、服务器进程与日志；不包含模型权重 |
| `maintenance/case-neutral-1epoch-20261004/remote-results/` | 一轮大小写控制实验结果的 Mac 副本 | 原始/混合测试 96.84% / 96.08%；[一轮结果](../docs/reports/case-neutral-1epoch-results-20261004.md) |
| `maintenance/tensorboard-epochs-20261004/` | 当前三轮训练与 TensorBoard 验证 | 108 项测试、三轮样本覆盖、跨轮续训、事件读回与增量包检查；[记录](../docs/reports/tensorboard-epochs-20261004.md) |
| `training/smoke/llamafactory-direct-tensorboard-cpu-20261004/` | 当前含 TensorBoard 的四样本两步 CPU 小试 | 标量事件、检查点、标准导出与独立重载通过；不代表正式效果 |
| `maintenance/autodl-cu124-20261004/` | 三轮训练调整前的 torch 2.6 环境适配 | 保留 105 项测试和未启用 TensorBoard 时的记录；[记录](../docs/reports/autodl-cu124-20261004.md) |
| `training/smoke/llamafactory-direct-torch260-cpu-20261004/` | 此前 torch 2.6 四样本两步 CPU 小试 | 保留当时检查点、标准导出与独立重载结果 |
| `maintenance/llamafactory-20261004/` | 最初 LLaMA-Factory / torch 2.8 接入验证 | 105 项测试、续训、导出与依赖检查；[历史记录](../docs/reports/llamafactory-20261004.md) |
| `training/smoke/llamafactory-direct-cpu-20261004/` | 最初框架接入的 torch 2.8 四样本两步 CPU 小试 | 保留当时检查点、标准导出与独立重载结果 |
| `training/audits/llamafactory-direct-input-20261004.json` | 最初框架接入的真实 tokenizer 审计，当前沿用相同输入协议 | 无 HTTP、无截断，A/B ID=32/33 |
| `maintenance/modelscope-20261004/` | 迁移前 ModelScope 下载适配与独立 CPU 验证 | 98 项测试通过；不锁模型提交/哈希；[适配记录](../docs/reports/modelscope-20261004.md) |
| `training/audits/semif-direct-modelscope-latest-input-20261004.json` | 当前 ModelScope tokenizer 的完全离线输入审计 | 混合/原始/验证最长 1202/1176/1045 tokens，无截断，A/B ID=32/33 |
| `generation/bulk-openrouter-20261003-v3/` | 当前已完成全量合成运行 | `summary.json` 汇总，`completion-audit.json` 完成检查；[完成报告](../docs/reports/bulk-completion-20261003.md) |
| `training/audits/semif-direct-torch251-input-20261003.json` | 2026-10-03 tokenizers 0.22.1 / Direct chat prompt 审计 | 混合/原始/验证最长 1202/1176/1045 tokens，无截断，A/B ID=32/33 |
| `training/smoke/semif-direct-torch251-smoke-20261003/` | 历史 PyTorch 2.5.1 两步 tiny CPU 全参数小试 | `best.safetensors`，重载通过；未评测测试集，指标只验证流程 |
| `maintenance/autodl-env-20261003/` | 2026-10-03 固定环境的元数据核对与 CPU 测试 | 88 项测试、pip check、环境探针通过；Linux/4090 CUDA 尚未实测；[核定记录](../docs/reports/autodl-environment-20261003.md) |
| `training/audits/semif-direct-input-20261003.json`、`training/smoke/semif-direct-local-smoke-20261003/` | 最初 Direct 切换时的审计和 Torch 2.8 / .pt 小试 | 保留原文件；保留当时验证，不作为当前框架检查结果 |
| `maintenance/semif-direct-20261003/upstream-parity.json` | 与固定上游的独立对照 | 13 组提示字节、分词与槽位一致，微型模型原生读分一致；[迁移记录](../docs/reports/semif-direct-20261003.md) |
| `training/audits/jev-input-audit-*.json` | 历史 AgentJev 分段候选路径长度审计 | 方案已替代，不能沿用为 Direct 单 prompt 的长度结果 |
| `training/smoke/jev-local-smoke-20261003/` | 历史 AgentJev 随机候选头的小试 | 保留当时产物，不作为当前方案的验证记录 |
| `reviews/manual-review-20261002/` | 早期批次逐条人工审核工作表 | `review.md` / `manifest.json`；[审核报告](../docs/reports/manual-review-report.md) |
| `packages/autodl/` | 当前三轮训练、TensorBoard、torch 2.6 / cu124 / ModelScope 的上传包 | 完整 `customer-policy-verifier-training.tar.gz` 与增量 `customer-policy-verifier-update.tar.gz` 及清单；[AutoDL指南](../docs/guides/autodl-environment.md) |
| `packages/github/` | 适合上传GitHub的源码快照 | `customer-policy-verifier-source.tar.gz`；不包含实际数据、运行输出和密钥 |
| `maintenance/layout-20261003/` | 本次目录整理备份与校验 | `before-layout.tar.gz`、`before-files.json`、`after-audit.json` |

当前正式合成处理2397条来源，2022条提取通过，6066次生成尝试；筛选后4785条进入合成数据。仅3条助手抽检，不能把全部程序通过样本视作人工审核通过。当前采用 SemIf-OpenJev Direct，全参数训练完整预训练 Qwen3ForCausalLM。AutoDL 已完成历史三轮训练、最新一轮小写控制对比，以及规则二选一和逻辑四选一的外部评测；一轮客服测试原始/混合为 96.84% / 96.08%。最新两端各123项检查通过，历史108项测试记录保留。外部评测没有显示跨任务泛化提升，详见 [结果](../docs/reports/ood-choice-results-20261004.md)。历史 AgentJev 和最初 Direct 检查分别保留，tiny 指标只验证流程。

## 全量运行文件分别是什么

| 文件/子目录 | 用途 |
|---|---|
| `plan.json` | 预算、模型、提示版本、来源与脚本哈希 |
| `selection.json` | 本次选中的来源样本与固定划分 |
| `preflight-review.json` | 首批预检结论 |
| `execution.json` | 执行配置记录 |
| `summary.json` | 当前状态、数量、失败类型、用量与费用 |
| `completion-audit.json` | 完成状态检查 |
| `sources/00000/` 等 | 每条来源的提取与多变体生成请求/响应 |
| `*-request.json` | 实际发送的请求体；用于复现 |
| `extract.json` / `generate-01.json` 等 | API响应与程序校验结果；用于溯源 |
| `.lock` | 运行锁文件；保留历史运行结构 |

## 历史档案

| `archive/` 子目录 | 对应实验 |
|---|---|
| `pilot/` | 早期两条样本管线试验；[报告](../docs/reports/local-pilot.md) |
| `batch/` | 50条批量实验；[报告](../docs/reports/batch-50-report.md) |
| `extraction/` | 提取prompt v7/v8检查 |
| `expansion/` | 多变体生成试验；[报告](../docs/reports/multi-generation-report.md) |
| `bulk-preflight/` | 旧全量计划与预检v1/v2；[报告](../docs/reports/bulk-preflight-20261003.md) |
| `api-probes/` | 昆云接口探测；[报告](../docs/archive/kunyou-api-probe.md) |
| `packages/agentjev-v1/{github,autodl}/` | 已替代 AgentJev 方案的源码/上传包，保留交付历史 |
| `packages/semif-direct-torch28-v1/{github,autodl}/` | 固定 PyTorch 2.5.1 前的 Direct 源码/上传包，采用历史 .pt 保存格式 |
| `packages/llamafactory-direct-torch280-cu128-v1/{github,autodl}/` | 此前 LLaMA-Factory / torch 2.8 / cu128 交付包及原始清单，字节保留 |
| `packages/llamafactory-direct-torch260-cu124-steps600-v1/{github,autodl}/` | 三轮训练与 TensorBoard 启用前的 torch 2.6 / cu124 / 600 步交付包，字节保留 |
| `packages/autodl/` | 更早的首版 AutoDL 包 |
| `misc/` | 历史杂项目录 |

[catalog.json](catalog.json) 按每次运行列出旧/新路径、用途、文件数量、大小和关键文件哈希；不会把约两万份请求/响应铺到项目首页。冻结JSON仍保留旧路径和原始哈希，由公共路径解析器兼容读取。脚本迁移改变了代码哈希，因此旧全量任务不得用新代码强行续跑；新的调用应使用新运行目录。

当前配置与 checkpoint 协议为 `customer-policy-semif-direct-v1`；旧 `customer-policy-jev-v1` 自定义头 checkpoint 不兼容。方案对照见 [当前指南](../docs/guides/jev-training.md)和 [历史指南](../docs/archive/agentjev-training-v1.md)。

训练产生的正式对照实验建议放在 `training/llamafactory-direct-original-3epochs-v1/` 和 `training/llamafactory-direct-mixed-3epochs-v1/`；实验检查点、模型缓存与运行日志已被Git忽略。完整模块说明见 [项目结构](../docs/project-structure.md)。

ModelScope 适配前的 HF 下载版交付包保留在 `archive/packages/semif-direct-huggingface-v1/`。下载仍沿用当前模型分支；本次外部评测另以权重文件哈希固定三模型身份，记录在 `configs/ood-models-autodl-v1.json`，与提前锁定下载版本不同。

LLaMA-Factory 迁移前的 ModelScope / torch 2.5.1 手写训练交付包保留在 `archive/packages/semif-direct-modelscope-torch251-v1/`。当前完整续训检查点位于各实验的 `checkpoints/checkpoint-N/`；验证集所选模型另导出为 `export/`。
