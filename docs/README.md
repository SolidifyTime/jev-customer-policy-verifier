# 文档导航

当前训练路线为 SemIf-OpenJev Direct：完整预训练 Qwen3-0.6B 全参数监督训练，读取 A/B 概率。AutoDL RTX 4090 / Python 3.12 / PyTorch 2.6.0+cu124 已完成两组三轮训练；因发现原始数据的大小写与标签高度相关，当前对比改用三个证据字段统一小写、1 epoch 和 10% warmup，见 [新实验指南](guides/case-neutral-comparison.md)。旧实验与配置保留为历史记录。

## 从这里开始

- 项目首页：[English](../README.md) / [简体中文](../README.zh-CN.md)，包含目标、当前状态和快速入口。
- [一轮大小写控制实验](guides/case-neutral-comparison.md)：归一化协议、10% warmup、统一选模与训练评测入口。
- [一轮实验验证与启动](reports/case-neutral-1epoch-20261004.md)：两端 114 项测试、真实 tokenizer 审计和 AutoDL 后台运行位置。
- [一轮大小写控制结果](reports/case-neutral-1epoch-results-20261004.md)：训练评测已全部完成；原始/混合测试准确率为 96.84% / 96.08%，包含错误对比与核对记录。
- [跨场景选择题评测](guides/ood-choice-evaluation.md)：RuleTaker 二选一与 LogiQA 2.0 四选一的固定协议、三模型对照、选项换位与复现入口。
- [跨场景选择题结果](reports/ood-choice-results-20261004.md)：三模型评测已完成；没有跨任务泛化提升，换位诊断揭示微调模型的 A/B 偏好。
- [客服任务基础模型对照](reports/base-task2-results-20261004.md)：同一小写 Direct 协议，基础测试 50.00%，原始/混合 96.84% / 96.08%；逐题概率已保存并复核。
- [项目结构](project-structure.md)：每个模块的输入输出、数据与运行目录层次、路径迁移规则。
- [AutoDL 4090 环境指南](guides/autodl-environment.md)：2.5.1 基础镜像开机，独立 Conda 环境安装 Python 3.12 / torch 2.6.0+cu124，随后设备自检与 GPU 小试。
- [三轮训练与 TensorBoard 记录](reports/tensorboard-epochs-20261004.md)：当前 108 项测试、跨轮续训、日志写入与增量更新包。
- [cu124 环境适配记录](reports/autodl-cu124-20261004.md)：当前环境组合的选择依据，保留当时未启用 TensorBoard 的验证记录。
- [LLaMA-Factory 接入设计](design/llamafactory-training.md)：自定义损失、梯度累积、框架生命周期、续训与导出。
- [LLaMA-Factory 接入记录](reports/llamafactory-20261004.md)：最初 2.8.0 / cu128 接入时的 105 项测试、CPU 小试和真实 tokenizer 审计。
- [ModelScope 下载适配](reports/modelscope-20261004.md)：官方当前版本下载、必要文件检查、98 项离线测试与本地 tokenizer 审计。
- [历史 AutoDL 环境核定记录](reports/autodl-environment-20261003.md)：发行元数据、88 项固定版本 CPU 测试、safe checkpoint 与尚待 GPU 验证的范围。
- [当前 Direct 训练指南](guides/jev-training.md)：预训练原生头、A/B 读取、三轮监督训练、TensorBoard、本地检查、AutoDL 与评测。
- [最初 Direct 实现与验证记录](reports/semif-direct-20261003.md)：本次路线切换、输入审计与训练流程检查。
- [GitHub 上传指南](guides/github.md)：公开代码的范围和以后创建远程仓库的步骤。
- [本次目录整理记录](reports/layout-cleanup-20261003.md)：备份、文件哈希核对、离线验证与两类交付包。

## 当前数据与操作

| 文档 | 阅读目的 |
| --- | --- |
| [Task 2 来源与固定分区](data/task2-dataset.md) | 官方九个数据视图、规范英文来源、分组划分与生成输入 |
| [统一训练数据](data/training-data.md) | 7,227 条混合训练文件、2,442 条基线、固定评测和输入边界 |
| [全量生成指南](guides/bulk-generation.md) | 当前 OpenRouter 流程、调用预算和恢复规则；全量已完成 |
| [少量抽查与导出](guides/spot-check-guide.md) | 当前每批默认三条抽查、质量状态和导出格式 |
| [人工复核指南](guides/manual-review-guide.md) | 首批逐条复核的历史流程与结果解释 |

## 事实与实验报告

优先阅读 [全量完成检查](reports/bulk-completion-20261003.md)和 [人工复核结果](reports/manual-review-report.md)。其余报告保留当时的版本、数字和判断，不能直接当作现在的运行状态。

- [两样本小试](reports/local-pilot.md)、[首批 50 条](reports/batch-50-report.md)、[多次生成](reports/multi-generation-report.md)：早期生成与审核的取舍。
- [全量启动前验证](reports/bulk-preflight-20261003.md)：v10 → v12 的真实端到端小试。
- [数据构建经验](design/data-lessons-learned.md)：引用、近重复、语义漂移、标签和证据链。
- [提取提示词调整](design/extraction-prompt-update.md)、[生成提示词调整](design/generation-prompt-update.md)：当时的小样本复测和局限。
- [最初项目规划](design/project-plan.md)：历史原 README，保留规划背景；当前实现以首页和训练指南为准。
- [旧 AgentJev 训练方案](archive/agentjev-training-v1.md)：已替代的随机候选头、分段编码、CE＋Brier 与当时的小试记录。
- [公益站 API 探测](archive/kunyou-api-probe.md)：历史接入记录，当前生成使用 OpenRouter。

真实数据和运行输出保存在 Git 忽略目录；公开目录索引位于 [data](../data/README.md)与 [runs](../runs/README.md)。文档中的历史路径可通过 [迁移映射](../configs/layout-migration.json)定位，冻结文件内的路径字符串和哈希保持原样。
