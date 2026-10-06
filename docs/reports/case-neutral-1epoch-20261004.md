# 一轮大小写控制实验：实现、验证与启动

运行已于 2026-10-04 15:57:45 完成，最终指标见 [结果报告](case-neutral-1epoch-results-20261004.md)。下文保留启动时的状态与验证证据。

记录日期：2026-10-04，Asia/Shanghai。Mac 与 AutoDL 项目中的代码、配置、测试和指南已同步，正式对比在 AutoDL `cpv` 环境启动。本文是启动记录，训练与最终评测尚未完成，不能作为新模型效果报告。

## 实现

新配置为 `configs/jev-training-case-neutral-1epoch.json`；旧三轮配置及产物保留。仅在模型输入边界对规则、对话历史和候选回复统一小写，原始 JSONL、标签和固定分区未改动。处理方式保存到检查点输入协议，评测从检查点恢复。旧模型无归一化字段时继续使用旧 prompt，已核对两组历史检查点的协议兼容性。

两组均从原始预训练 Qwen3-0.6B 初始化，全参数训练 1 epoch，microbatch 1、累积 16、学习率 `1e-5`、bf16、cosine；第 0 步和 epoch 结束验证，按验证交叉熵选模。两组选择均冻结后才评测测试集。最后一步完整训练检查点、标准 export 和 TensorBoard 保留。

| 组别 | 样本数 | 更新预算 | warmup 更新 |
| --- | ---: | ---: | ---: |
| 原始 | 2,442 | 153 | 16 |
| 原始＋合成 | 7,227 | 452 | 46 |

配置使用 `warmup_ratio=0.1`、`warmup_steps=0`，由实际更新预算向上取整。原始组的实际 `run.json` 已核对为 153 步、16 步 warmup；日志中学习率在第 17 次更新使用 `1e-5`，随后 cosine 衰减，符合框架记录本次更新所用学习率的方式。

## 验证证据

- Mac 独立 Python 3.12 / torch 2.6.0 / LLaMA-Factory 0.9.5 环境：**114 项测试通过，无跳过**；`pip check` 通过。原有 `.venv` 未修改。
- AutoDL `cpv` / torch 2.6.0+cu124：**114 项测试通过，无跳过**。
- 回归覆盖大小写变化下相同 token 输入、原始记录不被修改、归一化后标签冲突与跨分区重叠拒绝、按实际 153/452 步计算 16/46 步 warmup、cosine 起点/峰值/终点、归一化模型独立重载和评测。
- Mac 和 AutoDL 真实 tokenizer 审计、归一化证据哈希与输入协议完全一致。最长 token：混合 1,201、原始 1,175、验证 1,042、测试 1,038，全部不截断。
- 归一化后无完整输入标签冲突、无跨分区重复。混合训练中 4 条同标签完整输入重复仍保留；原始训练、验证、测试均无完整输入重复。
- 所有分区的候选回复均无大写。原来的大小写规则在平衡验证和测试集上均降为 50%。这不意味着其他样式线索或标签质量问题已经全部解决。
- 同步的文件逐项 SHA-256 核对通过；远端覆盖前已备份公开文件，并记录旧配置、数据 manifest、旧 run/selection/best 权重哈希。原始预训练权重内容摘要也已记录，便于后续核对。

详细证据：Mac `runs/maintenance/case-neutral-1epoch-20261004/`，远端项目同名维护目录。CPU 验证环境位于该目录的 `venv/`；测试与审计日志不纳入公开源码包。

## 远端运行

启动时间：2026-10-04 15:13:13（Asia/Shanghai）。使用独立后台进程，脱离 SSH 会话；启动器 PID 为 23852。

- 项目：`/root/autodl-tmp/customer-policy-verifier`
- 原始组：`/root/cpv-experiments/llamafactory-direct-original-case-neutral-1epoch-v1`
- 混合组：项目 `runs/training/llamafactory-direct-mixed-case-neutral-1epoch-v1`
- 状态：项目 `runs/training/case-neutral-1epoch-v1/status.json`
- 总日志：项目 `runs/training/case-neutral-1epoch-v1.log`
- 最终汇总：项目 `runs/training/case-neutral-1epoch-v1/comparison.json`（完成后生成）

原始组放在系统盘、混合组放在数据盘，以容纳完整 optimizer 检查点与 export，旧实验未清理。最后检查时状态为 `running / training_original`，已完成 35/153 次更新。状态和最终结果以远端产物为准。

启动命令及重新运行方法见 [实验指南](../guides/case-neutral-comparison.md)。相同 epoch 仍不是相同计算预算；本轮同时改变大小写处理、epoch 和 warmup，因此不能把与旧结果的全部差异单独归因于大小写。
