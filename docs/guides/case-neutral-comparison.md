# 大小写控制的一轮对比实验

2026-10-04 检查 AutoDL 三轮结果时发现：原始训练和验证集的合规回复全部小写、违规回复全部含大写。只根据这一特征分类，验证准确率为 100%，测试为 789/790（99.87%）。原来的高准确率不足以说明模型理解了业务规则；这项诊断本身不能证明模型确实采用了该特征。

本轮使用 `configs/jev-training-case-neutral-1epoch.json`。旧 `configs/jev-training.json` 和已有三轮模型保留供历史复现。

## 实验条件

- 在模型输入边界，对 `policy`、每条 `conversation` 和 `candidate_response` 统一执行 `str.lower()`，与标签无关。原始 JSONL、标签、ID、metadata 和分区不修改。system、判断标准、A/B 选项及答案 token 保持原协议。
- 原始训练 2,442 条；混合训练 7,227 条；验证 730 条；测试 790 条。归一化后再次检查完整输入标签冲突及跨分区重叠。
- 两组从同一本地预训练 Qwen3-0.6B 初始化，均全参数训练 **1 epoch**，学习率 `1e-5`，microbatch 1，梯度累积 16，bf16，cosine 调度。
- `warmup_ratio=0.1`、`warmup_steps=0`。框架计算 `ceil(optimizer_steps × 0.1)`。配置拒绝同时设置正的固定步数和比例，避免固定值覆盖比例。10% 是本轮选定的超参数，并非所有任务都最优的固定要求；接口依据见 [Transformers TrainingArguments](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer#transformers.TrainingArguments)。
- 每组在第 0 步和 epoch 结束时验证，按验证交叉熵选模。两组选择冻结后，分别评测完整验证集和测试集；测试不参与选模。完整最后一步检查点保留 1 个，包含 optimizer、scheduler 和 RNG，支持原 CLI 中断续训。
- 归一化协议存入 `run.json`、检查点 header、`export/cpv_model.json` 和评测报告。独立评测从检查点恢复处理方式；直接使用标准 HF export 时，调用方也须遵循该输入协议。

| 组别 | epoch | optimizer 更新 | warmup 更新 | 选模步数 |
| --- | ---: | ---: | ---: | --- |
| 原始 | 1 | 153 | 16 | 0、153 |
| 原始＋合成 | 1 | 452 | 46 | 0、452 |

`run.json` 的 `training_plan` 记录实际更新预算和 warmup 步数。TensorBoard 每步写入 loss、learning rate 和 grad norm。相同 epoch 不代表相同计算量，混合组仍使用更多样本和更新次数。

此处理让模型看不到证据字段的大小写差异，包括回复与历史之间的差异。它不解决其他措辞、标点或标签质量问题，也不是原模型依赖大小写的因果检验。标识符也会小写，适用于当前英文规则任务；结果不能推广到大小写承载业务语义的任务。

## 使用

激活 AutoDL 的 `cpv` 环境，在项目根目录执行：

```sh
python scripts/cpv.py train check --config configs/jev-training-case-neutral-1epoch.json
python scripts/cpv.py train audit --config configs/jev-training-case-neutral-1epoch.json \
  --output runs/training/audits/case-neutral-1epoch-v1.json
python -u scripts/run_case_neutral.py
```

运行器顺序完成两组训练，然后评测两组。默认运行目录为 `runs/training/llamafactory-direct-{original,mixed}-case-neutral-1epoch-v1/`；状态和对比汇总位于 `runs/training/case-neutral-1epoch-v1/{status,comparison}.json`。它拒绝覆盖已有运行；重做时指定新的 `--status-dir`、`--original-run-dir` 和 `--mixed-run-dir`。

可脱离 SSH 会话运行：

```sh
nohup python -u scripts/run_case_neutral.py > runs/training/case-neutral-1epoch-v1.log 2>&1 < /dev/null &
```

当前 AutoDL 数据盘约剩 13 GB，系统盘约剩 19 GB。本轮将原始组放在 `/root/cpv-experiments/llamafactory-direct-original-case-neutral-1epoch-v1`，混合组放在项目的 `runs/training/`，通过运行器参数指定，保留旧实验。手动续训时使用本轮配置及原 CLI 的 `--resume-from-checkpoint`，指向本组最后一个完整检查点。

Mac 用于代码、数据审计和 CPU 小模型回归；正式对比运行于远程 RTX 4090。
