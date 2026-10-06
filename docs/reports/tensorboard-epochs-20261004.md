# 三轮训练与 TensorBoard 启用验证

2026-10-04 将正式训练预算改为 **3 个完整 epoch**，启用 TensorBoard。模型路线沿用 LLaMA-Factory 0.9.5＋SemIf Direct＋完整预训练 Qwen3-0.6B，全参数训练、A/B 受限交叉熵、ModelScope 当前版本下载及其余超参数保持原有设置。

## 配置和实际行为

`configs/jev-training.json` 设置 `num_train_epochs=3`、`max_steps=-1`、`report_to=["tensorboard"]`。正数 `max_steps` 会优先限制训练，因此必须解除此前的 600 步预算。正式轮数可用 `--num-train-epochs N` 覆盖；`--max-steps N` 可显式切换回步数预算，两参数互斥。两步 smoke 继续使用自己的小试预算。

单卡、microbatch 1、梯度累积 16，由实际安装的 Trainer 与冻结数据数量计算：

| 训练集 | 样本 | 每轮 optimizer 更新 | 三轮更新 |
| --- | ---: | ---: | ---: |
| 原始 | 2,442 | 153 | 459 |
| 混合 | 7,227 | 452 | 1,356 |

每轮末尾的不足 16 个微批也执行更新，不丢样本。两组各遍历三次，混合组计算量更大。`run.json.training_plan` 记录框架计算的预算；`selection.json` 记录完成步数、轮数与所选模型。第 0 步、每 100 步与最后一步验证，最后一步也保存完整检查点，因此 459 / 1,356 步均能完成选模与导出。

TensorBoard 2.21.0 加入共同依赖，每个实验写入 `tensorboard/events.out.tfevents.*`。训练 loss、学习率、梯度范数、epoch 每 20 次更新及最后一步记录；验证交叉熵、准确率、宏 F1、违规召回率等随验证记录。只向 TensorBoard 发送标量，混淆矩阵继续完整保存在 JSON 中。续训沿用原目录和恢复的 global step，新增事件文件。

## 数值验证与续训修复

本次新增跨轮续训测试发现：Accelerate 1.11 的 `skip_first_batches` 会新建 iteration 为 0 的 DataLoaderShard，恢复第二轮中途的训练时改变原有洗牌轮次。项目增加按 seed＋Trainer 当前/恢复轮次生成排列的 `EpochSeededSampler`，框架继续执行训练循环、跳过已用数据、优化器、调度器和 RNG 恢复。修复后逐 tensor 比对通过；未修改第三方安装目录。

独立 Mac arm64 CPU / Python 3.12.14 / PyTorch 2.6.0 / LLaMA-Factory 0.9.5 / TensorBoard 2.21.0 安装完整共同依赖，原 `.venv` 保留。证据位于 Git 忽略的 `runs/maintenance/tensorboard-epochs-20261004/`：

| 检查 | 结果与记录 |
| --- | --- |
| 完整依赖与冲突 | 安装成功、`pip check` 通过；`install-report.json`、`installed-packages.txt`、`pip-check.log` |
| 离线回归 | **108 项通过，无跳过**；`tests.log` |
| 三轮完整覆盖 | 3 条样本、微批 1、累积 2，形成每轮两次更新；共 6 次更新，每条样本恰好使用三次 |
| 最后一步验证保存 | 即使验证间隔为 100，仍保存第 6 步验证、日志和完整检查点，初始第 0 步验证仍可选模 |
| 跨轮断点续训 | 第二轮中途第 3 步恢复至第 6 步；最终权重与连续训练逐 tensor 完全相同，TensorBoard loss 步数连续为 1–6 |
| 已完成训练后的导出恢复 | 最后一步检查点恢复只完成重载与导出，不再增加训练更新 |
| TensorBoard 实际写入 | EventAccumulator 读回 loss / learning rate / grad norm / 验证指标，最终验证交叉熵与 JSON 一致；`training-evidence.json` |
| CPU 环境探针 | 实际 LF 累积、完整续训和 TensorBoard 标量保存通过；`cpu-environment-final.json` |
| 冻结数据两步小试 | 随机 tiny Qwen 四条训练、四条验证、两次更新，事件、选模、重载与标准导出通过；`smoke-final.log` |
| 正式预算核对 | 使用实际 Trainer 与冻结数据数量计算 459 / 1,356 步，不执行正式训练；`training-evidence.json` |
| Linux 发布元数据 | 17 个直接依赖具有 Python 3.12 兼容 wheel；38 项直接依赖间约束满足，官方 cu124 三件套具有 Linux CP312 wheel；`linux-metadata.json` |

小试保存在 `runs/training/smoke/llamafactory-direct-tensorboard-cpu-20261004/`；测试集未评测。输入协议、tokenizer 与冻结数据未改变，此前真实 tokenizer 长度审计继续适用。

本轮验证使用 CPU 随机小模型。Linux 元数据检查不执行 Linux/CUDA wheel，也不代表已在 Linux 完成所有传递依赖安装。**RTX 4090 / CUDA / bf16、完整 Qwen3-0.6B GPU 训练、速度、显存与正式模型效果尚未实测。**

## 交付与已有 AutoDL 项目更新

- `runs/packages/autodl/customer-policy-verifier-training.tar.gz`：最新完整上传包，包含冻结训练数据。
- `runs/packages/autodl/customer-policy-verifier-update.tar.gz`：相对此前 torch 2.6 / cu124 上传包的增量，只包含变更后的公开文件，不含数据、模型、环境或运行输出；顶层目录仍为 `customer-policy-verifier/`。
- `runs/packages/github/customer-policy-verifier-source.tar.gz`：最新公开源码快照，不含真实数据和运行产物。

各包附成员清单。交付核对覆盖整包和成员、源码解包测试、旧 AutoDL 包叠加增量后与新完整包一致，以及冻结数据检查；记录为 `delivery-verification.json`。旧 torch 2.6 / cu124 / 600 步包与清单原样归档到 `runs/archive/packages/llamafactory-direct-torch260-cu124-steps600-v1/`，已有数据与历史运行字节核对为 `artifact-integrity.json`。

用户已上传项目，可按 [增量更新步骤](../guides/autodl-environment.md#已上传项目的增量更新)覆盖公开文件、补装依赖、运行 GPU 环境探针，然后按 [训练指南](../guides/jev-training.md#原始与混合数据对照)分别训练三轮。TensorBoard 查看方式见 [日志访问](../guides/autodl-environment.md#tensorboard-日志检查与访问)。本轮未连接远程实例、启动正式训练或推送 GitHub。
