# LLaMA-Factory Direct 接入与验证记录

> 本页保留最初 PyTorch 2.8.0 / cu128 接入时的实现与验证。当前用 PyTorch 2.5.1 / CUDA 12.4 基础镜像开机，在独立环境安装 PyTorch 2.6.0 + cu124；最新结果见 [cu124 环境适配记录](autodl-cu124-20261004.md)。

2026-10-04 将手写 PyTorch 训练循环迁移至 **LLaMA-Factory 0.9.5＋自定义 Direct Trainer**，目标环境调整为 **Ubuntu 22.04 / Python 3.12 / PyTorch 2.8.0 / CUDA 12.8 / RTX 4090**。数据、Direct chat prompt、A/B 选项、原始 Qwen 模型结构和受限交叉熵保持一致。

## 实际实现

- 数据 collator：保留原始非思考 prompt，左 padding，标签作为独立监督目标。
- Direct Trainer：重写 loss 和 prediction，明确梯度累积归一化，只累计两个选项 logits。
- 工作流：全参数 AdamW、cosine 调度、验证日志、包含第 0 步的选模、最近两个完整续训检查点。
- 续训：检查运行身份，恢复框架 optimizer、scheduler、RNG 和项目选模状态。
- 导出：独立重载验证所选模型后保存标准 Qwen safetensors 和正式 tokenizer。
- 环境：更新 cu128 三件套与框架依赖，增加实际 Trainer 续训探针，更新 CI 和指南。

上游依据为 [LLaMA-Factory 0.9.5 Trainer](https://github.com/hiyouga/LlamaFactory/blob/v0.9.5/src/llamafactory/train/sft/trainer.py)和 [依赖声明](https://github.com/hiyouga/LlamaFactory/blob/v0.9.5/pyproject.toml)。项目通过继承扩展，未修改安装的第三方库。设计见 [接入说明](../design/llamafactory-training.md)。

## 已完成验证

验证环境为独立 **Mac arm64 CPU / Python 3.12.14 / PyTorch 2.8.0**，实际安装 requirements/training.txt 的完整依赖。

| 检查 | 结果 / 本地记录 |
| --- | --- |
| 完整离线回归 | **105 项通过，无跳过**；`runs/maintenance/llamafactory-20261004/tests.log` |
| 依赖安装与冲突检查 | 完整安装成功，`pip check` 通过；`install-report.json`、`pip-check.log` |
| 梯度累积 | 单次整批与两次微批的各参数梯度数值一致 |
| 真正中断续训 | 优化器恢复到第 1 步，完成第 2 步；最终权重与连续训练逐 tensor 完全相同 |
| 原生标准导出 | 权重与所选 `best.safetensors` 一致，原生词嵌入与 LM head 保持共享 |
| 环境探针 | 原生 Qwen3 / SDPA / 梯度检查点 / AdamW / safetensors 和实际 LF 续训通过；`cpu-environment.json` |
| 四样本两步 CPU 小试 | `runs/training/smoke/llamafactory-direct-cpu-20261004/`，所选模型独立重载通过，未评测测试集 |
| 真实 tokenizer 本地审计 | `runs/training/audits/llamafactory-direct-input-20261004.json`；禁止 HTTP 请求，最长 1202/1176/1045 tokens，A/B=32/33 |
| Linux 发布信息 | 直接依赖的 CPython 3.12 兼容 wheel 标签、核心版本约束与官方 cu128 三件套；`linux-metadata.json` |

Linux 记录核对发行信息，不执行 Linux wheel，也不声称已在 Linux 完成全部传递依赖安装。实际 CUDA、bf16、显存、速度及 Qwen3-0.6B 的效果还没有 GPU 实测；tiny 指标仅说明流程可用。

## 数据与交付历史

冻结的四份 JSONL 和 manifest 没有重写或重新划分。旧数据、生成请求/响应、运行记录与此前交付包均保留字节校验；结果保存在维护目录的 `artifact-integrity.json`。

迁移前的 ModelScope 手写训练版源码与 AutoDL 包归档到 `runs/archive/packages/semif-direct-modelscope-torch251-v1/`。当前交付为 `runs/packages/autodl/customer-policy-verifier-training.tar.gz` 与 `runs/packages/github/customer-policy-verifier-source.tar.gz`，前者包含冻结训练数据，后者只包含公开源码与索引。解包后的检查与成员校验见 `delivery-verification.json`。

此轮没有创建 GitHub 仓库或推送，也没有下载完整 Qwen3-0.6B 权重。ModelScope 继续下载官方当前 `master`，项目不锁下载模型的提交或文件哈希。
