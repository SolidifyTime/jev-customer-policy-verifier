# 2026-10-03 AutoDL 环境核定

> 本页保留当时的实现与验证记录。当前使用 LLaMA-Factory / torch 2.6.0 / cu124，见 [最新环境记录](autodl-cu124-20261004.md)。

> 此页保留 2026-10-03 的环境核验记录；当前下载已改用 ModelScope，见 [2026-10-04 适配记录](modelscope-20261004.md)。

当前目标为 **RTX 4090、Ubuntu 22.04、Python 3.12、PyTorch 2.5.1、CUDA 12.4**。官方提供 `torch==2.5.1+cu124` 的 Linux x86_64 / CPython 3.12 wheel；当前原生 Qwen3、SDPA、梯度检查点和 bf16 训练接口与此版本匹配。安装与实例自检步骤见 [环境指南](../guides/autodl-environment.md)。

## 依赖与实现变更

共同训练依赖固定为 torch 2.5.1、Transformers 4.57.6、huggingface-hub 0.36.0、tokenizers 0.22.1、safetensors 0.6.2、NumPy 1.26.4 和 Jinja2 3.1.6。AutoDL 使用 `requirements/autodl-cu124.txt`，先从官方 cu124 索引安装 torch；不会按旧文件升级到 2.8。

官方发布元数据核对了 7 个核心包及 31 个额外发行构件的 Python 版本条件、Linux wheel 标签，51 条依赖约束全部满足；报告位于本地 `runs/maintenance/autodl-env-20261003/linux-wheel-metadata.json`。其中官方 cu124 wheel 的小型 metadata 接口返回 403，Torch 的自动 CUDA 依赖证据另取自 PyPI 2.5.1 Linux wheel metadata，报告明确区分来源；未将元数据核对称为 Linux/GPU 运行验证。

训练继续使用 SemIf Direct 单 prompt、原生 A/B logits、二项交叉熵与全参数更新，数据和模型结构保持不变。当前预训练加载显式使用 safetensors，自产 checkpoint 也改为 `best.safetensors`：完整 tensor 与 `safetensors-json-v1` JSON 元数据保存在同一文件，保留共享词嵌入、校验模型/数据来源和输入协议，严格重载权重。运行代码不使用 `torch.load`，不自动转换历史 `.pt`。

新增 `python scripts/cpv.py train env`，无需正式模型权重或数据，检查依赖与目标设备，再用 tiny Qwen 执行 SDPA、非重入梯度检查点、优化器更新及 safetensors 共享权重重载。CUDA 模式还检查 Ubuntu 22.04、Python 3.12、单张 RTX 4090、CUDA 12.4 和 bf16。

## 实际完成的验证

在独立的本机 Mac arm64 / Python 3.12.14 CPU 环境安装上述全部固定依赖，保留原先本机的 Torch 2.8 虚拟环境。

- `pip check` 通过，没有依赖冲突。
- **88 项离线测试全部通过**，无跳过；覆盖原生模型加载、输入编码、全参数反传、safetensors 保存/重载、共享权重、元数据/旧格式拒绝、选模、打包及数据完整性。
- `train env --device cpu` 的 native Qwen3 / SDPA / 梯度检查点 / AdamW / safetensors 探针通过。
- 4 条样本的独立两步 tiny CPU 训练通过，输出位于 `runs/training/smoke/semif-direct-torch251-smoke-20261003/`；重载预测一致，未评测测试集。
- 固定真实 Qwen tokenizer 在 tokenizers 0.22.1 下重新审计通过，A/B token ID 32/33，混合训练/原始基线/验证最长分别为 1,202 / 1,176 / 1,045 tokens，无截断；记录为 `runs/training/audits/semif-direct-torch251-input-20261003.json`。

测试与安装日志、解析结果和发行元数据保存在 `runs/maintenance/autodl-env-20261003/`。旧源码/上传包归档到 `runs/archive/packages/semif-direct-torch28-v1/`，历史数据与运行文件不改写。

**尚未创建 AutoDL 实例，Linux / RTX 4090 的驱动、CUDA 内核、bf16 GPU 训练、显存和真实 Qwen3-0.6B 训练仍未实测。** 实例创建后按环境指南先运行 CUDA 自检、tiny GPU 小试，再运行固定真实权重的两步小试。
