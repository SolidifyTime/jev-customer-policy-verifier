# 2026-10-03 SemIf Direct 实现与验证

> 本页保留当时的实现与验证记录。当前使用 LLaMA-Factory / torch 2.6.0 / cu124，见 [最新环境记录](autodl-cu124-20261004.md)。

> 本页记录最初切换 Direct 时的验证，包含当时 Torch 2.8 / `.pt` 小试。后续固定为 Torch 2.5.1 / CUDA 12.4，当前检查点改用 safetensors，见 [环境核定记录](autodl-environment-20261003.md)。

根据当前选择，项目改为 SemIf-OpenJev Direct：从原始 `Qwen/Qwen3-0.6B` checkpoint 加载完整 `Qwen3ForCausalLM`，保留预训练 Transformer、词嵌入与原生词表头，用自己的数据进行全参数监督训练。

输入与读分参考 [SemIf-OpenJev 的固定版本](https://github.com/TheoLeeCJ/SemIf-OpenJev/tree/23cf1f39fc9534fe81437200959b6dfc7106e45a)，Qwen revision 仍为 `c1899de289a04d12100db370d81485cdf75e47ca`。上游提供推理方法；监督损失、训练和评测流程由本项目实现。具体操作见 [当前指南](../guides/jev-training.md)。

## 实际变更

- 每条样本编码为一条 chat prompt，包含证据、判断问题和 A/B 两个选项；正确标签只进入监督目标。
- 用 Qwen 原生词表头最后位置的 A/B logits 做二项 softmax；A=合规，B=违规。没有新增训练参数或评分头。
- 训练损失为二项交叉熵，全部原生参数以同一学习率进入 AdamW；Brier 保留为评测指标。
- 使用左侧 padding 和真实 token 的位置序号，`logits_to_keep=1` 只计算最后位置的词表分数。
- 新配置与 checkpoint 协议为 `customer-policy-semif-direct-v1`。保存和重载还检查实际 chat template 摘要、A/B token ID、固定问题与选项及模型来源摘要。
- 旧指南、配置和交付包分别保留在 `docs/archive/agentjev-training-v1.md`、`configs/archive/agentjev-training-v1.json` 与本地 `runs/archive/packages/agentjev-v1/`。旧自定义头 checkpoint 不兼容。

## 数据与本地检查

冻结五文件训练包保持原样：原始基线 2,442 条、混合训练 7,227 条、验证 730 条、测试 790 条。源数据、生成与复核档案均保留原字节。变更前后哈希检查保存在本地 `runs/maintenance/semif-direct-20261003/`。

真实固定版本 Qwen tokenizer 的 A/B token ID 为 32/33，单 token 往返与 prompt 边界检查通过；未下载完整模型权重。

| 数据 | prompt 数 | 最长 tokens | 中位数 | P95 | 截断 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 混合训练 | 7,227 | 1,202 | 566 | 814 | 0 |
| 原始基线 | 2,442 | 1,176 | 572 | 815 | 0 |
| 验证 | 730 | 1,045 | 558 | 814 | 0 |

审计记录为 `runs/training/audits/semif-direct-input-20261003.json`，所有 prompt 均低于 2,048 token 上限。长度审计不计算最终测试指标。

85 项离线测试通过，涵盖输入标签隔离、槽位分词与边界、原生 logits 一致性、无附加参数、左侧 padding、全参数反传、完整本地 checkpoint 加载、保存重载、选模、旧协议拒绝和打包后独立 CLI 检查。

独立检查还直接使用固定上游的 `core.py` 与 `direct.py`：13 组真实样本及构造 Unicode 探针的消息字节、prompt token IDs、摘要和 A/B 槽位均与本地一致。随机微型 Qwen 的批量左 padding 读分与上游单条读分最大差异为 `1.49e-8`，选项交叉熵与手算一致；记录为本地 `runs/maintenance/semif-direct-20261003/upstream-parity.json`。

另用随机 tiny Qwen 和离线 tokenizer 执行两步流程小试，输出到 `runs/training/smoke/semif-direct-local-smoke-20261003/`，重载预测一致且未评测测试集。该小试只验证代码流程，不代表真实 Qwen3-0.6B 的模型效果。

当前未创建 AutoDL 实例，未运行正式 Qwen3-0.6B GPU 训练；尚无真实模型准确率、增强收益或显存实测。源码包与包含冻结训练数据的 AutoDL 包分别保存在 `runs/packages/github/` 与 `runs/packages/autodl/`。
