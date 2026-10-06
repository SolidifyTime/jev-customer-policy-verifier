> **历史方案，已被替代。** 本页记录此前的 Xiaoke / AgentJev 自定义候选评分头路线及其参数，不适用于当前训练代码。当前采用 SemIf-OpenJev Direct 与完整预训练 Qwen3ForCausalLM，见[当前训练指南](../guides/jev-training.md)。旧配置保留在 [agentjev-training-v1.json](../../configs/archive/agentjev-training-v1.json)，旧交付包保留在本地 `runs/archive/packages/agentjev-v1/`；本页的审计数字、测试结果和命令属于当时版本。

# AgentJev 全参数监督训练与 AutoDL 运行

训练路线固定为 **`Qwen/Qwen3-0.6B` 预训练骨干＋新初始化的 AgentJev 候选评分头**。骨干和评分头全部参与训练，损失为候选概率上的交叉熵＋`0.1 × Brier`；不加载已发布的 AgentJev checkpoint，不生成答案文本。

参考实现固定在 [XiaokeAILabs 的 24277dc 提交](https://github.com/li-xiu-qi/XiaokeAILabs/tree/24277dc067e2c1809013068a559c6b273b202d5d/experiments/test_jev_open_source/text_jev_train)，Qwen 模型固定 revision `c1899de289a04d12100db370d81485cdf75e47ca`。实际使用的 HF repo ID 是 `Qwen/Qwen3-0.6B`；参考项目的本地目录名虽然含 `-Base`，下载的也不是独立的 `Qwen3-0.6B-Base` 仓库。

## 实现与数据

- `src/customer_policy_verifier/training/jev_model.py`：Qwen3 骨干、独立候选路径编码、无位置嵌入的候选集合 Transformer、标量评分头及监督损失。
- `src/customer_policy_verifier/training/jev_data.py`：原数据包校验、三输入字段的稳定序列化、one-hot 目标、无截断编码及二分类指标。
- `src/customer_policy_verifier/training/train_jev.py`：下载、输入长度检查、小试、全量训练、checkpoint 选优与重载、独立评测及 AutoDL 打包。
- `configs/jev-training.json`：模型版本、学习率、步数、批量和精度。

`state` 只含 `policy`、`conversation`、`candidate_response`。固定英文问题询问当前这一步的回复是否符合规则；候选顺序为 `compliant`、`violating`，目标分别为 `[1,0]`、`[0,1]`。ID、metadata、源标签解释都不进入输入。

每个候选采用上游 `[STATE]`、`[QUESTION]`、`[CANDIDATE]` 编码路径，取最后有效 token 的隐藏状态评分。使用基础 tokenizer 编码，不套生成式 chat template。任何路径超过 2,048 tokens 都会报错，代码不截断。业务规则、完整历史和待审回复均保留。

仍使用现有 `data/training/task2-v1/` 的四个文件与 manifest，不重新划分。`train_original.jsonl` 是 2,442 条原始基线，`train.jsonl` 是 7,227 条混合训练数据；验证 730 条、测试 790 条。文件摘要、来源组隔离、原始子集一致性和固定评测文件都会校验。manifest 内的历史绝对路径仅作来源记录，运行时从 `--data-dir` 读取，因此数据包可以复制到 AutoDL。

## 本地检查

只用 Python 标准库核对数据：

```sh
python3 scripts/cpv.py train check
```

已有本地 `.venv` 时，可以进行完全离线的小模型集成检查：

```sh
.venv/bin/python scripts/cpv.py train smoke --tiny-model \
  --run-dir runs/training/smoke/offline-v1 --samples 4
```

这会用随机初始化的小型 Qwen 和简化 tokenizer 验证梯度、两步训练、选优、保存和重载。它仅证明代码流程可运行，不能作为 Qwen3-0.6B 的模型效果或 CUDA 显存测量。

新环境先建立虚拟环境并安装训练依赖，再运行上面的命令：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements/training.txt
```

真实分词器检查只需下载配置和 tokenizer，无需模型权重：

```sh
.venv/bin/python scripts/cpv.py train download --tokenizer-only
.venv/bin/python scripts/cpv.py train audit --output runs/training/audits/input-audit.json
```

2026-10-03 已使用固定版本的真实 Qwen tokenizer，按上游分段编码检查完整数据：混合训练集最长路径 **1,064 tokens**，中位数 456、P95 693；原始基线最长 1,064，验证最长 928，均在 2,048 上限内，无截断。记录位于本地 `runs/training/audits/jev-input-audit-upstream-format-20261003.json`。目录调整后，24 项训练相关离线测试通过，包括打包解压后独立运行 CLI 检查；普通权重加载路径也用本地随机小型 Qwen 验证。完整 0.6B 权重尚未在 GPU 上训练，当前没有正式模型指标或显存实测。

长度检查仅编码训练和验证数据。`check` 会读取测试文件进行哈希和分区完整性验证；训练、小试和长度检查不计算测试指标。

## 上传 AutoDL

本地创建训练上传包：

```sh
python3 scripts/cpv.py train package \
  --output runs/packages/autodl/customer-policy-verifier-training.tar.gz
```

包内包含当前 Python 源码包、统一 CLI、项目配置/元数据、训练依赖、Markdown 文档、测试与 CI、公开目录索引、忽略规则和空密钥模板、上游代码许可证，以及当前五个固定训练数据文件。模型权重在 AutoDL 下载；原始数据归档、API 请求/响应、历史运行和实际密钥不打包。压缩包旁有成员摘要和包摘要清单；已有输出会拒绝覆盖。该包用于自己的训练实例，GitHub 提交范围见 [上传指南](../guides/github.md)。

通过 [AutoDL 的上传方式](https://www.autodl.com/docs/scp/)将压缩包放到实例的数据盘，在终端解压：

```sh
cd /root/autodl-tmp
tar -xzf customer-policy-verifier-training.tar.gz
cd customer-policy-verifier
```

选择支持 CUDA 和 bf16 的 GPU 环境，Python 使用 3.10–3.12。当前训练依赖固定在 `requirements/training.txt`，PyTorch 2.8.0；环境镜像的 CUDA 与 PyTorch 应匹配。GPU 配置与批量大小以小试的实际显存和耗时为依据。[AutoDL 环境说明](https://www.autodl.com/docs/base_config/)

```sh
python -m pip install -r requirements/training.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
python scripts/cpv.py train check
python scripts/cpv.py train download
python scripts/cpv.py train audit --output runs/training/audits/input-audit.json
```

如使用 Hugging Face 镜像，下载命令可显式指定 endpoint，仍下载固定 revision：

```sh
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  python scripts/cpv.py train download
```

## 先跑真实 GPU 小试

```sh
python scripts/cpv.py train smoke --train-set original \
  --run-dir runs/training/smoke/gpu-v1 --samples 4 \
  --microbatch-questions 1 --gradient-accumulation 1
```

本次小试使用真实 Qwen3-0.6B 权重与随机评分头，执行两步全参数训练，评测少量原始验证样本，然后从磁盘构建模型、严格恢复所有权重并核对结果。检查 `run.json` 的全参数数量、`events.jsonl` 的有限 loss、`selection.json` 的显存占用和 `reload_check.json` 的通过状态。

小试不足以衡量正式效果，不能使用其四条验证样本选择最终训练配置。若显存不足，先维持 microbatch 1；配置已启用 SDPA 和梯度检查点。bf16 不支持时可使用 `--precision fp32`。

## 原始基线与混合数据训练

先完成原始基线：

```sh
python scripts/cpv.py train train --train-set original \
  --run-dir runs/training/original-v1
```

再以相同模型初始化、随机种子、有效批量和训练步数训练混合数据：

```sh
python scripts/cpv.py train train --train-set mixed \
  --run-dir runs/training/mixed-v1
```

默认骨干学习率 `1e-5`、评分头 `1e-4`，训练 600 个 optimizer steps，microbatch 1、梯度累积 16，有效批量 16 个问题，每个问题两条候选路径。两组均使用 float32 参数、bf16 autocast、AdamW 和梯度裁剪，全参数都进入 optimizer。约对应原始基线 3.93 轮、混合数据 1.33 轮；这是固定更新预算的对比，需要在报告中说明，不能同时称为相同 epoch 的对比。

每 100 步及最后一步，按完整原始验证集的交叉熵选择 checkpoint。第 0 步也保存，因此训练无提升时仍能保留并恢复初始化模型。训练输出目录非空时拒绝覆盖。首版没有断点续训；中断后使用新目录重新启动，不把已有 checkpoint 当成新的初始化方案。

正式输出包括：

- `run.json`：配置、模型与数据来源、代码摘要、依赖版本、硬件、全参数数量和输入长度。
- `initial_validation.json`：随机评分头初始化时的验证结果。
- `events.jsonl`：训练 loss、耗时和验证指标。
- `best.pt`、`selection.json`：完整模型权重、输入协议、数据版本、选中步数及 checkpoint 摘要。
- `reload_check.json`：独立恢复模型后与选中结果的一致性检查。

训练命令不自动评测最终测试集。首版固定违规阈值为 0.5，暂不做温度校准；输出概率不宣称已校准。

## 评测与结果范围

恢复选中的模型并导出验证结果：

```sh
python scripts/cpv.py train evaluate --checkpoint runs/training/original-v1/best.pt \
  --split validation --output runs/training/original-v1/validation-report.json
python scripts/cpv.py train evaluate --checkpoint runs/training/mixed-v1/best.pt \
  --split validation --output runs/training/mixed-v1/validation-report.json
```

配置和两组对比方案锁定后，再分别执行最终测试：

```sh
python scripts/cpv.py train evaluate --checkpoint runs/training/original-v1/best.pt \
  --split test --output runs/training/original-v1/test-report.json
python scripts/cpv.py train evaluate --checkpoint runs/training/mixed-v1/best.pt \
  --split test --output runs/training/mixed-v1/test-report.json
```

报告包含准确率、宏 F1、违规召回率、合规误报率、交叉熵、Brier，以及 `[[TN,FP],[FN,TP]]` 混淆矩阵与逐样本概率。违规是正类。指标范围是 **TOD-ProcBench Task 2 自定义分组划分**；合成标签继承及大量未逐条复核的限制仍成立，实际增强收益要由训练结果确认。

代码、模型和报告可通过 [AutoDL 数据下载](https://www.autodl.com/docs/down/)取回。远程训练还需要实际实例的连接信息；当前只完成本地准备，未租用实例或运行完整 0.6B GPU 训练。
