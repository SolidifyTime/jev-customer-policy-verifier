# SemIf-OpenJev Direct 全参数监督训练与 AutoDL 运行

2026-10-04 新对比改用 **证据统一小写、1 epoch、10% warmup**，启动与评测见 [大小写控制实验指南](case-neutral-comparison.md)。下文三轮命令与 `jev-training.json` 保留用于历史复现；新实验使用独立配置和运行目录。

当前方案由 **LLaMA-Factory 0.9.5 自定义 Direct Trainer**执行，使用 **完整预训练 `Qwen/Qwen3-0.6B`，按 SemIf-OpenJev Direct 读取 A/B 选项概率，再进行本项目的全参数监督训练**。模型由预训练 Transformer 和原生、与词嵌入共享权重的词表输出头组成；架构与原始 Qwen3ForCausalLM 保持一致。所有模型参数参与训练。

参考 [SemIf-OpenJev 的固定提交](https://github.com/TheoLeeCJ/SemIf-OpenJev/tree/23cf1f39fc9534fe81437200959b6dfc7106e45a) `23cf1f39fc9534fe81437200959b6dfc7106e45a` 的 Direct 提示与概率读取方式。该上游提供的是推理实现，**监督损失、数据适配、选模、重载和评测由本项目实现**，不能把它描述成直接使用上游训练器。

模型从 [ModelScope 官方 Qwen/Qwen3-0.6B](https://www.modelscope.cn/models/Qwen/Qwen3-0.6B) 的当前 `master` 下载，不锁定模型提交或文件哈希。下载后检查配置、tokenizer 和 safetensors 权重是否齐全，再从本地目录加载 `Qwen3ForCausalLM.from_pretrained`，同时恢复预训练骨干及原生词表头。不另设随机评分头，不加载已训练的 Jev、NLI 或其他任务 checkpoint。此前 AgentJev 自定义头、独立候选路径和 CE＋Brier 训练方案已归档，见[历史指南](../archive/agentjev-training-v1.md)与[旧配置](../../configs/archive/agentjev-training-v1.json)。

## 一条样本如何得到概率

模型每次接收一条 chat prompt，而非每个候选各跑一次。system 要求依据证据和判断标准，只回答一个选项字母；user 的 JSON 含 `evidence`、`criterion` 和 `options`。`evidence` 是仅由 `policy`、`conversation`、`candidate_response` 序列化的 JSON 字符串；`criterion` 是固定判断问题，选项固定如下：

| 字母 | 类别 / 目标 ID | 固定选项含义 |
| --- | --- | --- |
| A | `compliant` / 0 | 待审回复满足当前对话步骤适用的规则 |
| B | `violating` / 1 | 待审回复违反适用规则，包括条件、动作或执行顺序 |

例如，以下是说明格式的虚构输入，不写入训练数据：

```json
{
  "policy": "A refund is allowed only within seven days of purchase.",
  "conversation": ["User: I purchased this twenty days ago. Can I get a refund?"],
  "candidate_response": "Agent: Your refund is approved."
}
```

system 的固定指令为：

```text
Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.
```

user JSON 的实际结构如下，`evidence` 内容是序列化后的 JSON 字符串：

```json
{
  "evidence": "{\"candidate_response\":\"Agent: Your refund is approved.\",\"conversation\":[\"User: I purchased this twenty days ago. Can I get a refund?\"],\"policy\":\"A refund is allowed only within seven days of purchase.\"}",
  "criterion": "Does the candidate customer-service response comply with the policy at the current conversation step? Judge the response using the policy and the conversation history; a valid intermediate step need not complete the entire workflow.",
  "options": [
    {
      "letter": "A",
      "description": "Compliant: the candidate response follows the policy requirements that apply at the current conversation step."
    },
    {
      "letter": "B",
      "description": "Violating: the candidate response violates an applicable policy requirement, including a required condition, action, or order of steps."
    }
  ]
}
```

使用原始 Qwen tokenizer 的 `apply_chat_template(..., enable_thinking=False, add_generation_prompt=True)`，编码到 assistant 回答开始位置，**不追加正确答案，不生成或解码回答**。`A`、`B` 必须是不同的单 token，并通过字符往返与 prompt 边界检查。批次使用左侧 padding；模型前向设置 `logits_to_keep=1`，只计算最后位置的下一个 token 词表 logits。

从预训练原生词表头的输出选取 A、B 的 logits，记为 `z_A`、`z_B`，再仅对这两项做 softmax：

```text
[P(compliant), P(violating)] = softmax([z_A, z_B])
P(violating) = exp(z_B) / (exp(z_A) + exp(z_B))
```

这是在指定两个选项之间的归一化概率，完整词表中其他 token 的概率不进入该分母。例如 `z_A=1.2`、`z_B=0.2` 时，`P(violating)` 约为 0.269；这只是计算示例，不代表上述虚构样本的实测预测。当前没有温度校准，不宣称输出概率已校准。

**新方案的词表头已随 Qwen 预训练，参数初始化来源与骨干一致。** 历史 AgentJev 方案曾另加随机初始化的候选 Transformer 和标量评分头；当前保留原始 Qwen 模型结构，通过训练调整全部原生参数。

## 实现与固定数据

- `src/customer_policy_verifier/training/jev_data.py`：五文件数据包核验、三字段证据、Direct chat prompt、A/B token 检查、无截断编码和二分类指标。
- `src/customer_policy_verifier/training/jev_model.py`：完整 Qwen3ForCausalLM 加载、原生最后位置词表输出、A/B logit 读取和选项交叉熵。
- `src/customer_policy_verifier/training/train_jev.py`：统一 CLI、长度审计、小试分发、独立重载与评测、上传打包。
- `src/customer_policy_verifier/training/factory_trainer.py`：LLaMA-Factory Trainer 子类、A/B loss、两项 logits 评测与 CPU collator。
- `src/customer_policy_verifier/training/factory_workflow.py`：框架参数、全参训练、初始验证、选模回调、续训与标准导出。
- `src/customer_policy_verifier/training/backbone_download.py`：通过 ModelScope 下载，检查必要文件，记录来源；不做模型文件哈希锁定。
- `configs/jev-training.json`：当前 `customer-policy-semif-direct-v1` 协议、ModelScope 下载渠道及训练参数。文件名为兼容现有 CLI 保留。

训练损失仅为 A/B 两个选项上的交叉熵。`compliant`、`violating` 的目标分别为 `[1,0]`、`[0,1]`。Brier 仅作为评测指标，不加入训练 loss。`id`、metadata、来源标签解释、场景分析和审核结论均不进入模型输入。

继续使用冻结的 `data/training/task2-v1/` 五个文件，不改写或重新划分：原始基线 2,442 条、混合训练 7,227 条、验证 730 条、测试 790 条，加 `manifest.json`。数据摘要、分组隔离、原始子集一致性和固定评测文件都会校验。manifest 中的创建时路径保留原样；运行时从 `--data-dir` 读取五文件，上传包无需同时携带来源归档。

## 本地检查

仅用 Python 标准库核对冻结数据：

```sh
python3 scripts/cpv.py train check
```

本地 CPU 检查环境使用共同依赖；AutoDL CUDA 环境按后文或 [4090 环境指南](autodl-environment.md)安装：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements/training.txt
```

离线 tiny 小模型只用于训练流程验证：

```sh
.venv/bin/python scripts/cpv.py train smoke --tiny-model --device cpu --precision fp32 \
  --run-dir runs/training/smoke/llamafactory-direct-cpu-v1 --samples 4
```

该模式用随机小型 Qwen3ForCausalLM 和简化 tokenizer 检查梯度、两步训练、选优与保存重载。它与正式预训练初始化不同，只验证实现流程，不能代表 Qwen3-0.6B 效果或 CUDA 显存。

当前启用 TensorBoard 的四样本两步 CPU 小试记录在 `runs/training/smoke/llamafactory-direct-tensorboard-cpu-20261004/`，使用 PyTorch 2.6.0 / LLaMA-Factory 0.9.5 / Python 3.12.14。事件文件、`best.safetensors`、框架检查点、标准导出和独立重载均通过；测试集未评测。此前小试保留在 `runs/training/smoke/llamafactory-direct-torch260-cpu-20261004/` 和 `runs/training/smoke/llamafactory-direct-cpu-20261004/`。已有目录不覆盖，重跑选择新目录。

从 ModelScope 下载当前 tokenizer 并按 **当前 Direct chat template** 检查训练和验证长度：

```sh
.venv/bin/python scripts/cpv.py train download --tokenizer-only
.venv/bin/python scripts/cpv.py train audit \
  --output runs/training/audits/semif-direct-modelscope-input-v1.json
```

任何 prompt 超过 2,048 tokens 都报错，不截断规则、历史或待审回复。2026-10-04 已在新框架依赖环境，按当前 Direct chat template 和真实 ModelScope tokenizer 完成无 HTTP 审计，记录为 `runs/training/audits/llamafactory-direct-input-20261004.json`：

| 分区 | Prompt 数量 | 最长 tokens | 中位数 | P95 |
| --- | ---: | ---: | ---: | ---: |
| 混合训练 | 7,227 | 1,202 | 566 | 814 |
| 原始基线 | 2,442 | 1,176 | 572 | 815 |
| 原始验证 | 730 | 1,045 | 558 | 814 |

原始 tokenizer 的 A/B ID 为 32、33，通过单 token、往返和 prompt 边界检查。当前 torch 2.6.0 完整共同依赖环境已通过 **108 项离线测试，无跳过**，包括受限 CE、累积梯度数值比较、完整三轮样本覆盖、跨轮中断续训、TensorBoard 标量写入、选模重载与导出，详见 [三轮训练与日志验证](../reports/tensorboard-epochs-20261004.md)。此前框架接入、手写训练与 ModelScope 适配的检查保留在历史报告，不能作为 GPU 性能结果。

长度审计只编码训练和验证数据。`check` 会读取测试文件做哈希和分区核验；训练、小试和长度审计不计算最终测试指标。

## 上传 AutoDL

当前方案重新生成上传包：

```sh
python3 scripts/cpv.py train package \
  --output runs/packages/autodl/customer-policy-verifier-training.tar.gz
```

包内含当前源码、CLI、配置、依赖、文档、测试、CI、公开索引、空密钥模板、代码许可，以及冻结五文件训练数据。模型权重在 AutoDL 下载；来源归档、API 请求/响应、历史运行和真实密钥不打包。旁边的 manifest 记录成员与整包摘要；已有包拒绝覆盖。历史 AgentJev 包归档在 `runs/archive/packages/agentjev-v1/`，不要混用旧配置或 checkpoint。源码公开范围见 [GitHub 指南](github.md)。

若已经上传此前 torch 2.6 / cu124 项目，本次只需上传 `runs/packages/autodl/customer-policy-verifier-update.tar.gz` 并覆盖更新源码、配置与依赖；不用重新上传数据或下载模型。具体解压与补装命令见 [增量更新步骤](autodl-environment.md#已上传项目的增量更新)。

按 [AutoDL 上传说明](https://www.autodl.com/docs/scp/)把包放到实例数据盘：

```sh
cd /root/autodl-tmp
tar -xzf customer-policy-verifier-training.tar.gz
cd customer-policy-verifier
```

AutoDL 使用 **RTX 4090 / PyTorch 2.5.1 / Python 3.12 / Ubuntu 22.04 / CUDA 12.4 基础镜像开机**，然后在独立 Conda 环境中安装 **PyTorch 2.6.0 + cu124**。实际训练环境的完整依赖、驱动说明与安装步骤见 [环境指南](autodl-environment.md)。

```sh
nvidia-smi
source /root/miniconda3/etc/profile.d/conda.sh
conda create --prefix /root/autodl-tmp/conda/envs/cpv python=3.12 pip -y
conda activate /root/autodl-tmp/conda/envs/cpv
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements/autodl-cu124.txt
python -m pip check
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python scripts/cpv.py train env --device cuda --output runs/training/audits/autodl-cu124-env.json
python -m unittest discover -s tests -v
python scripts/cpv.py train check
```

后续登录先激活同一 Conda 环境，不重复创建。框架版本、实际 CUDA runtime 与设备以内置自检为准；`nvidia-smi` 显示驱动支持情况，与 torch runtime 是不同信息。当前尚未完成 AutoDL GPU 验证。

先执行不下载正式权重的 tiny CUDA 小试：

```sh
python scripts/cpv.py train smoke --tiny-model --device cuda --precision bf16 \
  --run-dir runs/training/smoke/llamafactory-direct-cu124-tiny-v1 --samples 4 \
  --microbatch-questions 1 --gradient-accumulation 1
```

通过后下载原始 Qwen safetensors，并审计实际 prompt，再进行下面的真实 GPU 小试：

```sh
python scripts/cpv.py train download
python scripts/cpv.py train audit --output runs/training/audits/llamafactory-direct-cu124-input.json
```

下载命令默认使用 ModelScope，目录为 `models/Qwen3-0.6B-modelscope/`。`model_revision=master` 表示获取官方当前版本；SDK 负责常规下载与缓存。项目仅检查必要文件和 Qwen3 配置，原生 Transformers 在加载时检查权重。之后的 tokenizer 审计、训练和评测均使用 `local_files_only=True`，不需要访问 Hugging Face。`huggingface-hub` 仍是 Transformers 的本地运行依赖，安装这个 Python 包不会要求从 Hugging Face 下载模型。

## 先跑真实 GPU 小试

```sh
python scripts/cpv.py train smoke --train-set original --device cuda --precision bf16 \
  --run-dir runs/training/smoke/llamafactory-direct-cu124-qwen-v1 --samples 4 \
  --microbatch-questions 1 --gradient-accumulation 1
```

该模式加载真实 Qwen3-0.6B Transformer 和预训练词表头，进行两步全参数更新，评测少量原始验证数据，再独立加载同一结构、恢复完整权重并比对结果。检查 `run.json` 的初始化来源与全参数数量、`events.jsonl` 的有限 loss、`selection.json` 的显存和 `reload_check.json`。

小试只验证流程和设备资源，不能用其少量验证样本决定最终实验配置。默认 microbatch 1，SDPA 和梯度检查点启用；GPU 不支持 bf16 时可指定 `--precision fp32`。

## 原始与混合数据对照

两组都从同一原始预训练 Qwen3-0.6B 初始化；使用相同模型版本、随机种子、批量配置，各训练完整 **3 个 epoch**：

```sh
python scripts/cpv.py train train --train-set original \
  --run-dir runs/training/llamafactory-direct-original-3epochs-v1
python scripts/cpv.py train train --train-set mixed \
  --run-dir runs/training/llamafactory-direct-mixed-3epochs-v1
```

配置设置 `num_train_epochs=3`、`max_steps=-1`，由框架按数据加载器计算更新次数。单卡、microbatch 1、梯度累积 16 时：

| 训练集 | 每轮样本 | 每轮更新 | 三轮更新 | 三轮样本访问 |
| --- | ---: | ---: | ---: | ---: |
| 原始 | 2,442 | 153 | 459 | 7,326 |
| 混合 | 7,227 | 452 | 1,356 | 21,681 |

每轮末尾不足 16 个微批的窗口也执行一次更新，不丢弃样本。两组遍历次数一致，混合组计算量更大，结果中应同时报告 epoch 和 optimizer steps。若调整批量，实际更新预算以 `run.json` 的 `training_plan` 为准。`--num-train-epochs N` 可覆盖轮数；显式 `--max-steps N` 则切换为更新次数预算，两者不能同时指定。`smoke` 仍只进行最多两次更新。

默认统一学习率 `1e-5`，所有原生参数进入框架 AdamW，权重衰减为 `0.01`，norm/bias 按框架默认不衰减。参数使用 float32，CUDA 前向由 Accelerate 管理 bf16，启用 SDPA、非重入梯度检查点、梯度裁剪、20 步 warmup 与 cosine 调度。

每 100 步及最后一步，根据完整原始验证集交叉熵选择模型，第 0 步也作为候选。训练由自定义 LLaMA-Factory Trainer 执行，不使用普通文本 SFT 的完整词表损失；详细职责见 [接入设计](../design/llamafactory-training.md)。

### 中断后续训

只从同一运行的最新完整检查点恢复，例：

```sh
python scripts/cpv.py train train --train-set original \
  --run-dir runs/training/llamafactory-direct-original-3epochs-v1 \
  --resume-from-checkpoint runs/training/llamafactory-direct-original-3epochs-v1/checkpoints/checkpoint-300
```

将示例中的 300 替换为实际最新完整步数。恢复 optimizer、scheduler、RNG、训练步数和验证选模状态；配置、数据、提示、样本子集、代码和框架版本必须与原运行一致。默认保留最近两个完整检查点。全部更新完成但重载/导出中断时，同一恢复命令可只完成剩余交付步骤。首次保存前中断时，使用新目录重新开始；非空目录未指定恢复参数会拒绝覆盖。

### 输出文件

| 文件 | 含义 |
| --- | --- |
| `run.json` | 配置、数据来源、代码摘要、依赖、硬件、全参数数量、prompt 审计、框架参数与实际更新预算 `training_plan` |
| `initial_validation.json` | 监督更新前的原始模型验证结果 |
| `events.jsonl` | 框架 loss、梯度范数、学习率、epoch、耗时与 `eval_*` 验证指标 |
| `tensorboard/events.out.tfevents.*` | 训练与验证标量曲线；续训会追加事件文件，沿用已恢复的更新步数 |
| `checkpoints/checkpoint-N/` | 完整训练恢复状态；模型为 safetensors，另有 optimizer / scheduler / RNG 与 `cpv_resume.json` |
| `best.safetensors`、`selection.json` | 含第 0 步候选的最优完整 Qwen 权重、完成轮数/步数与冻结选择结果 |
| `reload_check.json` | 独立恢复后，验证交叉熵与选中结果一致 |
| `export/` | 所选模型的标准 Qwen 配置、safetensors、正式 tokenizer 和 `cpv_model.json` |

标准导出可由 `Qwen3ForCausalLM.from_pretrained(..., local_files_only=True)` 加载；评分仍需同一 Direct prompt 和 A/B 读取。tiny 模式导出随机小模型，不导出正式 tokenizer，不能当作训练结果部署。

### TensorBoard 日志

默认 `report_to=["tensorboard"]`，每个实验在自己的 `tensorboard/` 下保存事件文件。训练每 20 次更新记录 `train/loss`、`train/learning_rate`、`train/grad_norm` 和 `train/epoch`，最后一次更新也记录。验证在第 0 步、每 100 步及最后一步记录 `eval/cross_entropy`、`eval/accuracy`、`eval/macro_f1`、`eval/violating_recall`、`eval/compliant_false_positive_rate` 和 `eval/brier_sum` 等标量。选模依据始终是验证集交叉熵。

混淆矩阵保存在 JSON 指标中；TensorBoard 这里只接收标量。查看 AutoDL 实例上的曲线见 [日志检查与访问](autodl-environment.md#tensorboard-日志检查与访问)。

Direct 协议仍为 `customer-policy-semif-direct-v1`，所选 checkpoint 使用 `safetensors-json-v1`，保留原生共享词嵌入。评估入口只读 safetensors 模型；框架续训另外使用自产的 optimizer / scheduler / RNG 状态。历史 `.pt` 与旧 AgentJev 头 checkpoint 单独保留，不自动转换。训练不自动评测最终测试集，违规阈值固定为 0.5，暂不做温度校准。

## 评测与结果范围

恢复所选模型导出验证结果：

```sh
python scripts/cpv.py train evaluate --checkpoint runs/training/llamafactory-direct-original-3epochs-v1/best.safetensors \
  --split validation --output runs/training/llamafactory-direct-original-3epochs-v1/validation-report.json
python scripts/cpv.py train evaluate --checkpoint runs/training/llamafactory-direct-mixed-3epochs-v1/best.safetensors \
  --split validation --output runs/training/llamafactory-direct-mixed-3epochs-v1/validation-report.json
```

锁定配置和对照方案后，分别评测最终测试集：

```sh
python scripts/cpv.py train evaluate --checkpoint runs/training/llamafactory-direct-original-3epochs-v1/best.safetensors \
  --split test --output runs/training/llamafactory-direct-original-3epochs-v1/test-report.json
python scripts/cpv.py train evaluate --checkpoint runs/training/llamafactory-direct-mixed-3epochs-v1/best.safetensors \
  --split test --output runs/training/llamafactory-direct-mixed-3epochs-v1/test-report.json
```

报告包含准确率、宏 F1、违规召回率、合规误报率、交叉熵、Brier、`[[TN,FP],[FN,TP]]` 混淆矩阵和逐样本选项概率。违规是正类，Brier 仅在评测中计算。结果范围为 TOD-ProcBench Task 2 的自定义分组划分；大量合成样本尚未逐条复核，实际增强收益要由正式对照确认。

尚未进行完整 Qwen3-0.6B GPU 训练，没有正式准确率、增强收益或显存实测。训练完成后可按 [AutoDL 下载说明](https://www.autodl.com/docs/down/)取回模型和报告。
