# 项目结构与输入输出

项目按数据准备、生成、复核、训练分包。统一命令为 `python scripts/cpv.py <分组> <命令>`；安装项目后也可使用 `cpv`。各命令的参数以 `--help` 为准。

## 可执行模块

下表的模块路径均相对 `src/customer_policy_verifier/`。普通检查使用本地文件；生成命令只有显式加入 `--execute` 才调用模型 API，`train download` 从 ModelScope 下载官方模型的当前版本。

| CLI | 模块 | 输入 | 输出与用途 |
| --- | --- | --- | --- |
| `data prepare` | `data/prepare_data.py` | 早期英文 f1 来源 `data/archive/pilot-source/raw/` | 校验字段并固定 `data/archive/pilot-source/splits/` 分组；保留供小试流程使用 |
| `data archive` | `data/prepare_full_data.py` | `data/source/task2/raw/` 的九个官方视图、历史分组与源标签排除记录 | 规范来源、原始三分区、生成排除清单和 manifest；`--verify-existing` 验证已有归档 |
| `data build` | `data/build_training_data.py` | 来源 manifest、原始分区、全量变体和源标签排除项 | `data/training/task2-v1/` 的四个 JSONL 与 manifest；`--verify-existing` 验证冻结结果 |
| `generate pilot` | `generation/pilot.py` | 早期冻结英文来源、两条训练源与提示词 | `runs/generation/pilot-*` 的选样、请求及阶段结果；无 `--execute` 只准备请求 |
| `generate batch` | `generation/batch_experiment.py` | 冻结来源、2–50 条训练源、模型与预算 | `runs/generation/batch-*` 的阶段缓存、summary 和可恢复候选；默认关闭模型审核 |
| `generate expand` | `generation/expand_scenes.py` | 已完成 batch 的解析结构及原始样本 | `runs/generation/expand-*` 的追加变体、summary 和复核队列；`--recheck` 仅重查已存输出 |
| `generate bulk` | `generation/bulk_generate.py` | `data/source/task2/manifest.json`、训练候选、版本和预算 | `runs/generation/` 下的选样、每源解析/生成槽位、用量与完成状态；支持显式恢复 |
| `review report` | `review/report_batch.py` | 已完成 batch 的 selection、summary、阶段输出 | 原运行目录中的 `aggregate.json` 与复核队列，便于了解早期实验 |
| `review prepare` | `review/prepare_manual_review.py` | 已完成 batch 与 expansion | 指定目录中的统一 `review.md` 和来源 manifest；历史逐条复核入口 |
| `review import` | `review/import_manual_review.py` | 已填写的统一复核清单与原始生成证据 | 接收变体、源审计、决策和 baseline 排除项；已填写文件不覆盖 |
| `review export` | `review/export_candidates.py` | 已完成 batch/expansion/bulk、已知排除项 | 指定目录中的候选、默认三条抽查、质量状态和 `training_variants.jsonl`；回填不调用 API |
| `train env` | `training/check_environment.py`（由 CLI 直接分发） | 当前依赖和所选 CPU/CUDA 设备，无需模型或数据 | 固定依赖、Qwen3/SDPA/safetensors、CUDA/bf16、实际 LF 续训与 TensorBoard 写入探针 |
| `train check` | `training/train_jev.py` | 固定五文件训练包、训练配置 | 控制台完整性和数量结果，标准库即可运行 |
| `train download` | `training/backbone_download.py`（经 `train_jev.py` 分发） | ModelScope 官方仓库当前 `master`、目标目录 | `models/Qwen3-0.6B-modelscope/` 与简要来源记录；`--tokenizer-only` 仅取配置/分词器 |
| `train audit` | 同上 | 训练和验证数据、固定 tokenizer | 指定 JSON 长度统计；任何路径超长直接拒绝，不截断 |
| `train smoke` | 同上 | 少量训练数据、完整预训练 Qwen3ForCausalLM 与 Direct 输入 | `runs/training/smoke/` 下两步小试、选优和重载检查；`--tiny-model` 为离线流程验证 |
| `train train` | 同上 | 原始或混合训练集、完整固定验证集 | `runs/training/{llamafactory-direct-original-3epochs-v1,llamafactory-direct-mixed-3epochs-v1}/` 的日志、`best.safetensors`、选择记录和重载检查 |
| `train evaluate` | 同上 | 已锁定的 `selection.json`、完整 checkpoint 和指定评测分区 | 指定 JSON 报告和逐样本概率；训练不自动执行最终测试 |
| `train package` | 同上 | 当前源码/文档/公开索引、配置、固定训练包 | `runs/packages/autodl/customer-policy-verifier-training.tar.gz` 与成员/整包摘要；用于自己的 AutoDL 实例 |

## 公共与库模块

| 模块 | 输入 → 输出 | 职责 |
| --- | --- | --- |
| `scripts/cpv.py`、`cli.py`、`__main__.py` | 命令参数 → 对应模块入口 | 从源码 checkout 或已安装包分发命令；不自行生成数据 |
| `common/paths.py` | 当前项目位置、历史路径、迁移映射 → 实际路径 | 统一 ROOT；读取旧记录时解析搬迁位置，不修改记录 |
| `generation/compatible_api.py` | 配置、请求、schema、预算/并发状态 → 保存响应与用量 | JSON API 客户端、有限重试、限流退避和调用证据 |
| `generation/pilot.py` 的共享函数 | 源材料、场景结构、生成结果 → 提示词/schema/程序问题 | 生成流程共用提取、变体检查和机械约束 |
| `training/jev_data.py` | 固定 JSONL → 三字段证据、Direct chat prompt、A/B token 与目标 | 校验训练包、隔离 metadata、使用非思考 chat template；验证选项 token 与边界，拒绝截断 |
| `training/jev_model.py` | 单 prompt token、左侧 padding / mask → 原生词表 A/B logits；目标 → loss | 完整预训练 Qwen3ForCausalLM，`logits_to_keep=1` 读取最后位置，A/B softmax 和交叉熵；全部参数参与训练 |
| `training/backbone_download.py` | ModelScope 仓库 → 本地模型文件 | 不锁模型提交或文件哈希；检查必要文件，保留纯本地加载能力 |
| `training/factory_trainer.py` | CPU collator、原生 Qwen → A/B loss 与两项 logits | LLaMA-Factory Trainer 子类，保留 Direct 目标、累积归一化、跨轮续训采样与 TensorBoard 标量回调 |
| `training/factory_workflow.py` | 冻结数据、框架参数 → 训练、选模、续训与标准导出 | 框架执行优化、调度与检查点；项目回调保留第 0 步选择 |
| `training/check_environment.py` | 依赖版本、模型接口和设备 → 环境检查报告 | 无模型/数据下载的运行前自检；`train env` 入口 |
| 各目录 `__init__.py` | 包导入 | 声明模块边界；无单独操作入口 |

当前训练协议为 `customer-policy-semif-direct-v1`：`apply_chat_template(enable_thinking=False, add_generation_prompt=True)`，单个 prompt 含证据、criterion 和 A/B 选项，不追加答案。预训练 Transformer 与原生共享词嵌入的词表头一起加载，结构不变；统一学习率 `1e-5`，Brier 仅作为评测指标。上游 Direct 提供推理参考，本项目通过 LLaMA-Factory 自定义 Trainer 实现监督训练。详细示例见 [当前训练指南](guides/jev-training.md)。

模块名 `jev_data.py`、`jev_model.py`、`train_jev.py` 和配置名 `jev-training.json` 为兼容 CLI 保留，不表示继续使用已归档的 AgentJev 自定义头。

正式训练默认三轮，由 Trainer 计算实际更新预算并写入 `run.json.training_plan`；实验目录包含 `tensorboard/` 标量日志、`events.jsonl` 结构化指标、完整恢复检查点和所选模型导出。两步小试保留独立步数预算。

AutoDL 安装使用 `requirements/autodl-cu124.txt`（共同依赖 `training.txt`＋`torch==2.6.0+cu124`）。用 PyTorch 2.5.1 / CUDA 12.4 基础镜像开机，在独立 Conda 环境中训练；目标为 Python 3.12 / Ubuntu 22.04 / RTX 4090 / PyTorch 2.6.0 / CUDA runtime 12.4，顺序及分层验证见 [环境指南](guides/autodl-environment.md)。当前模型读取协议保持 v1，保存格式为 `safetensors-json-v1`：完整权重与 JSON 元数据保存到 `best.safetensors`，保留原生词嵌入共享，不加载历史 `.pt`。

## 本地文件层次

| 目录 | 保存什么 | 处理方式 |
| --- | --- | --- |
| `data/source/task2/` | 官方源文件、README/许可、冻结分组、完整原始分区 | 保留原文和来源哈希，不把多语言/格式视图计作独立场景 |
| `data/derived/` | 当前全量生成批次的筛选、抽查和合成变体导出 | 与来源和运行记录关联，保留已抽查/未逐条复核状态 |
| `data/training/task2-v1/` | 训练实际消费的 7,227/2,442/730/790 四文件与 manifest | 保持已有字节，不重新划分；训练入口首先验证 |
| `data/archive/` | 早期英文来源副本、首批完整复核和轻量小试导出 | 历史证据，不直接重复追加到当前混合训练集 |
| `runs/generation/` | 当前生成入口的计划、选样、请求、响应、恢复与用量 | 全量主批位于 `bulk-openrouter-20261003-v3/`；后续新生成也在此层 |
| `runs/reviews/` | 填写中的或已完成的复核工作目录 | 保留人工结论和填写快照；导出产物放入对应 data 层 |
| `runs/training/` | tokenizer 审计、离线/GPU 小试、基线与混合训练和评测 | 使用 `audits/`、`smoke/`、`llamafactory-direct-original-3epochs-v1/`、`llamafactory-direct-mixed-3epochs-v1/` 区分阶段 |
| `runs/packages/autodl/` | 新 AutoDL 上传包和摘要 | 本地数据传输产物，不提交到 GitHub |
| `runs/packages/github/` | 仅含可提交文件的源码快照和摘要 | 解压后可用于新建独立GitHub仓库；不含实际数据与密钥 |
| `runs/archive/` | 旧生成实验、API 探测与旧交付包 | AgentJev 源码与 AutoDL 包在 `packages/agentjev-v1/{github,autodl}/`；保留版本，不能与新代码混用 |
| `runs/maintenance/` | 本次目录调整的完整性备份和校验记录 | 用于核对搬迁前后文件字节 |
| `models/`、`.venv/` | 本地模型/tokenizer、Python 环境 | 不属于源码版本；按已核定依赖准备；模型默认下载当前版本 |

真实 data、runs、models 内容被 Git 忽略。仓库仅保留 [data/README.md](../data/README.md)、[data/catalog.json](../data/catalog.json)、[runs/README.md](../runs/README.md)、[runs/catalog.json](../runs/catalog.json) 等公开索引。目录说明见相应 README；索引不会替代实际数据包。

## 旧路径如何定位

全部映射保存在 [configs/layout-migration.json](../configs/layout-migration.json)。主要变化如下；旧日期、版本号和文件内容保持原样。

| 原位置 | 当前位置 |
| --- | --- |
| `src/*.py` | 对应 `src/customer_policy_verifier/{data,generation,review,training}/*.py` |
| 最初 `README.md` | `docs/design/project-plan.md`；首页为当前状态 |
| `docs/*.md` | 对应 `docs/{guides,data,reports,design,archive}/` |
| `requirements-training.txt` | `requirements/training.txt` |
| `data/task2/` | `data/source/task2/` |
| `data/processed/bulk-openrouter-20261003-v3/` | `data/derived/bulk-openrouter-20261003-v3/` |
| `data/processed/{manual-review,spot-check}-20261002/` | `data/archive/reviews/` 下同名目录 |
| `data/raw/`、`data/splits/` | `data/archive/pilot-source/{raw,splits}/` |
| `runs/bulk-openrouter-20261003-v3/` | `runs/generation/bulk-openrouter-20261003-v3/` |
| `runs/manual-review-20261002/` | `runs/reviews/manual-review-20261002/` |
| 其他旧小试与探测目录 | 对应 `runs/archive/{pilot,batch,expansion,extraction,api-probes,bulk-preflight}/` |
| 旧 Jev 审计与离线小试 | 仍在 `runs/training/{audits,smoke}/`，仅作已替代方案的历史记录 |
| `runs/autodl/` | `runs/archive/packages/autodl/`；新包放在 `runs/packages/autodl/` |

当前 ModelScope tokenizer 审计位置为 `runs/training/audits/semif-direct-modelscope-latest-input-20261004.json`，当前 LF tiny 小试为 `runs/training/smoke/llamafactory-direct-tensorboard-cpu-20261004/`。旧审计的候选路径长度不能用于单 prompt 的新协议，历史 `customer-policy-jev-v1` checkpoint 也不能加载到新结构。当前 checkpoint 的输入协议另绑定 chat template SHA-256、A/B token ID 和数据来源哈希，加载时核验。旧方案见 [历史训练指南](archive/agentjev-training-v1.md)，配置见 [旧配置](../configs/archive/agentjev-training-v1.json)。

冻结 manifest、请求和模型输出仍可能保存旧绝对路径。`common.paths.resolve_path()` 在读取时映射到当前位置，避免重写后失去原哈希。检查或归档已有数据时优先使用 `--verify-existing`；新的输出选择新的目录。历史配置和提示词版本不同的运行不能只因为路径能解析就直接续跑。
