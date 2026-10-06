# OpenRouter 全量数据生成

更新：2026-10-03。**全量已完成：2,397 个源全部处理，导出 4,785 条实验变体，API 报告费用约 $0.90。**结果见[全量完成检查](../reports/bulk-completion-20261003.md)；此前小试见[启动前验证报告](../reports/bulk-preflight-20261003.md)。

当前数据量已足够开展首轮训练，本页保留生成流程说明。目录迁移后的新代码采用统一 CLI；历史 v3 主批保留原请求、参数与代码哈希，不用当前代码在原目录补生成或混合续跑。下方的新批次占位路径只演示操作，不代表仍有待执行的全量任务。

## 数据与规模

官方 Task 2 的 9 个文件已经完整归档，包含英文三种规则格式和另外六种语言。它们共享同一批底层配对样本；本项目以完整英文 f1 的 1,982 对、3,964 个带标签候选为来源，其他版本不重复计入训练。来源、哈希、格式对应关系见 [完整数据说明](../data/task2-dataset.md)。

沿用种子 `20261002` 和原始对话 `orig_conv_id` 的分组划分，同一对话的正反例及不同片段不会跨区。原始导出文件为 `data/source/task2/originals/train.jsonl`（2,444 条）、`validation.jsonl`（730 条）、`test.jsonl`（790 条）。这些是原始档案；前缀异常和标签问题仍保留在档案中，排除清单另存。

生成仅选择训练区中可处理的 2,397 个源候选，合规 1,221、违规 1,176，覆盖 299 个原始对话组；排除 45 条回复前缀异常和 2 条已审核源标签问题。同一配对行的两个标签分别提取；通过检查的场景复用一次提取结果，执行 3 次生成。最多 2,397 次提取＋7,191 次生成＝9,588 次逻辑调用。提取拒绝会减少生成量，3 次尝试不承诺产出 3 条有效变体。相较此前每场景 10 次的计划，生成尝试量减少 70%；原来的待执行目录保留为历史记录，当前使用下方的新目录。

## 准备：无模型调用

在项目目录运行：

```sh
# 新环境：从官方目录下载固定版本的全部 Task 2 文件。
python3 scripts/cpv.py data archive --download

# 本地已有冻结文件时，只验证现有归档。
python3 scripts/cpv.py data archive --verify-existing

# 只创建计划、选样及配置，不读取密钥或调用模型。
python3 scripts/cpv.py generate bulk --run-dir runs/generation/bulk-新批次 --concurrency 8 --variants-per-scene 3
```

查看运行目录的 `plan.json`、`selection.json` 和 `summary.json`。当前提示词版本为 `pilot-v12`。选样文件冻结模型、参数、提示词、实现代码哈希、源数据和分区；变更后会拒绝混合续跑。

## 显式执行新批次

沿用隐藏文件 `.env` 中的 `OPENROUTER_API_KEY`，只使用 OpenRouter 的 `qwen/qwen3.7-flash`。不会读取 `.env.kunyou`。关闭推理，温度 0.2，输出上限 1,800 tokens，使用 JSON mode 并在本地验证结构，沿用 Flash 小试配置。

以下命令包含付费 API 调用，只用于已经准备好的新批次。2026-10-03 的正式 v3 主批已由用户执行完成，原输出现在位于 `runs/generation/bulk-openrouter-20261003-v3/`。

```sh
python3 scripts/cpv.py generate bulk --resume runs/generation/bulk-新批次 --execute --concurrency 8 --max-cost-usd 10
```

`--max-cost-usd` 是本批累计已报告费用的停止阈值，包括续跑前保存的调用。达到阈值后停止派发请求，已有在途请求可能带来少量超出；这不是供应商侧的硬限额。缺少费用字段时先保存输出并暂停，避免将缺失费用当作零。HTTP 200 的模型输出即保存，截断或格式错误也记录用量，不为凑够有效数量无限补采。

历史价格快照在 `configs/openrouter-model.json`。2026-10-02 核对的短上下文单价为输入 $0.03/百万 tokens、输出 $0.13/百万 tokens；这不是当前价格承诺，重新执行前查看 [模型页面](https://openrouter.ai/qwen/qwen3.7-flash)。此前定向小试平均提取约 $0.000134/次、生成约 $0.000095/次，按最大调用量外推约 $1.00，仅是当时的估算。新批次的长度、前文变体和重试会改变费用；命令中的 $10 是停止阈值示例，以实际报告费用为准。

如需先处理少量源，在执行命令追加 `--max-sources 2`，之后去掉此参数续跑即可复用结果。它只限制本次处理的未完成源数量，不改变完整选样。

## 并发、保存与恢复

- 默认 8 个并发请求；429 触发共享退避，并逐次减半至最低 1 个。
- 瞬时网络错误、408、429 和部分 5xx 最多尝试 4 次；鉴权、余额不足或不可恢复错误暂停整批。
- 每源保存到 `sources/00000/` 等目录，含 `extract.json`、`generate-01.json` 至 `generate-03.json`、请求及 `result.json`。结果原子写入并关联请求哈希。
- `summary.json` 定期记录进度、候选数、tokens 及费用。Ctrl+C 后等在途请求保存结束，再用相同 `--resume ... --execute` 命令继续。
- 同一目录有进程锁，避免双进程重复调用。正常续跑只请求未保存阶段；断电或连接断开时，本地无法确认供应商是否已处理未返回请求，重试可能再次计费。

没有模型自动审核调用。只执行字段、引用、格式、历史是否变化及重复检查。

## 全批完成后的 3 条抽查

```sh
python3 scripts/cpv.py review export runs/generation/bulk-新批次 --output-dir data/derived/新批次 --spot-check-count 3
```

填写生成的 `spot_check.md`，关注标签是否成立、是否新增影响判断的事实。**整批只抽查 3 条**，优先覆盖两类标签和不同对话组。导入已填写结论：

```sh
python3 scripts/cpv.py review export --output-dir data/derived/新批次 --apply-spot-check --reviewer human
```

实际由助手抽查时改用 `--reviewer assistant`。已完成的 v3 导出明确记录抽查者为助手，不能用上述 human 示例改变其历史身份。输出 `training_variants.jsonl`，区分已抽查和未逐条复核。详见 [少量抽查说明](spot-check-guide.md)。这一步只准备变体，不开始模型训练；后续合并原始训练材料时沿用已记录的源标签排除项。

## 本地验证

```sh
python3 -m unittest discover -s tests -v
```

测试使用本地 fixture 与模拟 API，覆盖训练分区隔离、同一行两种标签、每场景多次生成、解析失败跳过、途中失败恢复、费用停止、429 降并发、完整性校验和抽查导出。真实模型质量沿用此前 Flash 小试及[本次端到端验证](../reports/bulk-preflight-20261003.md)；仍存在解释偏差和近重复风险。
