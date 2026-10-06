# 数据目录索引

正式训练只读取 `data/training/task2-v1/`。此目录首页和 [catalog.json](catalog.json) 会进入 Git；实际数据保留本地，不随源码提交。完整背景见 [数据说明](../docs/data/task2-dataset.md) 和 [训练集说明](../docs/data/training-data.md)。

## 当前数据链路

```text
source/task2（官方快照与固定划分）
    ├── originals → 两条存疑标签排除 → train_original.jsonl
    └── train 来源 → runs/generation → derived/…/training_variants.jsonl
                                      ↓
                           training/task2-v1/train.jsonl
```

| 位置 | 对应内容 | 当前用途 |
|---|---|---|
| `source/task2/raw/` | 9 个语言/格式原始 JSON | 官方归档；英语 f1 是规范来源，其他视图不重复扩充训练 |
| `source/task2/splits/` | 固定分组划分 JSON | 按 `orig_conv_id` 分组，seed=20261002，避免对话跨分区 |
| `source/task2/originals/` | 原始 train / validation / test JSONL | 2444 / 730 / 790 条；原始 train 尚未执行两条标签排除 |
| `source/task2/manifest.json` | 官方文件版本、哈希、划分与统计 | 来源核验 |
| `source/task2/generation_exclusions.json` | 47 条生成排除项 | 45 条前缀问题 + 2 条标签问题；前缀问题不排除原始训练样本 |
| `derived/bulk-openrouter-20261003-v3/` | 当前全量合成候选及3条抽检结果 | 4785 条合成样本，送入当前训练集 |
| `training/task2-v1/` | 最终冻结训练交付 | 当前训练、验证与测试的唯一默认输入 |
| `evaluation/sources/` | RuleTaker 发布包与 LogiQA 2.0 固定英文测试来源 | 外部评测来源归档，仅 Mac 保存完整来源 |
| `evaluation/ood-choice-v1/` | 固定二选一 1,000 题、四选一 1,572 题及换位诊断 | 不参与客服训练；数据和渲染提示哈希已冻结；[评测指南](../docs/guides/ood-choice-evaluation.md) |
| `archive/pilot-source/` | 早期来源及划分副本 | 只用于理解早期实验 |
| `archive/reviews/manual-review-20261002/` | 早期逐条审核、15条接受变体、2条基线排除 | 15条变体未追加；基线排除清单仍参与当前构建 |
| `archive/reviews/spot-check-20261002/` | 早期11条批次及3条抽检 | 未追加到当前训练集 |

## 正式训练目录的五个文件

| 文件 | 样本数 | 用途 |
|---|---:|---|
| `train.jsonl` | 7227 | 混合训练：2442原始 + 4785合成；合成仅3条助手抽检，4782条未逐条审核 |
| `train_original.jsonl` | 2442 | 原始样本基线实验；合规/违规各1221条 |
| `validation.jsonl` | 730 | 验证选模；保持原始来源标签 |
| `test.jsonl` | 790 | 选模后评估；保持原始来源标签 |
| `manifest.json` | — | 输入协议、输入/输出哈希、排除理由、标签/审核状态统计 |

模型只读取 `policy`、`conversation`、`candidate_response`；`label` 是目标，`id` 和 `metadata` 只做溯源。当前 [SemIf-OpenJev Direct 训练方案](../docs/guides/jev-training.md)将三字段证据、criterion 和 A/B 选项组成一条 chat prompt，从完整预训练 Qwen3 的原生词表头读概率；A=合规、B=违规。训练路线变更不改写这五个冻结文件，数据目录与标签映射保持原样。

## 合成批次中的文件

`candidates.jsonl` 是筛选候选，`training_variants.jsonl` 增加来源和审核状态；`summary.json` 解释数量与被排除样本，`manifest.json` 记录来源和哈希。`spot_check.md` / `spot_check_snapshot.md` / `spot_check_results.json` 对应抽检工作表、冻结快照和结论，`quality-check.json` 是导出校验结果。

[文件清单](catalog.json) 列出每个现有数据文件的用途、相对路径、大小、SHA-256，以及 JSONL 行数。目录整理保留已有数据字节与哈希；冻结记录中的旧路径由 `common/paths.py` 和 `configs/layout-migration.json` 在读取时解析，不直接重写数据。

## 检查与构建

在项目根目录运行；这些校验使用 Python 标准库：

```bash
python scripts/cpv.py train check
python scripts/cpv.py data archive
python scripts/cpv.py data build --verify-existing
```

前两层来源数据需要先放在本地，第三条才能重建并比对最终四个 JSONL。已有冻结产物默认校验，不覆盖。AutoDL上传包只带最终五个训练文件，可以运行 `train check`；完整来源重建还需要 source / derived / archive 中的相应输入。

原始快照附有 `source/task2/License` 和上游说明。发布实际数据前须单独确认分发条件，当前 GitHub 准备只包含索引。详情见 [GitHub提交指南](../docs/guides/github.md)。
