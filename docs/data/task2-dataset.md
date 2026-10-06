# Task 2 完整数据归档与分区

当前状态（2026-10-03）：完整归档、全量生成和训练数据准备已完成，统一训练集为 7,227 条，见[训练数据说明](training-data.md)。训练入口已准备，尚未租用 GPU 开始真实模型训练。本文保留归档阶段的来源与分区记录。

2026-10-02 已完整下载作者公开的 Task 2 文件，并固定文件 SHA-256。来源是 [Amazon Science 官方页面](https://www.amazon.science/publications/tod-procbench-benchmarking-complex-instruction-following-in-task-oriented-dialogues)链接的[数据根目录](https://drive.google.com/drive/folders/12n3z9N1LX3hTNruvKtVaK7XEDEg2G-u5)，具体数据位于 [Task 2 目录](https://drive.google.com/drive/folders/1aNzCjbYNKUvesqaygrUODV_RlZh0vpTs)。作者 README 和 License 同步归档在 `data/source/task2/`。

## 完整范围

官方 Task 2 目录共 **9 个 JSON 文件**：英文 `en_f1`、`en_f2`、`en_f3`，以及阿拉伯语、德语、西班牙语、法语、印地语、中文的 `f1`。每个文件都是 **1,982 对原始/违规回复，合计 3,964 条二分类候选、498 个原始会话组、55 个意图**。

完整下载并不意味着把九份文件叠加成九倍独立样本。实际文件核对结果如下：

- `en_f2`、`en_f3` 的行顺序与 `en_f1` 不同；去掉 `instruction` 后，其余五个字段构成的多重集合完全相同。它们提供规则的不同表示格式。
- 六个其他语言文件的每一行 `orig_conv_id`、`intent`、`instruction`、历史轮数都与 `en_f1` 对应。这验证了源样本对齐，没有逐句审核翻译语义。
- 新下载的完整 `en_f1` 与之前本地原文件逐字节相同，SHA-256 为 `5e5402bafdae2ed2c63d6688e161c9e4af8028f596da30dd26b33a3544846b17`。之前的 50 条实验只是从训练区抽取的一次试验，原始文件本身已是完整英文版本。

本轮保留全部九份源文件用于追溯，以完整英文 `en_f1` 的 1,982 对作为生成和评测的规范来源。其他视图不重复计为新场景。

## 固定分区

文件名中的 `test` 是作者发布文件的命名；本项目按 `orig_conv_id` 重新进行自定义分区，不能称作官方训练/测试成绩。同一原始会话的多个轮次、正负回复和后续生成样本归入同一分区。种子为 `20261002`，目标组比例为 60/20/20，**完整保留此前已冻结的分组结果**。

| 分区 | 会话组 | 原始候选 | 合规 / 违规 |
|---|---:|---:|---:|
| 训练 | 299 | 2,444 | 1,222 / 1,222 |
| 验证 | 100 | 730 | 365 / 365 |
| 测试 | 99 | 790 | 395 / 395 |

`data/source/task2/originals/train.jsonl`、`validation.jsonl`、`test.jsonl` 保留原始标签和来源索引。验证、测试原始数据不送入生成流程。

训练区生成输入为 **2,397 条**：1,221 条合规、1,176 条违规。排除 45 条没有可解析 agent 轮次前缀的回复，以及此前人工审核已经排除的两个源标签（1028/violating、1177/compliant）。这些排除仅影响生成输入，原始归档和原始分区文件保留完整数据。详情记录在 `data/source/task2/generation_exclusions.json`。

归档阶段曾计划每个成功解析的源场景生成 10 次。最终全量运行改为每场景 **3 次**：2,022 个解析通过场景完成 6,066 次生成尝试，导出 4,785 条实验变体，详见[全量完成检查](../reports/bulk-completion-20261003.md)。旧计划不代表实际调用量。

## 脚本与重现

在项目根目录执行：

```bash
# 从官方目录下载缺失文件并校验固定版本，不调用模型
python3 scripts/cpv.py data archive --download

# 全离线：验证归档、分区、原始 JSONL 和 manifest 均可重现
python3 scripts/cpv.py data archive

# 与本次归档、分区相关的离线回归校验
python3 -m unittest discover -s tests -p 'test_prepare_full_data.py' -v
```

脚本下载时核对官方目录仍为已固定的九个文件，逐文件校验 SHA-256。若远端目录、文件内容或既有分区改变，明确报错，不覆盖已有结果。已有人工审核文件及历史实验均保留。

生成模块 `src/customer_policy_verifier/generation/bulk_generate.py` 统一读取 `data/source/task2/manifest.json` 中的规范原始文件、文件哈希、固定分区及三个原始 JSONL 路径。归档命令本身不加载密钥、不调用模型；模型生成已在后续独立阶段完成。
