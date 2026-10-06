# 本地数据构建小试运行

文档状态（2026-10-03）：这是 2026-10-02 两样本预试验的历史记录，保留当时的模型审核流程。后续已完成全量生成和 7,227 条训练数据准备，见[全量完成检查](bulk-completion-20261003.md)和[训练数据说明](../data/training-data.md)。训练入口已准备，尚未租用 GPU 开始真实模型训练；当前批量流程默认关闭模型自动审核。

这个试运行只处理英文 f1 的两条训练样本：一条合规、一条违规，来自不同的原始对话。Mac 负责数据读取、分区、调用 API、程序检查和保存记录；生成模型由 OpenRouter 提供，不需要在本机加载大模型。

## 已核对的源文件

- 官方 [数据目录](https://drive.google.com/drive/folders/1aNzCjbYNKUvesqaygrUODV_RlZh0vpTs)中的 `test_task2_en_f1.json`。
- 文件 SHA-256：`5e5402bafdae2ed2c63d6688e161c9e4af8028f596da30dd26b33a3544846b17`。
- 1,982 条记录，每条含原始合规回复与改写的违规回复，共 3,964 个候选样本；498 个不同的 `orig_conv_id`。
- 74 条改写回复不遵循常见的 `turn N: agent:` 前缀。首批生成选样会跳过它们。这是生成脚本的格式适配限制，不能据此判定原始分类样本无效；最终基线仍保留训练区中 45 条此类候选。
- 官方 [README](https://drive.google.com/drive/folders/12n3z9N1LX3hTNruvKtVaK7XEDEg2G-u5)说明该数据取自 ABCD 的 `test` 分区，许可受 Llama 3.3 CLA 约束。发布原始数据或合成数据前还需单独核对许可。

## 运行

在项目根目录执行：

```sh
mkdir -p data/archive/pilot-source/raw
curl -L -o data/archive/pilot-source/raw/test_task2_en_f1.json 'https://drive.google.com/uc?export=download&id=1itHuTgGiY5J_WOoANAxJIoj_3rqa0ulY'
python3 scripts/cpv.py data prepare
python3 scripts/cpv.py generate pilot
```

`src/customer_policy_verifier/data/prepare_data.py` 会校验字段、计算源文件哈希，并按 `orig_conv_id` 固定 60/20/20 左右的分区。当前结果是训练 299 组、验证 100 组、测试 99 组；对应的样本数分别为 2,444、730、790。`data/archive/pilot-source/splits/en_f1_seed_20261002.json` 保存全部组的分区，之后再次运行只能验证同一结果，不能静默改写。

`src/customer_policy_verifier/generation/pilot.py` 默认是无 API 调用的试运行。它从训练区确定性地挑选两条样本，检查源回复格式，保存选样及场景提取请求到 `runs/generation/pilot-*/`。该目录和 `data/` 都被 Git 忽略。

要实际生成，在项目根目录的隐藏文件 `.env` 中填写：

```text
OPENROUTER_API_KEY=你的密钥
```

然后运行：

```sh
python3 scripts/cpv.py generate pilot --execute
```

脚本默认使用 [OpenRouter 的 `qwen/qwen3.7-flash`](https://openrouter.ai/qwen/qwen3.7-flash)，最多对两条样本各执行四次调用：场景提取、变体生成、事实保留审核、独立合规判断。该模型只支持 JSON 输出模式，脚本会在本地检查所有字段结构。各阶段请求、输出、API 用量与失败原因写入 `runs/generation/pilot-*/`。审核通过的变体仅标为 `candidate_for_human_review`，还不是正式训练样本。

当前程序检查能验证来源引用、对话轮次、角色、占位符、对话是否有改写及精确重复；语义审核仍可能出错。当时要求首批结果人工检查事实、未知状态、规则逻辑和标签是否保留，再决定是否扩量。OpenRouter 及所选模型的费用、服务条款与输出使用条件应在正式扩量前再次核对。

## 2026-10-02 的两样本试运行记录

同两条训练样本分别用两个模型试过。以下费用是 API 返回的数值，只覆盖列出的成功返回并已保存用量的调用；一次 Flash 输出截断的调用没有留下用量记录。

| 模型与设置 | 完整记录的调用 | API 报告费用 | 自动结果 | 复核发现 |
| --- | ---: | ---: | --- | --- |
| Qwen3-32B，`pilot-v2` | 8 | $0.00234545 | 2 条进入人工复核 | 两条生成对话都与原文完全相同，只有待审核回复改写；其中一条提取记录错误地把“提供身份资料”写成“身份已核验” |
| Qwen3.7 Flash，默认推理，`pilot-v4` | 3 | $0.00123150 | 0 条 | 一条被误判为不能扩写，另一条把 JSON 字段说明当作答案输出 |
| Qwen3.7 Flash，关闭推理，`pilot-v5` | 8 | $0.00052697 | 1 条进入人工复核，1 条被审核拒绝 | 两条对话均有实际改写；违规样本的保留性审核理由前后矛盾 |

随后仅对 Flash 的同一违规变体使用 `pilot-v6` 保留性审核提示词重审，返回 `pass`，费用 $0.00005560。该提示词明确说明“保留原有违规行为”可以通过事实保留检查。独立合规判断仍是 `violating`。原始 `pilot-v5` 的拒绝记录保持不变，以便追溯；这条变体尚未加入训练集。

这些历史数字不足以判断哪个模型整体质量更高。两次完整运行的提示词和推理设置不同，样本仅有两条。当前可以确认 Flash 关闭推理后的这次调用费用更低，且产出了真正改写对话的候选；是否能稳定保留复杂规则和未知状态，需要扩大首批样本并做人工抽查。程序现在会拒绝“对话完全没改”的变体。
