# 跨场景二选一与四选一评测

这组评测检查客服合规训练后的模型在其他规则与逻辑任务上的迁移表现。三组分别为未微调的 Qwen3-0.6B、原始数据一轮模型、混合数据一轮模型。两个已训练模型沿用大小写控制实验的固定检查点；本次不训练、不调阈值、不按外部测试表现选择模型或修改提示词。

## 数据与固定协议

| 测试 | 来源 | 使用范围 | 选择格式 |
| --- | --- | --- | --- |
| 规则推理 | [RuleTaker 官方仓库](https://github.com/allenai/ruletaker)，V2020.2.5 发布包 | Problog / depth-3 / test；固定种子 20261004，抽取 1,000 个不同规则库，每库一题，True/False 各 500；抽样候选为含两类问题的规则库 | A=True、B=False |
| 阅读逻辑 | [LogiQA 2.0 官方仓库](https://github.com/csitfun/LogiQA2.0)，提交 `955e1d3df6c59d9bfb44d9913da1e1a27ec14e18` | 官方英文 test.txt 全部 1,572 行，保留原顺序与选项顺序 | A/B/C/D 对应原四个选项 |

RuleTaker 使用 Problog 校验版本各记录自身的英文规则、问题和布尔标签，不跨文件按编号拼接。遵循闭世界假设：无法推导的正命题为假，其否定为真。证明、形式逻辑和标签均不进入模型输入。LogiQA 的源编号有重复，评测 ID 加上源行号，全部官方记录仍保留；完整输入重复数量和标签冲突数量写入 manifest。

所有模型都将英文证据、问题、选项描述和任务判据转成小写；选项字母仍为大写。沿用 Direct 的系统提示与原生 LM head，关闭 thinking，不生成解释，不使用示例，只在允许的字母上计算下一 token 的 softmax。四选一没有新增分类头。每个字母必须能独立编码为一个 token，且附加答案不能改变提示末尾的分词。

每个任务从主测试中按金标签均衡抽取 128 题做固定换位诊断：二选一交换两个选项；四选一做三种循环移位。换位时同步映射金标签，统计答案内容的一致性、换位准确率与字母分布。诊断结果独立呈现，不混入主测试分数。

OOD 输入上限预先设为 4,096 token，拒绝截断或静默丢题。具体最大长度和所有渲染提示的哈希在首次预测前冻结。该评测是本项目的统一 Direct 协议，不能当作数据集官方论文的标准生成式评测分数。

## 文件与运行

- 模块：`src/customer_policy_verifier/training/ood_choice.py`。
- 入口：`scripts/run_ood_choice.py`。
- 独立复核：`scripts/analyze_ood_choice.py`，检查逐题输入、标签和概率，重算配对统计并核算 LogiQA 去重后的分数。
- 推理配置：`configs/ood-choice-v1.json`。
- AutoDL 固定权重计划：`configs/ood-models-autodl-v1.json`；迁移到另一机器时需要提供指向相同权重的计划文件。
- 本地来源归档：`data/evaluation/sources/`；数据遵循其原始许可，LogiQA 2.0 标示为 CC BY-NC-SA 4.0。
- 固定测试集：`data/evaluation/ood-choice-v1/`，与客服训练数据目录分开。
- 运行结果：`runs/evaluation/ood-choice-v1/`。

先准备固定数据与真实 tokenizer 审计：

```bash
python scripts/run_ood_choice.py prepare --device cpu --precision fp32
```

随后在 AutoDL 的 cpv 环境执行：

```bash
conda activate cpv
python -u scripts/run_ood_choice.py run
```

两条命令均要求新的输出目录，避免覆盖已有结果。运行会先核对全部三份权重 SHA-256、已有选模记录与固定输入哈希，写入 execution.json 后开始预测。保存逐题概率、提示哈希、金标签、混淆矩阵、准确率、macro-F1、受限交叉熵和换位诊断。status.json 记录当前模型与阶段。

## 解释边界

外部测试数据没有参与本项目微调，但公开基准是否进入 Qwen 预训练无法确认，因此不能表述为基础模型绝对没见过。RuleTaker 保留二选一而切换任务和领域；LogiQA 同时切换任务、领域与选项数量，四选一失分可能与 A/B 训练造成的字母偏好有关。换位检查用于暴露这一限制。

二选一均衡随机准确率为 50%，四选一均匀随机准确率为 25%；同时报告实际金标签多数类基线。迁移结果应对照未微调基础模型，不能只因高于随机猜测就声称微调提升了泛化。每组仅一个训练种子，外部评测也是一次固定实验。

2026-10-04 已完成的实际分数、逐题验证和字母偏好诊断见 [结果报告](../reports/ood-choice-results-20261004.md)。
