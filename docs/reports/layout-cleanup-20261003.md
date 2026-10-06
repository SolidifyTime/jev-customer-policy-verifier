# 2026-10-03 项目目录整理记录

> 本页记录切换模型路线之前的目录整理与验证。当时的 AgentJev 交付包现已归档到 `runs/archive/packages/agentjev-v1/`；当前 SemIf Direct 方案与验证见 [迁移记录](semif-direct-20261003.md)。

已将平铺脚本改成 `src/customer_policy_verifier/` Python包，按 `data`、`generation`、`review`、`training` 分工，`common` 负责公共路径。统一入口为 `python scripts/cpv.py`，安装后也可用 `cpv`。

文档划分为当前指南、数据说明、实验报告、设计背景和历史接口档案；原长README保留在 `docs/design/project-plan.md`，项目首页展示当前状态与入口。每个模块的输入输出见 [项目结构](../project-structure.md)，数据与运行文件见 [数据索引](../../data/README.md)和 [运行索引](../../runs/README.md)。

## 完整性与兼容

- 迁移前已保存本地备份 `runs/maintenance/layout-20261003/before-layout.tar.gz` 和旧文件哈希清单 `before-files.json`。
- 45 个原数据文件、19,851 个原运行产物，共19,896个文件、233,406,496字节，迁移后SHA-256全部一致，没有缺失或内容变更。
- 完整结果保存在同目录的 `after-audit.json`，包含每个文件的新旧路径与哈希。
- 原始manifest、请求、响应和审核记录保留旧路径字符串；读取时用 [迁移映射](../../configs/layout-migration.json)解析当前位置。
- 最终训练manifest的10条来源引用，以及4,785条合成样本的9,570条生成/提取证据引用，均能定位到新目录且哈希符合记录。
- 来源归档和训练数据构建在已有冻结输出时只读核验；复现构建的四个JSONL与当前冻结文件一致。
- 源码重组改变了脚本哈希，旧全量运行不得强行以新代码续跑；后续调用应使用新的运行目录。

## 本地验证

73项离线回归测试全部通过，涵盖数据划分、生成/导出与接口mock、路径迁移、CLI、tiny模型训练、保存重载和打包。仅复制Git可提交文件到隔离目录、移除实际数据与历史运行后，同样73项通过。

全部命令帮助入口可从项目外目录调用；标准库 `train check`、官方九视图归档检查、训练集重建比对和历史审核溯源通过。AutoDL包解压后也可独立执行固定训练数据检查。GitHub CI配置使用Python3.12和CPU依赖运行离线测试；远程CI尚未执行。

尚未创建AutoDL实例，也未进行正式Qwen3-0.6B GPU训练。本次是项目结构与交付整理，没有新增模型效果结论。

## 本地交付

| 文件 | 内容 |
|---|---|
| `runs/archive/packages/agentjev-v1/github/customer-policy-verifier-source.tar.gz` | 当时的代码、配置、文档、测试、许可证、CI和四个公开索引；无实际数据、运行档案、模型、环境或密钥 |
| `runs/archive/packages/agentjev-v1/autodl/customer-policy-verifier-training.tar.gz` | 当时的工程文件与固定五文件训练包；保留历史交付 |
| 压缩包旁的 `.manifest.json` | 压缩包大小及整包、每个成员的SHA-256 |

当前项目是独立本地Git仓库，没有提交或远程地址。本次没有暂存、提交或推送；确定目标仓库后按 [GitHub指南](../guides/github.md)操作。
