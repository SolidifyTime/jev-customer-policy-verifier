# ModelScope 下载适配记录

> 本页保留当时的实现与验证记录。当前使用 LLaMA-Factory / torch 2.6.0 / cu124，见 [最新环境记录](autodl-cu124-20261004.md)。

AutoDL 实例无法连接 Hugging Face，因此模型下载改用 [ModelScope 官方 Qwen/Qwen3-0.6B](https://www.modelscope.cn/models/Qwen/Qwen3-0.6B)。按当前需求，直接下载官方 `master` 的当前版本，**不锁定模型提交、不对照预设模型文件哈希**。训练路线继续使用完整预训练 Qwen3ForCausalLM 和 SemIf Direct，全参数训练原生骨干与词表头。

## 下载与本地加载

```sh
python scripts/cpv.py train download
```

默认目录为 `models/Qwen3-0.6B-modelscope/`。`--tokenizer-only` 只取配置和分词器，再次执行完整下载可补齐权重。ModelScope SDK 处理下载与缓存；项目只检查 Qwen3 配置、必要 tokenizer 文件和 safetensors 权重是否存在且非空，分片权重检查索引引用的文件是否齐全。原生 Transformers 在实际加载时检查权重结构。

下载入口不调用 Hugging Face；训练、审计和评测从本地目录以 `local_files_only=True` 加载。已有 SDK 下载目录可以直接通过 `--backbone` 使用，无需本项目的来源清单。`backbone-source.json` 只记录仓库、分支和文件名；历史记录里的固定提交、文件摘要不再作为模型加载门槛。

当前 Python 依赖增加 **`modelscope==1.35.1` 裸包**，保持已核定的 PyTorch 2.5.1 / Python 3.12 环境。其 [官方发布元数据](https://pypi.org/pypi/modelscope/1.35.1/json)提供纯 Python wheel、`Requires-Python>=3.9`；基础依赖为 filelock、packaging、requests、setuptools、tqdm、urllib3，无需安装 `[all]` 或框架扩展。该 SDK 的 [公开下载接口](https://github.com/modelscope/modelscope/blob/v1.35.1/modelscope/hub/snapshot_download.py)支持本项目使用的 `model_id`、`revision`、`local_dir`、`allow_patterns`。

## 已完成检查

2026-10-04 在独立 Mac arm64 CPU / Python 3.12.14 环境实际安装全部固定训练依赖与 ModelScope 1.35.1：

| 检查 | 结果 / 本地记录 |
| --- | --- |
| 依赖安装、`pip check` 与官方 SDK 导入 | 通过；`runs/maintenance/modelscope-20261004/isolated-environment-verification.json` |
| ModelScope 安装是否改变 torch | 单独安装报告仅新增 ModelScope，torch 保持 2.5.1；原项目 `.venv` 保持 2.8.0 |
| 当前完整离线测试 | **98 项通过，无跳过**；`simple-download-tests.log` |
| 真实官方 tokenizer 下载 | 通过；`simple-tokenizer-download.json`。开启 HF 离线模式，并拦截任何 HF HTTP 请求 |
| 下载后完全禁止联网的 tokenizer 审计 | 通过；`simple-local-only-audit.json` |
| Python 3.12 / Linux wheel 解析 | ModelScope 基础 wheel 解析通过；实际 SDK 执行环境为 macOS CPU |

除首行外，表中短记录路径均位于 `runs/maintenance/modelscope-20261004/`。输入审计保存在 `runs/training/audits/semif-direct-modelscope-latest-input-20261004.json`：混合训练 / 原始基线 / 验证最长 **1202 / 1176 / 1045 tokens**，A/B ID 为 **32/33**，无截断，未计算测试集指标。

完整 Qwen3-0.6B 权重未在本机下载，RTX 4090 / CUDA 12.4 也尚未实测。实例准备好后按 [AutoDL 环境指南](../guides/autodl-environment.md)完成完整下载、真实模型小试和正式训练。当前上传包包含新版源码、依赖、文档与冻结训练数据；GitHub 源码包不含真实数据、模型、环境或运行记录。

早先核对两站文件摘要的研究资料及严格校验原型留在本地维护目录，供历史参考，当前源码和配置不使用这些内容。
