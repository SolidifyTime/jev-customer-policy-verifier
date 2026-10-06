# AutoDL RTX 4090 环境与逐步自检

使用一张 RTX 4090，选择 **PyTorch 2.5.1 / Python 3.12 / Ubuntu 22.04 / CUDA 12.4 基础镜像开机**，随后创建独立 Conda 环境，安装 **PyTorch 2.6.0 + cu124**。基础镜像与实际训练环境是两层配置；本项目自检检查的是当前激活环境中的包和实际 CUDA 运行库。

项目通过 LLaMA-Factory 自定义 Direct Trainer 进行全参数训练。训练步骤见 [训练指南](jev-training.md)，实现边界见 [接入设计](../design/llamafactory-training.md)。

## 为什么采用这个组合

[AutoDL 基础环境说明](https://www.autodl.com/docs/base_config/)支持开机后自行安装框架和 Python；[Miniconda 说明](https://www.autodl.com/docs/miniconda/)指出平台镜像均自带 Miniconda。因此无需为了创建独立环境更换镜像，也无需改动镜像原有的 PyTorch 2.5.1。

PyTorch 2.6.0 官方提供 CUDA 12.4 构建，可继续使用当前能开机的基础镜像。[官方安装组合](https://pytorch.org/get-started/previous-versions/#v260)为 torch 2.6.0、torchvision 0.21.0、torchaudio 2.6.0。[LLaMA-Factory 0.9.5](https://github.com/hiyouga/LlamaFactory/blob/v0.9.5/pyproject.toml)要求 Python ≥3.11、torch ≥2.4，当前依赖位于允许范围。

本项目使用 Transformers 4.57.6 的完整检查点恢复。框架加载 optimizer、scheduler 状态时调用的 [版本检查](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/utils/import_utils.py#L1532)要求 torch ≥2.6；仅把模型权重保存成 safetensors 不能替代这部分恢复。因此 2.5.1 用于开机，2.6.0 用于训练。

安装 Python 包不会升级宿主机 NVIDIA 驱动。先确认 AutoDL 允许所选主机与 CUDA 12.4 镜像组合开机；启动后查看 `nvidia-smi`。其中 CUDA 版本反映驱动支持情况，镜像内的 toolkit 与 PyTorch 的 `torch.version.cuda` 是不同信息；最终以真实 CUDA 内核自检为准。[AutoDL CUDA 说明](https://www.autodl.com/docs/cuda/)

## 依赖版本

| 组件 | 版本 | 用途 |
| --- | --- | --- |
| torch | 2.6.0，AutoDL 使用 `+cu124` | 张量计算、SDPA、自动求导 |
| torchvision / torchaudio | 0.21.0 / 2.6.0，AutoDL 使用 `+cu124` | LLaMA-Factory 的声明依赖，与 torch 配套 |
| LLaMA-Factory | 0.9.5 | Trainer、优化器、调度与训练生命周期 |
| Transformers | 4.57.6 | 原生 Qwen3、tokenizer、框架检查点 |
| Accelerate | 1.11.0 | 设备、bf16 和梯度累积 |
| PEFT / TRL | 0.18.1 / 0.24.0 | 框架的声明依赖；本项目训练模式为 `full` |
| datasets / torchdata | 4.0.0 / 0.11.0 | 框架声明依赖；本项目直接适配冻结 JSONL |
| ModelScope | 1.35.1，裸包 | 下载官方 Qwen3-0.6B 当前版本 |
| huggingface-hub / tokenizers | 0.36.0 / 0.22.1 | Transformers 本地加载与编码 |
| safetensors / NumPy / Jinja2 | 0.6.2 / 1.26.4 / 3.1.6 | 模型保存、数值与原始 chat template |
| TensorBoard | 2.21.0 | 每个实验的训练与验证标量事件日志 |

当前使用原生 SDPA、bf16 和非重入梯度检查点。FlashAttention、Liger、DeepSpeed、bitsandbytes 等可选扩展不属于首版安装要求。PEFT、TRL 等包随框架安装，不改变项目的全参数监督训练目标。使用官方 PyTorch wheel 时，无需为这个流程另外升级系统 CUDA toolkit 或安装 cuDNN。

## 开机与安装

上传并解压 AutoDL 包后，在项目根目录执行：

```sh
nvidia-smi
source /root/miniconda3/etc/profile.d/conda.sh
conda create --prefix /root/autodl-tmp/conda/envs/cpv python=3.12 pip -y
conda activate /root/autodl-tmp/conda/envs/cpv
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements/autodl-cu124.txt
python -m pip check
```

Conda 环境放在数据盘，后续每次登录先执行 `source /root/miniconda3/etc/profile.d/conda.sh` 和 `conda activate /root/autodl-tmp/conda/envs/cpv`；已有环境直接激活，不重复创建。安装三件套时保留官方 `cu124` 下载源，其他 Python 包可使用实例可访问的 PyPI 镜像。

[CUDA 依赖文件](../../requirements/autodl-cu124.txt)引入共同依赖，并保留 `torch==2.6.0+cu124`、`torchvision==0.21.0+cu124` 和 `torchaudio==2.6.0+cu124`。不要安装此前 cu128 交付包中的依赖文件。

确认当前解释器与实际运行库：

```sh
python -c "import sys, torch; print(sys.executable); print(sys.version); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

应显示独立环境的 Python 3.12、torch 2.6.0、CUDA runtime 12.4 和 CUDA 可用。基础镜像自带的 torch 2.5.1 不满足当前训练依赖。部分厂商构建的版本标签没有 `+cu124`，自检仍核对实际运行库。

若改用 Miniconda 基础镜像，同样选择能开机的 **Ubuntu 22.04** 镜像并执行上述独立环境安装；不依赖镜像自带的旧 Python。该选择也不会改变宿主机驱动。

## 依赖与框架自检

```sh
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python scripts/cpv.py train env --device cuda \
  --output runs/training/audits/autodl-cu124-env.json
python -m unittest discover -s tests -v
```

`train env` 不需要数据或预训练权重。它核对共同依赖、Ubuntu 22.04 / Python 3.12 / 单张 RTX 4090 / torch 2.6.0 / CUDA runtime 12.4 / bf16，验证 tiny Qwen 的原生 SDPA、全参数反传、AdamW 和共享权重保存重载，再实际运行 LLaMA-Factory Direct Trainer 的梯度累积、检查点续训与 TensorBoard 写入。JSON 中 `probe.tensorboard_scalars_saved=true` 表示事件已写入并读回；`gpu_verified` 只在 CUDA 探针通过后为 true。报告文件已有时使用新的输出名。

## 已上传项目的增量更新

本次增量包适用于此前已上传的 torch 2.6 / cu124 项目，包含变更后的公开文件，不包含训练数据、模型或运行输出。将 `customer-policy-verifier-update.tar.gz` 上传至 `/root/autodl-tmp/`；它的顶层目录为 `customer-policy-verifier/`，与此前完整包一致。若项目解压在其他位置，将下面的路径替换为实际位置。

```sh
cd /root/autodl-tmp
tar -xzf customer-policy-verifier-update.tar.gz
cd customer-policy-verifier
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda/envs/cpv
python -m pip install -r requirements/autodl-cu124.txt
python -m pip check
python scripts/cpv.py train env --device cuda \
  --output runs/training/audits/autodl-cu124-tensorboard-env.json
python scripts/cpv.py train check
```

这一步补装 TensorBoard 并同步三轮训练配置；已有独立环境直接激活，尚未安装环境时先完成上面的开机与安装步骤。正式原始/混合训练命令见 [训练指南](jev-training.md#原始与混合数据对照)。

## TensorBoard 日志检查与访问

训练启动后，在项目根目录检查事件文件和标量：

```sh
python -m tensorboard.main --inspect \
  --logdir runs/training/llamafactory-direct-original-3epochs-v1/tensorboard
```

小试或混合实验改为对应目录。第 0 步就会写入验证标量；正式训练的 loss 每 20 次更新写入，最后一次更新也会写入。续训追加新的事件文件并延续 global step。

[AutoDL 内置 TensorBoard](https://www.autodl.com/docs/tensorboard/)默认读取 `/root/tf-logs/`，本项目事件文件位于各实验的 `tensorboard/`。使用项目目录启动独立服务，可以同时比较两组实验：

```sh
python -m tensorboard.main --logdir runs/training --host 127.0.0.1 --port 6007
```

让该终端保持运行。在本机另开终端，将下列 SSH 端口、用户名和主机替换为实例提供的实际连接信息：

```sh
ssh -N -L 6007:127.0.0.1:6007 -p SSH_PORT USER@SSH_HOST
```

随后用本机浏览器打开 `http://127.0.0.1:6007/`。该方式使用 SSH 隧道，依据 [AutoDL SSH 代理说明](https://api.autodl.com/docs/ssh_proxy/)，不占用平台已有的 6006 服务。只查看正式两组实验时可将 `--logdir` 改为它们共同的专用父目录。

## 数据与 GPU 小试

```sh
python scripts/cpv.py train check
python scripts/cpv.py train smoke --tiny-model --device cuda --precision bf16 \
  --run-dir runs/training/smoke/llamafactory-direct-cu124-tiny-v1 --samples 4 \
  --microbatch-questions 1 --gradient-accumulation 1
python scripts/cpv.py train download
python scripts/cpv.py train audit \
  --output runs/training/audits/llamafactory-direct-cu124-input.json
python scripts/cpv.py train smoke --train-set original --device cuda --precision bf16 \
  --run-dir runs/training/smoke/llamafactory-direct-cu124-qwen-v1 --samples 4 \
  --microbatch-questions 1 --gradient-accumulation 1
```

模型从 [ModelScope 官方仓库](https://www.modelscope.cn/models/Qwen/Qwen3-0.6B)的当前 `master` 下载到 `models/Qwen3-0.6B-modelscope/`，无需连接 Hugging Face。下载检查必要文件，后续所有模型和 tokenizer 均为本地加载；不锁模型提交或预设文件哈希。

检查 `run.json`、`events.jsonl`、`selection.json` 和 `reload_check.json`，确认框架版本、全参数数量、有限损失、显存和选中模型重载。通过后按 [训练指南](jev-training.md)运行原始基线与混合数据的完整对照。

## 验证范围与历史

PyTorch 2.6.0 环境的安装依据见 [cu124 环境适配记录](../reports/autodl-cu124-20261004.md)，当前三轮训练与 TensorBoard 检查见 [最新验证记录](../reports/tensorboard-epochs-20261004.md)。Linux 发行元数据检查与本机 CPU 验证分别记录；4090 驱动、CUDA、bf16、完整 0.6B 模型训练、速度和显存需要在 AutoDL 实例上实测。

此前 PyTorch 2.8.0 / cu128 的接入结果保留在 [LLaMA-Factory 接入记录](../reports/llamafactory-20261004.md)，更早的手写 PyTorch 2.5.1 流程保留在 [历史环境记录](../reports/autodl-environment-20261003.md)。旧数据、实验与交付包均保留。

选中模型仍使用 `best.safetensors`。完整续训检查点还包含框架的 optimizer、scheduler 和 RNG 状态；标准原生模型导出位于实验的 `export/`。PyTorch 2.6.0 满足当前 Transformers 的恢复加载版本要求。
