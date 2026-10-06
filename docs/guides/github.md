# 准备 GitHub 版本

项目远程仓库为 [SolidifyTime/jev-customer-policy-verifier](https://github.com/SolidifyTime/jev-customer-policy-verifier)。本地公开文件、中英文 README 和对应的矢量图已准备完成；可按下列步骤上传代码与公开索引。GitHub 默认展示英文 `README.md`，中文版本为 `README.zh-CN.md`。

## 提交范围

可以提交 `src/`、`scripts/`、`configs/`、`docs/`、`tests/`、`requirements/`、`LICENSES/`、`pyproject.toml`、首页、CI、忽略规则和不含实际密钥的 `.env.example`。data 与 runs 仅提交以下四个索引文件：

```text
data/README.md
data/catalog.json
runs/README.md
runs/catalog.json
```

真实源数据、训练/生成数据、复核清单、API 请求与响应、模型权重、checkpoint、AutoDL 压缩包、环境和密钥均留在本地。上传包包含训练数据，不能作为 GitHub 附件一起发布。先单独明确源数据、合成数据和模型的分发条件；SemIf 的 MIT 与历史 AgentJev 的 Apache-2.0 代码许可不能代表这些材料具备同一分发许可。

公开 checkout 不包含冻结训练包，因此首次克隆不能直接运行完整数据检查或训练。首页与 [训练指南](jev-training.md)说明了本地恢复位置；自己的训练实例使用单独的 AutoDL 上传包。

## 先检查本地提交

当前项目已经是独立 Git 仓库，主分支为 `main`。在本项目根目录检查提交与远程状态即可：

```sh
git rev-parse --show-toplevel
git status --short
git remote -v
```

历史纯源码快照位于本地 `runs/packages/github/customer-policy-verifier-source.tar.gz`；它早于本次中英文文档更新，本次上传以当前项目目录为准。如果从压缩包解压到全新目录开始，那个目录不含 `.git`，先在那里执行初始化；当前原项目无需重复此步骤：

```sh
git init -b main
```

在独立项目根目录选择性暂存，检查暂存文件列表及内容：

```sh
git add README.md README.zh-CN.md pyproject.toml src scripts configs docs tests requirements LICENSES .gitignore .env.example .github
git add data/README.md data/catalog.json runs/README.md runs/catalog.json
git diff --cached --name-only
git diff --cached
```

暂存清单应与上述提交范围一致，不应包含真实数据、模型、日志或实际凭证。

运行离线检查和项目入口帮助；训练相关数值检查还需要训练依赖：

```sh
python3 scripts/cpv.py --help
.venv/bin/python -m unittest discover -s tests -v
```

本次上传前，本地运行 **126 项测试**：**106 项通过，20 项因缺少训练依赖跳过**。GitHub CI 会安装固定的 CPU 训练依赖，再运行完整离线测试。正式 AutoDL GPU 训练及独立评测已经完成，结果见 [最终评测报告](../reports/case-neutral-1epoch-results-20261004.md)。`train check` 和来源重建命令需要另行准备相应数据，这与运行离线测试不同。核对完成后创建本地提交：

```sh
git commit -m "Organize customer policy verifier project"
```

## 连接远程仓库并上传

本项目的 GitHub 仓库已创建。若当前 checkout 尚未配置 `origin`，使用下列地址添加远程；已有 `origin` 时先核对地址，再执行推送：

```sh
git remote add origin https://github.com/SolidifyTime/jev-customer-policy-verifier.git
git branch -M main
git push -u origin main
```

若已经存在 `origin`，先检查并确定要沿用的地址，不盲目替换。推送后核对文件列表、首页导航和 CI 结果，确认仓库没有真实数据、运行输出或密钥。以后更新同样先检查差异，再提交推送。
