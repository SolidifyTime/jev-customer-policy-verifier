# 公益站 API 接入历史记录（2026-10-02）

文档状态（2026-10-03）：此 API 探测方案已归档。后续 OpenRouter 全量生成、7,227 条训练数据和训练入口准备已完成，见[全量完成检查](../reports/bulk-completion-20261003.md)和[训练数据说明](../data/training-data.md)；尚未租用 GPU 开始真实模型训练。

用户提供的地址是 `https://api.kunyou.asia/v1`，曾计划使用 `qwen3.8-max`。用户随后决定因稳定性问题切回 OpenRouter，当前全量流程见 [OpenRouter 批量生成](../guides/bulk-generation.md)。本记录不再作为正在等待解除的阻塞项。

最初用 Python urllib 请求 `GET /v1/models`，返回 HTTP 403 / Cloudflare Error 1010，错误名 `browser_signature_banned`。随后同客户端请求实际 Chat Completions 仍被拒绝。最初仅凭这一客户端失败推断服务不可用，判断不充分。

之后不改变 VPN，改用标准 curl 客户端请求 `POST /v1/chat/completions`，`glm-5.3-flash` 和 `qwen3.8-max` 均返回 HTTP 200。Qwen 在约 3.39 秒内正确回复 `OK`，返回模型标识 `intern/inkstone/qwen3.8-max`，但该名称只代表服务商报告的路由身份，不能核实底层权重。GLM 的短响应发生输出截断。本次结果说明不同客户端的访问表现不同，无法据此把问题归因于 VPN。

上述探测没有验证真实场景提取、批量稳定性或生成质量。服务未返回费用字段，不能将其表述为 API 已报告零费用。

记录目录：

- `runs/archive/api-probes/kunyou-probe-20261002T091047Z/`：最初列表请求失败。
- `runs/archive/api-probes/kunyou-chat-probe-20261002T091856Z/`：Python 实际聊天请求失败。
- `runs/archive/api-probes/kunyou-curl-probe-20261002T092046Z/`：curl 两模型响应。

凭证仍仅保存在 Git 忽略的 `.env.kunyou`，权限为 `0600`，本文和请求记录不包含密钥。当前 OpenRouter 批量客户端不读取它。没有使用公益站启动全量生成。
