# MemAddr 中文快速部署

[English / 完整接口说明](README.md) · [论文](https://doi.org/10.1016/j.knosys.2026.117296)

MemAddr 用历史用户消息组成按时间排列的 **User-Spine**，把助手回复原文保存在带有稳定地址的组件库中。
模型根据当前问题和 User-Spine 直接选择要读取的地址，运行时校验后返回原文。
不需要额外摘要、Embedding 模型、向量数据库或独立相关性检索器。

这是 Python 方法库和命令行工具，不是模型推理服务。连接云端 API 时，本机不需要 GPU。
需要 Python 3.10 及以上版本、Git 和支持原生工具调用的模型接口。

## 1. 安装并离线验证

在终端依次执行（macOS / Linux）：

```bash
git clone https://github.com/fs3012093583/temporal-spine.git
cd temporal-spine
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
memaddr demo
```

Windows PowerShell 用 `py -m venv .venv` 创建环境，再运行 `.venv\Scripts\Activate.ps1`。

`demo` 使用虚构对话和预设模型动作，不访问网络、不需要密钥、不收费。
输出会包含 `loaded_action_ids: ["result:1"]`、`model_calls: 2` 和 `protocol_followed: true`。
它验证方法链路，不表示模型在真实任务上的效果。
如果提示找不到 `memaddr`，可执行 `python -m temporal_spine demo`。

## 2. 接入真实模型

将下面三个占位符换成你自己的配置。接口必须支持 Chat Completions 的原生函数/工具调用，
不能只提供普通文本生成。

```bash
export OPENAI_API_KEY="你的密钥"
export MEMADDR_BASE_URL="https://你的接口域名/v1"
export MEMADDR_MODEL="你有权限使用的模型ID"

memaddr run \
  --history examples/history.json \
  --question "What dinner did you previously suggest?" \
  --current-date 2026-01-12
```

这一步会向供应商发送可见历史及选中的原文，并产生相应 API 费用。
密钥只放在环境变量里；不要写进代码、历史文件或 Git。
如果密钥变量叫 `PROVIDER_API_KEY`，增加 `--api-key-env PROVIDER_API_KEY`，这里填变量名而不是密钥。

`MEMADDR_BASE_URL` 可以是 `/v1` 根地址，也可以是完整的 `/chat/completions` 地址。
本地兼容推理服务可用 `http://localhost:端口/v1`，忽略鉴权的本地服务可设置 `OPENAI_API_KEY=local`。
远程接口要求 HTTPS。

## 3. 换成自己的对话

参考仓库中的 `examples/history.json`，建立 UTF-8 JSON 文件：

```json
{
  "turns": [
    {
      "turn_id": 0,
      "timestamp": "2026-01-11",
      "user_request": "帮我推荐一个素食晚餐。",
      "assistant_result": "可以试试扁豆咖喱。"
    }
  ]
}
```

`turn_id` 和 `user_request` 必填，回复和日期可省略。
ID 是按时间递增的非负整数，不能重复；追加新对话时保留旧 ID，不能重新编号。
当前问题通过 `--question` 单独提供，不要在历史里放入其参考答案或未来对话。
程序不会修改 JSON 文件；持久化由你的应用负责。

## 4. 常用设置

- 默认 `--access-rounds 1`：最多一次访问，每批最多 5 条。
- `--access-rounds 2`：读取第一批后，模型可再选择第二批；已读取的组件不会重复加载。
- `--access-rounds 0`：只用 User-Spine 回答。
- `--batch-size 3`：把每批上限降到 3 条，允许范围为 1–5。
- 默认 `--max-model-calls 3`：包含访问、最终回答及格式修复，和访问次数不是同一个概念。
- 默认 `--max-tokens 8192`：每次请求的输出上限。部分思考模型会把推理 token 也计入这个上限，
  可按接口限制提高，但也要考虑费用。
- 接口要求 `max_completion_tokens` 时，用 `--token-parameter max_completion_tokens`。
- 模型不支持温度时，用 `--temperature omit`。
- 思考开关是供应商特定参数；仅在接口支持时使用 `--thinking-json '{"type":"disabled"}'`。
- `--strict-tools` 禁止把普通文本回答作为最终结果兜底捕获。

没有自动 HTTP 重试、隐藏的全历史回退或额外付费调用。
模型可以提前回答，不会为凑满预算强行加载。
`protocol_followed: false` 表示出现格式错误或普通文本捕获，需要检查，不能当成严格工具协议成功。

## 5. Python 集成与排错

Python 导入名仍为 `temporal_spine`，完整示例和接口说明见 [README](README.md#python-integration)。
输出包含答案、加载的地址、调用次数、每次用量和协议状态。
Python 的详细 `trace` 含恢复的原文，应视为私密数据；命令行默认不输出这些原文。

HTTP 401/403 通常需检查密钥或模型权限；400/404 需检查地址、模型 ID 和工具参数。
输出被截断时提高输出上限；上下文超限时需要减少输入历史或使用更大上下文模型。
User-Spine 仍随历史增长，MemAddr 不会自动让无限历史适配有限上下文。

本仓库只发布方法、使用示例与必要测试，不包含论文实验数据、模型答卷、评分记录、私有配置或论文文件。
