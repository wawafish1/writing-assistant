# 写作助手

一个面向中文财经内容创作的本地 AI 写作 Agent。它可以扫描热点、调用多种数据源完成研究、沉淀 KOL 写作风格、同时使用 GPT 和 DeepSeek 生成文章，并把你最终修改后的稿件保存为个人风格记忆。Grok 作为可选工具，用于在不改动正文内容的前提下整理 X / Twitter 发布格式。

## 项目定位

这是一个可独立运行的 **Agent 应用**，不是 Codex Skill。它拥有网页界面、数据库、模型调用、外部研究工具和完整的“热点发现 → 证据研究 → 双模型写作 → 人工定稿 → 风格记忆”工作流。

如果以后需要把其中某项能力提供给其他 Agent 调用，可以再把“热点扫描”“风格改写”等单项能力封装成 MCP 工具或 Codex Skill；当前仓库保留完整应用形态更合适。

## 能做什么

- 手动导入 X/Twitter、文章、长推等写作样本。
- 基于样本生成可复用的风格画像。
- 在写作时选择不同风格对象，生成符合该风格结构、节奏和论证方式的原创文章。
- 同时调用 GPT 和 DeepSeek 生成不同版本，单个模型失败不会影响另一个模型返回结果。
- 可选使用 Grok 整理最终稿发布格式，正文内容保持不变。
- 自动研究主题，整理证据简报，再基于证据写分析文章。
- 扫描当日美股、加密热点，选择主题后自动拆解关键词和行情代码并开始研究。
- 支持 Tavily、Finnhub、Alpha Vantage、CoinGecko、OKX、BlockBeats、金十数据、WGD Insight 等资讯/行情/舆情来源。
- 支持参考文本改写，以及“研究内容 + 参考文本”结合写作。
- 支持最终稿记忆学习：你把自己最终修改后的版本保存进去，后续可以选择“我的风格”继续生成。
- 支持平台和篇幅设置：X / Twitter、小红书、公众号、知乎、即刻；标准、精简、展开；以及具体字数限制。

## 重要边界

- 本项目是写作辅助工具，不用于冒充任何作者。
- 风格画像只学习结构、节奏、语气、论证方式和平台表达技巧。
- 不应复制 KOL 原句、独特口头禅、可识别段落或专属表达。
- 自动研究得到的证据只作为写作支撑，模型仍可能出错，发布前需要人工核对。
- `.env` 里的 API Key 不要提交到 GitHub。

## 项目结构

```text
app/
  main.py          # FastAPI 页面和接口
  static/          # 全站视觉样式
  config.py        # 环境变量配置
  db.py            # SQLite 数据库读写
  llm.py           # GPT / Grok / DeepSeek 调用
  style.py         # 风格分析和文章生成提示词
  research.py      # Tavily / Finnhub / Alpha Vantage / BlockBeats / WGD 研究聚合
  hot_topics.py    # 美股/加密热点扫描、聚类和行情信号
  wgd_insight.py   # WGD Insight 美股公开舆情客户端
  jin10_mcp.py     # 金十数据 MCP 客户端
  cli.py           # 命令行工具
docs/
  conversation_migration.md
data/
  twitter_style.db # 本地数据库，已被 .gitignore 排除
```

## 本地启动

Windows PowerShell：

```powershell
git clone https://github.com/wawafish1/writing-assistant.git
cd writing-assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `.env`，填入你自己的 Key：

```text
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5

XAI_API_KEY=
XAI_BASE_URL=https://api.x.ai/v1
XAI_MODEL=grok-4

DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

HOT_TOPIC_PROVIDER=deepseek
HOT_TOPIC_MODEL=deepseek-chat

TAVILY_API_KEY=
ALPHA_VANTAGE_API_KEY=
FINNHUB_API_KEY=
BLOCKBEATS_API_KEY=
JIN10_MCP_TOKEN=
```

启动服务：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

项目路径包含中文时，Windows 上的 `uvicorn --reload` 可能无法可靠监听文件变化，因此日常使用建议采用上面的稳定启动命令。

打开页面：

```text
http://127.0.0.1:8010/research
```

接口文档：

```text
http://127.0.0.1:8010/docs
```

## 常用页面

- `/capture`：逐篇粘贴样本。
- `/import`：批量导入样本。
- `/compare`：对已经导入的样本进行风格分析。
- `/write`：不启用自动研究的普通双模型写作工作台。
- `/research`：热点扫描 + 自动研究 + 双模型生成 + 参考文本改写 + 最终稿记忆学习。

## 推荐使用流程

1. 打开 `/import` 或 `/capture`，导入某个 KOL 或你自己的文章样本。
2. 打开 `/compare`，选择样本数量并生成风格画像。
3. 打开 `/research`，选择风格对象。
4. 点击“扫描今日热点”，选择美股、加密或综合领域。
5. 点击某个热点的“选择并研究”，系统会自动填写主题、关键词、美股代码、加密代码、金十参数，并生成证据简报。
6. 也可以自己输入主题，点击“根据主题生成参数”，生成后继续手动修改，再点击“自动研究”。
7. 设置平台、篇幅倾向和长度。
8. 点击“双模型生成”，得到 GPT 和 DeepSeek 两版文章。
9. 在最右侧最终稿区域综合改写。
10. 点击“保存到记忆”，把最终成稿沉淀为自己的风格。

## 平台和长度设置

研究页里有三项：

- 平台：决定发布语境，例如 X / Twitter、小红书、公众号、知乎、即刻。
- 篇幅倾向：只决定相对节奏，例如标准、精简、展开。
- 长度：最终字数约束，例如 `300字以内`、`800字左右`、`1200-1800字`。

如果篇幅倾向和长度冲突，以长度为准。

## 环境变量说明

```text
OPENAI_API_KEY=OpenAI 官方 API Key
OPENAI_BASE_URL=OpenAI 官方 API 地址；使用官方服务时可以留空
OPENAI_MODEL=OpenAI 官方模型名，例如 gpt-5

XAI_API_KEY=xAI 官方 API Key
XAI_BASE_URL=xAI 官方 API 地址，默认 https://api.x.ai/v1
XAI_MODEL=xAI 官方模型名，例如 grok-4

DEEPSEEK_API_KEY=DeepSeek 官方 API Key
DEEPSEEK_BASE_URL=DeepSeek 官方 API 地址，默认 https://api.deepseek.com
DEEPSEEK_MODEL=DeepSeek 官方模型名，例如 deepseek-chat

HOT_TOPIC_PROVIDER=热点聚类使用的供应商，默认 deepseek
HOT_TOPIC_MODEL=热点聚类专用快速模型，默认 deepseek-chat

TAVILY_API_KEY=Tavily 搜索 Key
ALPHA_VANTAGE_API_KEY=Alpha Vantage 行情 Key
FINNHUB_API_KEY=Finnhub 行情/新闻 Key
BLOCKBEATS_API_KEY=BlockBeats 资讯 Key
JIN10_MCP_URL=金十 MCP 地址
JIN10_MCP_TOKEN=金十 MCP Bearer Token

DATABASE_PATH=SQLite 数据库路径
```

实际接入说明：

- DeepSeek 当前按官方 API 配置，默认使用 `https://api.deepseek.com`。
- OpenAI 和 xAI/Grok 可以使用官方 API，也可以按自己的服务条件改成兼容 OpenAI 格式的中转服务。
- 如果使用官方 API，通常只需要填写各家的 `API_KEY` 和模型名；`OPENAI_BASE_URL` 可以留空，`XAI_BASE_URL` 使用示例中的官方地址即可。
- 如果使用中转服务，请根据中转服务文档修改对应的 `BASE_URL`、`API_KEY` 和 `MODEL`，不要把真实 Key 提交到 GitHub。
- CoinGecko 和 OKX 当前使用无需 Key 的公共行情接口，不需要额外填写环境变量。
- WGD Insight 使用无需 Key 的公开接口，为美股热点和自动研究补充平台关注排名、情绪分及讨论量变化。公开接口可能限流，系统会在失败时自动跳过；其平台热门排名不等于全网热度、涨幅榜或投资推荐。
- 为避免直接复用第三方 AI 报告，项目只读取 WGD Insight 的结构化舆情指标和标题，不把完整 AI 摘要复制到生成材料中。

## 数据和安全

这个 GitHub 仓库只保存空白项目代码，不包含你的本地使用数据。首次在新电脑或服务器使用时，需要重新配置 `.env`，并重新导入样本、生成风格画像、保存个人最终稿记忆。

`.gitignore` 默认排除了：

```text
.env
.venv/
data/
__pycache__/
```

因此以下内容不会被提交：

- API Key
- 本地虚拟环境
- SQLite 数据库
- 已导入样本
- 风格画像
- 最终稿记忆
- Python 缓存

如果部署到线上，务必加登录保护，否则别人拿到网址就可以消耗你的模型和资讯 API 额度。

## 命令行用法

```powershell
python -m app.cli analyze yijiangren --sample-limit 120
python -m app.cli generate yijiangren "主题：AI 算力周期发生了什么变化？"
```

现在主要推荐使用网页界面，命令行功能作为补充。
