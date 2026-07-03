# 写作助手

一个面向中文内容创作的本地写作工作台。它可以沉淀 KOL 写作风格、采集财经/科技/加密相关证据、同时调用 GPT、Grok、DeepSeek 生成三版文章，并把你最终修改后的稿件保存为个人风格记忆。

## 能做什么

- 手动导入 X/Twitter、文章、长推等写作样本。
- 基于样本生成可复用的风格画像。
- 在写作时选择不同风格对象，生成符合该风格结构、节奏和论证方式的原创文章。
- 同时调用 GPT、Grok、DeepSeek 三个模型生成不同版本。
- 自动研究主题，整理证据简报，再基于证据写分析文章。
- 支持 Tavily、Finnhub、Alpha Vantage、BlockBeats、金十数据等资讯/行情来源。
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
  config.py        # 环境变量配置
  db.py            # SQLite 数据库读写
  llm.py           # GPT / Grok / DeepSeek 调用
  style.py         # 风格分析和文章生成提示词
  research.py      # Tavily / Finnhub / Alpha Vantage / BlockBeats 研究聚合
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
cd C:\Users\87271\.codex\project\写作助手
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

TAVILY_API_KEY=
ALPHA_VANTAGE_API_KEY=
FINNHUB_API_KEY=
BLOCKBEATS_API_KEY=
JIN10_MCP_TOKEN=
```

启动服务：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8010
```

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
- `/compare`：用 GPT、Grok、DeepSeek 分析同一批样本，生成三套风格画像。
- `/write`：普通三模型写作工作台。
- `/research`：自动研究 + 三模型生成 + 参考文本改写 + 最终稿记忆学习。

## 推荐使用流程

1. 打开 `/import` 或 `/capture`，导入某个 KOL 或你自己的文章样本。
2. 打开 `/compare`，选择样本数量，生成 GPT、Grok、DeepSeek 三种风格画像。
3. 打开 `/research`，选择风格对象。
4. 输入主题、关键词、股票代码、金十关键词等研究参数。
5. 点击“自动研究”，生成证据简报。
6. 设置平台、篇幅倾向和长度。
7. 点击“三模型生成”，得到 GPT、Grok、DeepSeek 三版文章。
8. 在最右侧最终稿区域综合改写。
9. 点击“保存到记忆”，把最终成稿沉淀为自己的风格。

## 平台和长度设置

研究页里有三项：

- 平台：决定发布语境，例如 X / Twitter、小红书、公众号、知乎、即刻。
- 篇幅倾向：只决定相对节奏，例如标准、精简、展开。
- 长度：最终字数约束，例如 `300字以内`、`800字左右`、`1200-1800字`。

如果篇幅倾向和长度冲突，以长度为准。

## 环境变量说明

```text
OPENAI_API_KEY=GPT 接口 Key
OPENAI_BASE_URL=OpenAI 官方 API 地址；使用官方服务时可以留空
OPENAI_MODEL=GPT 模型名

XAI_API_KEY=Grok 接口 Key
XAI_BASE_URL=xAI 官方 API 地址，默认 https://api.x.ai/v1
XAI_MODEL=Grok 模型名

DEEPSEEK_API_KEY=DeepSeek 接口 Key
DEEPSEEK_BASE_URL=DeepSeek 官方 API 地址，默认 https://api.deepseek.com
DEEPSEEK_MODEL=DeepSeek 模型名

TAVILY_API_KEY=Tavily 搜索 Key
ALPHA_VANTAGE_API_KEY=Alpha Vantage 行情 Key
FINNHUB_API_KEY=Finnhub 行情/新闻 Key
BLOCKBEATS_API_KEY=BlockBeats 资讯 Key

JIN10_MCP_URL=金十 MCP 地址
JIN10_MCP_TOKEN=金十 MCP Bearer Token

DATABASE_PATH=SQLite 数据库路径
```

如果使用官方 API，通常只需要填写各家的 `API_KEY` 和模型名；`OPENAI_BASE_URL` 可以留空，`XAI_BASE_URL` 与 `DEEPSEEK_BASE_URL` 使用示例中的官方地址即可。

## 数据和安全

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
- Python 缓存

如果部署到线上，务必加登录保护，否则别人拿到网址就可以消耗你的模型和资讯 API 额度。

## 命令行用法

```powershell
python -m app.cli collect yijiangren --limit 500
python -m app.cli analyze yijiangren --sample-limit 120
python -m app.cli generate yijiangren "主题：AI 算力周期发生了什么变化？"
```

现在主要推荐使用网页界面，命令行功能作为补充。
