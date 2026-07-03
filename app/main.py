from __future__ import annotations

import hashlib
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.db import Database, utc_now
from app.jin10_mcp import Jin10McpClient, Jin10McpError
from app.llm import LlmError, generate_text
from app.research import ResearchError, ResearchSettings, collect_research, format_evidence_brief
from app.style import analyze_style_with_llm, build_heuristic_profile, generate_article_with_llm
from app.x_api import XApiError, XClient


settings = get_settings()
db = Database(settings.database_path)
app = FastAPI(
    title="写作助手",
    description=(
        "按顺序使用：1. 检查配置 2. 采集账号 posts "
        "3. 分析写作风格 4. 根据主题生成原创文章。"
    ),
    version="0.1.0",
)


class CollectRequest(BaseModel):
    username: str = Field(
        ...,
        description="要学习的 X/Twitter 用户名，不需要写 @。",
        examples=["elonmusk"],
    )
    limit: int = Field(500, ge=1, le=500, description="最多采集多少条，1 到 500。")
    exclude_replies: bool = Field(True, description="是否排除回复。新手建议保持 true。")
    exclude_retweets: bool = Field(True, description="是否排除转发。新手建议保持 true。")


class AnalyzeStyleRequest(BaseModel):
    username: str = Field(..., description="已经采集过 posts 的用户名。")
    sample_limit: int = Field(120, ge=10, le=500, description="用多少条 posts 来分析风格。")
    use_llm: bool = Field(True, description="是否使用 OpenAI 分析。false 表示使用本地粗略分析。")
    model: str | None = Field(None, description="可选。留空时使用 .env 里的 OPENAI_MODEL，默认 gpt-5。")


class ImportSamplesRequest(BaseModel):
    username: str = Field(
        ...,
        description="给这批样本起一个名字。可以用原作者用户名，也可以用你自己方便记的名字。",
        examples=["sample_writer"],
    )
    samples: list[str] = Field(
        ...,
        min_length=1,
        description="手动粘贴的文章/推文样本。建议至少 20 条，越多越稳。",
        examples=[
            [
                "第一条样本文本，直接粘贴原文即可。",
                "第二条样本文本，可以是推文、短文或文章段落。",
            ]
        ],
    )
    display_name: str | None = Field(None, description="可选。这个样本集的显示名称。")


class ImportBulkSamplesRequest(BaseModel):
    username: str = Field(
        ...,
        description="给这批样本起一个名字。后续分析和生成都填这个名字。",
        examples=["sample_writer"],
    )
    text: str = Field(..., description="一次性粘贴的大段样本文本。")
    split_mode: str = Field(
        "paragraph",
        description="拆分方式：paragraph 按空行拆，line 按每行拆，separator 按自定义分隔符拆。",
    )
    separator: str | None = Field(None, description="split_mode 为 separator 时使用。")
    display_name: str | None = Field(None, description="可选。这个样本集的显示名称。")


class ImportOneSampleRequest(BaseModel):
    username: str = Field(
        ...,
        description="样本集名称。后续分析和生成都填这个名字。",
        examples=["yijiangren"],
    )
    text: str = Field(..., description="这一篇文章或这一条长推的正文。")
    title: str | None = Field(None, description="可选。文章标题，方便以后回看。")
    source_url: str | None = Field(None, description="可选。原文链接。")
    display_name: str | None = Field(None, description="可选。这个样本集的显示名称。")


class GenerateRequest(BaseModel):
    username: str = Field(..., description="要参考其风格画像的用户名。")
    brief: str = Field(
        ...,
        description="你想写什么。建议写清楚主题、观点、受众。",
        examples=["主题：普通人如何长期坚持健身。主旨：不要靠意志力，要靠系统设计。"],
    )
    platform: str | None = Field(None, description="发布平台，例如：公众号、小红书、即刻、知乎。")
    target_length: str | None = Field(None, description="目标长度，例如：800字左右、1500字左右。")
    extra_constraints: str | None = Field(None, description="额外要求，例如：语气克制、不要鸡汤、分小标题。")
    model: str | None = Field(None, description="可选。留空时使用 .env 里的 OPENAI_MODEL，默认 gpt-5。")


class GenerateThreeRequest(BaseModel):
    style_name: str = Field(..., description="风格对象名称，例如 yijiangren 或 my_style。")
    brief: str = Field(..., description="你想写什么。建议写清楚主题、观点、受众。")
    platform: str | None = Field(None, description="发布平台。")
    target_length: str | None = Field(None, description="目标长度。")
    extra_constraints: str | None = Field(None, description="额外要求。")


class LearnFinalArticleRequest(BaseModel):
    style_name: str = Field("my_style", description="要沉淀到哪个个人风格名称。")
    display_name: str | None = Field("我的风格", description="这个风格在下拉框里的显示名。")
    title: str | None = Field(None, description="可选。最终稿标题。")
    text: str = Field(..., description="你修改后的最终成稿。")
    source_style: str | None = Field(None, description="可选。最初参考的 KOL 风格。")


class AnalyzeThreeModelsRequest(BaseModel):
    username: str = Field(..., description="已经保存样本的用户名，例如 yijiangren。")
    sample_limit: int = Field(48, ge=10, le=500, description="拿多少篇样本做分析。")
    save_profiles: bool = Field(True, description="是否把三家分析结果保存成可用于生成的风格画像。")


class ResearchCollectRequest(BaseModel):
    topic: str = Field(..., description="Research topic")
    keywords: list[str] = Field(default_factory=list, description="General research keywords")
    symbols: list[str] = Field(default_factory=list, description="US stock symbols, for example META, NVDA, MU")
    jin10_keywords: list[str] = Field(default_factory=list, description="Jin10 flash/news keywords")
    jin10_quote_codes: list[str] = Field(default_factory=list, description="Jin10 quote codes, for example XAUUSD, USOIL")


class ResearchWriteThreeRequest(BaseModel):
    style_name: str = Field(..., description="Style object, for example yijiangren or my_style")
    topic: str = Field(..., description="Article topic")
    evidence_brief: str = Field(..., description="Evidence brief generated by research collection")
    platform: str | None = Field(None, description="Publishing platform")
    target_length: str | None = Field(None, description="Target length")
    extra_constraints: str | None = Field(None, description="Extra writing constraints")


class ResearchRewriteThreeRequest(BaseModel):
    style_name: str = Field(..., description="Style object, for example yijiangren or my_style")
    reference_text: str = Field(..., description="Reference text to rewrite")
    platform: str | None = Field(None, description="Publishing platform")
    target_length: str | None = Field(None, description="Target length")


class ResearchReferenceWriteThreeRequest(BaseModel):
    style_name: str = Field(..., description="Style object, for example yijiangren or my_style")
    topic: str = Field(..., description="Article topic")
    evidence_brief: str = Field(..., description="Evidence brief generated by research collection")
    reference_text: str = Field(..., description="Reference text to combine with research")
    platform: str | None = Field(None, description="Publishing platform")
    target_length: str | None = Field(None, description="Target length")
    extra_constraints: str | None = Field(None, description="Extra writing constraints")


class FormatFinalDraftRequest(BaseModel):
    style_name: str = Field("my_style", description="Saved personal style to use as light format reference")
    text: str = Field(..., description="Final draft text to format for X/Twitter")
    platform: str | None = Field("X / Twitter", description="Publishing platform")
    target_length: str | None = Field(None, description="Target length")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> str:
    return """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>写作助手</title>
  <style>
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2937; background: #f6f7fb; }
    main { max-width: 920px; margin: 0 auto; padding: 40px 20px 64px; }
    h1 { margin: 0 0 8px; font-size: 32px; }
    h2 { margin: 32px 0 12px; font-size: 22px; }
    p, li { line-height: 1.75; font-size: 16px; }
    a { color: #0f766e; font-weight: 700; }
    code, pre { background: #111827; color: #f9fafb; border-radius: 6px; }
    code { padding: 2px 6px; }
    pre { padding: 16px; overflow: auto; }
    .panel { background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 20px; margin: 16px 0; }
    .muted { color: #6b7280; }
  </style>
</head>
<body>
  <main>
    <h1>写作助手</h1>
    <p class="muted">照着下面 4 步走：检查配置、采集 posts、分析风格、生成文章。</p>

    <div class="panel">
      <h2>先打开接口操作页</h2>
      <p>常用入口：<a href="/write">文章生成工作台</a>、<a href="/capture">逐篇录入页面</a>、<a href="/compare">三模型分析页</a>。如果你一次性复制了很多内容，也可以用 <a href="/import">批量导入页面</a>。</p>
    </div>

    <h2>第 1 步：检查配置</h2>
    <p>在 <code>GET /health</code> 里点执行。你要看到：</p>
    <pre>{
  "ok": true,
  "has_x_token": true,
  "has_openai_key": true
}</pre>
    <p>如果两个 key 是 false，先在项目里的 <code>.env</code> 文件填好 X 和 OpenAI 的 key。</p>

    <h2>第 2 步：采集账号 posts</h2>
    <p>如果你没有 X API，推荐打开 <a href="/capture">逐篇录入页面</a>，一篇篇粘贴保存；如果一次性复制了很多内容，再用 <a href="/import">批量导入页面</a>。</p>
    <p>也可以在接口文档里打开 <code>POST /samples/import-bulk</code>，填：</p>
    <pre>{
  "username": "sample_writer",
  "text": "这里粘贴很多条样本文本，条目之间用空行隔开",
  "split_mode": "paragraph",
  "display_name": "手动样本"
}</pre>
    <p>如果你有 X API，也可以打开 <code>POST /collect</code> 自动采集：</p>
    <pre>{
  "username": "elonmusk",
  "limit": 100,
  "exclude_replies": true,
  "exclude_retweets": true
}</pre>

    <h2>第 3 步：分析写作风格</h2>
    <p>打开 <code>POST /style/analyze</code>，填：</p>
    <pre>{
  "username": "sample_writer",
  "sample_limit": 50,
  "use_llm": true
}</pre>
    <p>如果你只是想先试试流程，可以把 <code>use_llm</code> 改成 <code>false</code>，这样不调用 OpenAI，但分析会粗略一些。</p>

    <h2>第 4 步：生成文章</h2>
    <p>打开 <code>POST /generate</code>，填：</p>
    <pre>{
  "username": "sample_writer",
  "brief": "主题：普通人为什么很难长期坚持。主旨：问题不在意志力，而在系统设计。",
  "platform": "公众号",
  "target_length": "800字左右",
  "extra_constraints": "不要模仿具体原句，不要冒充原作者"
}</pre>
  </main>
</body>
</html>
"""


@app.get("/capture", response_class=HTMLResponse, include_in_schema=False)
def capture_page() -> str:
    return """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>逐篇录入样本</title>
  <style>
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2937; background: #f6f7fb; }
    main { max-width: 1080px; margin: 0 auto; padding: 32px 20px 64px; }
    h1 { margin: 0 0 8px; font-size: 30px; }
    h2 { margin: 0 0 12px; font-size: 20px; }
    p, label, li { line-height: 1.7; font-size: 16px; }
    a { color: #0f766e; font-weight: 700; }
    .layout { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 18px; align-items: start; margin-top: 18px; }
    .panel { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 18px; }
    .row { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    label { display: block; font-weight: 700; margin: 12px 0 6px; }
    input, textarea { width: 100%; box-sizing: border-box; border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px 12px; font: inherit; background: #fff; }
    textarea { min-height: 420px; resize: vertical; }
    button { border: 0; border-radius: 6px; padding: 11px 16px; font: inherit; font-weight: 800; background: #0f766e; color: #fff; cursor: pointer; }
    button.secondary { background: #334155; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    code, pre { background: #111827; color: #f9fafb; border-radius: 6px; }
    code { padding: 2px 6px; }
    pre { padding: 12px; white-space: pre-wrap; overflow: auto; max-height: 260px; }
    .muted { color: #6b7280; }
    .actions { display: flex; align-items: center; gap: 10px; margin-top: 14px; flex-wrap: wrap; }
    .metric { display: flex; justify-content: space-between; gap: 10px; padding: 10px 0; border-bottom: 1px solid #e5e7eb; }
    .metric strong { font-size: 24px; }
    .recent { display: grid; gap: 10px; margin-top: 12px; }
    .sample { border: 1px solid #e5e7eb; border-radius: 6px; padding: 10px; background: #f8fafc; }
    .sample p { margin: 0; font-size: 14px; }
    .status { min-height: 24px; }
    @media (max-width: 860px) { .layout, .row { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <h1>逐篇录入样本</h1>
    <p class="muted">适合你晚点一篇篇复制文章。保存后会自动清空正文，样本存在本地 SQLite 数据库里。</p>

    <div class="layout">
      <section class="panel">
        <h2>保存一篇文章</h2>
        <div class="row">
          <div>
            <label for="username">样本集名称</label>
            <input id="username" value="yijiangren" />
          </div>
          <div>
            <label for="displayName">显示名称</label>
            <input id="displayName" value="一将人样本" />
          </div>
        </div>

        <label for="title">标题</label>
        <input id="title" placeholder="可选，比如：收购 Iridium：不甘心当小火箭的 RKLB" />

        <label for="sourceUrl">原文链接</label>
        <input id="sourceUrl" placeholder="可选，粘贴 X 链接方便以后回看" />

        <label for="text">正文</label>
        <textarea id="text" placeholder="把一篇文章或一条长推全文粘贴到这里。"></textarea>

        <div class="actions">
          <button id="save">保存并清空正文</button>
          <button class="secondary" id="refresh" type="button">刷新统计</button>
          <span id="status" class="muted status"></span>
        </div>
      </section>

      <aside class="panel">
        <h2>当前样本集</h2>
        <div class="metric">
          <span class="muted">已保存样本</span>
          <strong id="count">0</strong>
        </div>
        <p class="muted">建议先攒 10-20 篇，再去分析风格。长文 5 篇左右也能起步。</p>
        <div class="actions">
          <a href="/docs">去分析 / 生成</a>
          <a href="/import">批量导入</a>
        </div>
        <h2 style="margin-top:22px;">最近保存</h2>
        <div id="recent" class="recent"></div>
        <pre id="raw" style="display:none;"></pre>
      </aside>
    </div>
  </main>

  <script>
    const username = document.querySelector("#username");
    const displayName = document.querySelector("#displayName");
    const title = document.querySelector("#title");
    const sourceUrl = document.querySelector("#sourceUrl");
    const text = document.querySelector("#text");
    const save = document.querySelector("#save");
    const refresh = document.querySelector("#refresh");
    const statusEl = document.querySelector("#status");
    const countEl = document.querySelector("#count");
    const recentEl = document.querySelector("#recent");

    function persistBasics() {
      localStorage.setItem("capture.username", username.value);
      localStorage.setItem("capture.displayName", displayName.value);
    }

    function loadBasics() {
      username.value = localStorage.getItem("capture.username") || username.value;
      displayName.value = localStorage.getItem("capture.displayName") || displayName.value;
    }

    async function loadSummary() {
      persistBasics();
      const name = username.value.trim();
      if (!name) return;
      const response = await fetch(`/samples/${encodeURIComponent(name)}/summary?limit=5`);
      const data = await response.json();
      countEl.textContent = data.count ?? 0;
      recentEl.innerHTML = "";
      for (const item of data.recent || []) {
        const div = document.createElement("div");
        div.className = "sample";
        const p = document.createElement("p");
        p.textContent = item.text_preview || "";
        div.appendChild(p);
        recentEl.appendChild(div);
      }
    }

    async function saveSample() {
      persistBasics();
      save.disabled = true;
      statusEl.textContent = "正在保存...";
      try {
        const payload = {
          username: username.value,
          display_name: displayName.value,
          title: title.value,
          source_url: sourceUrl.value,
          text: text.value
        };
        const response = await fetch("/samples/import-one", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "保存失败");
        }
        text.value = "";
        title.value = "";
        sourceUrl.value = "";
        statusEl.textContent = `保存成功，目前共 ${data.total_samples} 篇`;
        await loadSummary();
        text.focus();
      } catch (error) {
        statusEl.textContent = String(error.message || error);
      } finally {
        save.disabled = false;
      }
    }

    username.addEventListener("change", loadSummary);
    displayName.addEventListener("change", persistBasics);
    refresh.addEventListener("click", loadSummary);
    save.addEventListener("click", saveSample);
    loadBasics();
    loadSummary();
  </script>
</body>
</html>
"""


@app.get("/compare", response_class=HTMLResponse, include_in_schema=False)
def compare_page() -> str:
    return """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>三模型风格分析</title>
  <style>
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2937; background: #f6f7fb; }
    main { max-width: 1040px; margin: 0 auto; padding: 34px 20px 64px; }
    h1 { margin: 0 0 8px; font-size: 30px; }
    h2 { margin: 0 0 12px; font-size: 20px; }
    p, label, li { line-height: 1.7; font-size: 16px; }
    a { color: #0f766e; font-weight: 700; }
    .panel { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 18px; margin-top: 18px; }
    .grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
    label { display: block; font-weight: 700; margin-bottom: 6px; }
    input { width: 100%; box-sizing: border-box; border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px 12px; font: inherit; background: #fff; }
    button { border: 0; border-radius: 6px; padding: 11px 16px; font: inherit; font-weight: 800; background: #0f766e; color: #fff; cursor: pointer; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    pre { background: #111827; color: #f9fafb; border-radius: 6px; padding: 14px; white-space: pre-wrap; overflow: auto; max-height: 520px; }
    .muted { color: #6b7280; }
    .status { min-height: 24px; }
    .cards { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-top: 14px; }
    .card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; background: #f8fafc; }
    .card strong { display: block; margin-bottom: 8px; }
    @media (max-width: 820px) { .grid, .cards { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <h1>三模型风格分析</h1>
    <p class="muted">同一批样本分别交给 GPT、Grok、DeepSeek 分析。没填 key 的模型会自动跳过。</p>

    <section class="panel">
      <div class="grid">
        <div>
          <label for="username">样本集名称</label>
          <input id="username" value="yijiangren" />
        </div>
        <div>
          <label for="sampleLimit">分析样本数</label>
          <input id="sampleLimit" type="number" min="10" max="500" value="48" />
        </div>
        <div>
          <label>&nbsp;</label>
          <button id="run">开始三模型分析</button>
        </div>
      </div>
      <p id="status" class="muted status"></p>
      <div class="cards">
        <div class="card"><strong>GPT 画像名</strong><code id="gptName">yijiangren__gpt</code></div>
        <div class="card"><strong>Grok 画像名</strong><code id="grokName">yijiangren__grok</code></div>
        <div class="card"><strong>DeepSeek 画像名</strong><code id="deepseekName">yijiangren__deepseek</code></div>
      </div>
      <pre id="result" style="display:none;"></pre>
    </section>

    <section class="panel">
      <h2>使用方法</h2>
      <ol>
        <li>先在 <code>.env</code> 里填好三家的 API Key。</li>
        <li>点上面的按钮开始分析。</li>
        <li>分析完成后，去 <a href="/docs">/docs</a> 的 <code>/generate</code>，username 填想用的画像名。</li>
      </ol>
    </section>
  </main>

  <script>
    const username = document.querySelector("#username");
    const sampleLimit = document.querySelector("#sampleLimit");
    const run = document.querySelector("#run");
    const statusEl = document.querySelector("#status");
    const result = document.querySelector("#result");
    const names = {
      gpt: document.querySelector("#gptName"),
      grok: document.querySelector("#grokName"),
      deepseek: document.querySelector("#deepseekName")
    };

    function updateNames() {
      const base = username.value.trim() || "yijiangren";
      names.gpt.textContent = `${base}__gpt`;
      names.grok.textContent = `${base}__grok`;
      names.deepseek.textContent = `${base}__deepseek`;
    }

    username.addEventListener("input", updateNames);
    run.addEventListener("click", async () => {
      run.disabled = true;
      result.style.display = "none";
      statusEl.textContent = "正在分析，三家模型会依次调用，可能需要几分钟...";
      try {
        const response = await fetch("/style/analyze-three", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            username: username.value,
            sample_limit: Number(sampleLimit.value || 48),
            save_profiles: true
          })
        });
        const data = await response.json();
        result.textContent = JSON.stringify(data, null, 2);
        result.style.display = "block";
        statusEl.textContent = response.ok ? "分析完成" : "分析失败";
      } catch (error) {
        statusEl.textContent = "分析失败";
        result.textContent = String(error);
        result.style.display = "block";
      } finally {
        run.disabled = false;
      }
    });
    updateNames();
  </script>
</body>
</html>
"""


@app.get("/write", response_class=HTMLResponse, include_in_schema=False)
def write_v2_page() -> str:
    return """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>文章生成工作台</title>
  <style>
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2937; background: #f6f7fb; }
    main { max-width: 1280px; margin: 0 auto; padding: 32px 20px 64px; }
    h1 { margin: 0 0 8px; font-size: 30px; }
    h2 { margin: 0 0 12px; font-size: 20px; }
    h3 { margin: 0 0 10px; font-size: 16px; }
    p, label { line-height: 1.7; font-size: 15px; }
    .layout { display: grid; grid-template-columns: 380px minmax(0, 1fr); gap: 18px; align-items: start; margin-top: 18px; }
    .panel { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 18px; }
    label { display: block; font-weight: 700; margin: 12px 0 6px; }
    input, select, textarea { width: 100%; box-sizing: border-box; border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px 12px; font: inherit; background: #fff; }
    textarea { min-height: 110px; resize: vertical; }
    button { border: 0; border-radius: 6px; padding: 11px 16px; font: inherit; font-weight: 800; background: #0f766e; color: #fff; cursor: pointer; }
    button.secondary { background: #334155; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    .actions { display: flex; align-items: center; gap: 10px; margin-top: 14px; flex-wrap: wrap; }
    .muted { color: #6b7280; }
    .status { min-height: 24px; }
    .hint { background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 6px; padding: 10px; margin-top: 12px; color: #475569; font-size: 14px; }
    .result-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
    .result-card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; background: #fff; }
    .result-card textarea { min-height: 520px; font-size: 14px; line-height: 1.65; }
    .badge { display: inline-flex; align-items: center; border-radius: 999px; padding: 3px 9px; font-size: 12px; font-weight: 800; background: #e2e8f0; color: #334155; }
    .badge.ok { background: #dcfce7; color: #166534; }
    .badge.err { background: #fee2e2; color: #991b1b; }
    .memory { margin-top: 18px; }
    .memory textarea { min-height: 220px; }
    @media (max-width: 1080px) { .layout, .result-grid { grid-template-columns: 1fr; } .result-card textarea { min-height: 360px; } }
  </style>
</head>
<body>
  <main>
    <h1>文章生成工作台</h1>
    <p class="muted">选择一个风格对象，三家模型会同时生成三版文章。你最终改好的成稿可以提交到记忆学习，沉淀成“我的风格”。</p>

    <div class="layout">
      <section class="panel">
        <h2>生成设置</h2>

        <label for="styleName">风格对象</label>
        <select id="styleName"><option value="yijiangren">一将人 / yijiangren</option></select>

        <label for="topic">主题</label>
        <input id="topic" placeholder="例如：普通人为什么越来越难靠工资变富" />

        <label for="point">主旨 / 核心观点</label>
        <textarea id="point" placeholder="例如：真正的分水岭不是努力程度，而是是否拥有资产、杠杆和周期认知。"></textarea>

        <label for="platform">发布平台</label>
        <select id="platform">
          <option>X/Twitter</option>
          <option>公众号</option>
          <option>知乎</option>
          <option>小红书</option>
          <option>即刻</option>
        </select>

        <label for="length">目标长度</label>
        <select id="length">
          <option>800字左右</option>
          <option>1200字左右</option>
          <option>1500字左右</option>
          <option>一条长推</option>
          <option>短推串，5-8条</option>
        </select>

        <label for="constraints">额外要求</label>
        <textarea id="constraints">不要照抄样本原文，不要冒充原作者，保持原创；语言要有判断力和解释力。</textarea>

        <div class="actions">
          <button id="generate">三模型同时生成</button>
          <span id="status" class="muted status"></span>
        </div>

        <div class="hint">失败的模型会单独显示错误，不影响其它模型出稿。</div>
      </section>

      <section class="panel">
        <h2>三版生成结果</h2>
        <div class="result-grid">
          <div class="result-card">
            <h3>GPT <span id="gptBadge" class="badge">等待</span></h3>
            <textarea id="gptDraft" placeholder="GPT 生成结果"></textarea>
            <div class="actions"><button class="secondary copy-btn" data-target="gptDraft" type="button">复制 GPT</button></div>
          </div>
          <div class="result-card">
            <h3>Grok <span id="grokBadge" class="badge">等待</span></h3>
            <textarea id="grokDraft" placeholder="Grok 生成结果"></textarea>
            <div class="actions"><button class="secondary copy-btn" data-target="grokDraft" type="button">复制 Grok</button></div>
          </div>
          <div class="result-card">
            <h3>DeepSeek <span id="deepseekBadge" class="badge">等待</span></h3>
            <textarea id="deepseekDraft" placeholder="DeepSeek 生成结果"></textarea>
            <div class="actions"><button class="secondary copy-btn" data-target="deepseekDraft" type="button">复制 DeepSeek</button></div>
          </div>
        </div>

        <div class="memory">
          <h2>最终稿记忆学习</h2>
          <p class="muted">把你最终选取、融合、修改后的成稿粘到这里，保存后会出现在上方“风格对象”里。</p>
          <label for="memoryStyle">保存为哪个风格</label>
          <input id="memoryStyle" value="my_style" />
          <label for="memoryTitle">最终稿标题</label>
          <input id="memoryTitle" placeholder="可选" />
          <label for="finalArticle">最终成稿</label>
          <textarea id="finalArticle" placeholder="粘贴你最终确认的文章"></textarea>
          <div class="actions">
            <button id="learn" type="button">保存到记忆学习</button>
            <span id="learnStatus" class="muted status"></span>
          </div>
        </div>
      </section>
    </div>
  </main>

  <script>
    const styleName = document.querySelector("#styleName");
    const topic = document.querySelector("#topic");
    const point = document.querySelector("#point");
    const platform = document.querySelector("#platform");
    const length = document.querySelector("#length");
    const constraints = document.querySelector("#constraints");
    const generate = document.querySelector("#generate");
    const statusEl = document.querySelector("#status");
    const learn = document.querySelector("#learn");
    const learnStatus = document.querySelector("#learnStatus");
    const memoryStyle = document.querySelector("#memoryStyle");
    const memoryTitle = document.querySelector("#memoryTitle");
    const finalArticle = document.querySelector("#finalArticle");
    const outputs = {
      gpt: { draft: document.querySelector("#gptDraft"), badge: document.querySelector("#gptBadge") },
      grok: { draft: document.querySelector("#grokDraft"), badge: document.querySelector("#grokBadge") },
      deepseek: { draft: document.querySelector("#deepseekDraft"), badge: document.querySelector("#deepseekBadge") }
    };

    function persist() {
      localStorage.setItem("write.styleName", styleName.value);
      localStorage.setItem("write.platform", platform.value);
      localStorage.setItem("write.length", length.value);
      localStorage.setItem("write.constraints", constraints.value);
      localStorage.setItem("write.memoryStyle", memoryStyle.value);
    }

    function load() {
      platform.value = localStorage.getItem("write.platform") || platform.value;
      length.value = localStorage.getItem("write.length") || length.value;
      constraints.value = localStorage.getItem("write.constraints") || constraints.value;
      memoryStyle.value = localStorage.getItem("write.memoryStyle") || memoryStyle.value;
    }

    function setBadge(name, text, cls) {
      outputs[name].badge.className = `badge ${cls || ""}`;
      outputs[name].badge.textContent = text;
    }

    async function loadStyles() {
      const current = localStorage.getItem("write.styleName") || styleName.value;
      const response = await fetch("/styles/options");
      const data = await response.json();
      styleName.innerHTML = "";
      for (const item of data.styles || []) {
        const option = document.createElement("option");
        const absorbedCount = Number(item.absorbed_count ?? item.sample_count ?? 0);
        option.value = item.style_name;
        option.textContent = `${item.display_name || item.style_name} (${item.style_name}) · 已吸收 ${absorbedCount} 篇`;
        styleName.appendChild(option);
      }
      if ([...styleName.options].some((option) => option.value === current)) {
        styleName.value = current;
      }
    }

    async function runGenerate() {
      persist();
      generate.disabled = true;
      statusEl.textContent = "正在生成三版文章，可能需要几十秒到几分钟...";
      for (const name of Object.keys(outputs)) {
        outputs[name].draft.value = "";
        setBadge(name, "生成中", "");
      }
      const brief = `主题：${topic.value.trim()}\\n主旨：${point.value.trim()}`;
      try {
        const response = await fetch("/generate-three", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            style_name: styleName.value,
            brief,
            platform: platform.value,
            target_length: length.value,
            extra_constraints: constraints.value
          })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "生成失败");
        for (const name of Object.keys(outputs)) {
          const result = data.results?.[name];
          if (result?.ok) {
            outputs[name].draft.value = result.draft || "";
            setBadge(name, "成功", "ok");
          } else {
            outputs[name].draft.value = result?.error || "未返回结果";
            setBadge(name, "失败", "err");
          }
        }
        statusEl.textContent = "生成完成";
      } catch (error) {
        statusEl.textContent = String(error.message || error);
        for (const name of Object.keys(outputs)) setBadge(name, "失败", "err");
      } finally {
        generate.disabled = false;
      }
    }

    async function learnFinal() {
      persist();
      learn.disabled = true;
      learnStatus.textContent = "正在保存记忆...";
      try {
        const response = await fetch("/memory/learn-final", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            style_name: memoryStyle.value,
            display_name: "我的风格",
            title: memoryTitle.value,
            text: finalArticle.value,
            source_style: styleName.value
          })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "保存失败");
        learnStatus.textContent = `已学习，共 ${data.total_samples} 篇最终稿`;
        finalArticle.value = "";
        memoryTitle.value = "";
        await loadStyles();
        styleName.value = data.style_name;
        persist();
      } catch (error) {
        learnStatus.textContent = String(error.message || error);
      } finally {
        learn.disabled = false;
      }
    }

    generate.addEventListener("click", runGenerate);
    learn.addEventListener("click", learnFinal);
    document.querySelectorAll(".copy-btn").forEach((button) => {
      button.addEventListener("click", async () => {
        await navigator.clipboard.writeText(document.querySelector(`#${button.dataset.target}`).value);
        statusEl.textContent = "已复制";
      });
    });
    for (const el of [styleName, platform, length, constraints, memoryStyle]) {
      el.addEventListener("change", persist);
    }
    load();
    loadStyles();
  </script>
</body>
</html>
"""


@app.get("/write-legacy", response_class=HTMLResponse, include_in_schema=False)
def write_page() -> str:
    return """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>文章生成工作台</title>
  <style>
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2937; background: #f6f7fb; }
    main { max-width: 1160px; margin: 0 auto; padding: 32px 20px 64px; }
    h1 { margin: 0 0 8px; font-size: 30px; }
    h2 { margin: 0 0 12px; font-size: 20px; }
    p, label { line-height: 1.7; font-size: 16px; }
    .layout { display: grid; grid-template-columns: 420px minmax(0, 1fr); gap: 18px; align-items: start; margin-top: 18px; }
    .panel { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 18px; }
    label { display: block; font-weight: 700; margin: 12px 0 6px; }
    input, select, textarea { width: 100%; box-sizing: border-box; border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px 12px; font: inherit; background: #fff; }
    textarea { min-height: 110px; resize: vertical; }
    #draft { min-height: 620px; white-space: pre-wrap; }
    button { border: 0; border-radius: 6px; padding: 11px 16px; font: inherit; font-weight: 800; background: #0f766e; color: #fff; cursor: pointer; }
    button.secondary { background: #334155; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    .actions { display: flex; align-items: center; gap: 10px; margin-top: 14px; flex-wrap: wrap; }
    .muted { color: #6b7280; }
    .status { min-height: 24px; }
    .hint { background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 6px; padding: 10px; margin-top: 12px; color: #475569; font-size: 14px; }
    @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } #draft { min-height: 420px; } }
  </style>
</head>
<body>
  <main>
    <h1>文章生成工作台</h1>
    <p class="muted">选择一个风格画像，填主题和主旨，生成原创文章。</p>

    <div class="layout">
      <section class="panel">
        <h2>生成设置</h2>

        <label for="profile">使用哪个模型分析出的风格</label>
        <select id="profile">
          <option value="yijiangren__gpt">GPT 风格画像</option>
          <option value="yijiangren__grok">Grok 风格画像</option>
          <option value="yijiangren__deepseek">DeepSeek 风格画像</option>
          <option value="yijiangren">本地粗略画像</option>
        </select>

        <label for="topic">主题</label>
        <input id="topic" placeholder="例如：普通人为什么越来越难靠工资变富" />

        <label for="point">主旨 / 核心观点</label>
        <textarea id="point" placeholder="例如：真正的分水岭不是努力程度，而是是否拥有资产、杠杆和周期认知。"></textarea>

        <label for="platform">发布平台</label>
        <select id="platform">
          <option>X/Twitter</option>
          <option>公众号</option>
          <option>知乎</option>
          <option>小红书</option>
          <option>即刻</option>
        </select>

        <label for="length">目标长度</label>
        <select id="length">
          <option>800字左右</option>
          <option>1200字左右</option>
          <option>1500字左右</option>
          <option>一条长推</option>
          <option>短推串，5-8条</option>
        </select>

        <label for="constraints">额外要求</label>
        <textarea id="constraints">不要照抄样本原文，不要冒充原作者，保持原创；语言要有判断力和解释力。</textarea>

        <div class="actions">
          <button id="generate">生成文章</button>
          <button class="secondary" id="copy" type="button">复制结果</button>
          <span id="status" class="muted status"></span>
        </div>

        <div class="hint">如果某个模型生成失败，切换另一个风格画像再试。三份画像已经保存好了。</div>
      </section>

      <section class="panel">
        <h2>生成结果</h2>
        <textarea id="draft" placeholder="生成后的文章会出现在这里。"></textarea>
      </section>
    </div>
  </main>

  <script>
    const profile = document.querySelector("#profile");
    const topic = document.querySelector("#topic");
    const point = document.querySelector("#point");
    const platform = document.querySelector("#platform");
    const length = document.querySelector("#length");
    const constraints = document.querySelector("#constraints");
    const generate = document.querySelector("#generate");
    const copy = document.querySelector("#copy");
    const statusEl = document.querySelector("#status");
    const draft = document.querySelector("#draft");

    function persist() {
      localStorage.setItem("write.profile", profile.value);
      localStorage.setItem("write.platform", platform.value);
      localStorage.setItem("write.length", length.value);
      localStorage.setItem("write.constraints", constraints.value);
    }

    function load() {
      profile.value = localStorage.getItem("write.profile") || profile.value;
      platform.value = localStorage.getItem("write.platform") || platform.value;
      length.value = localStorage.getItem("write.length") || length.value;
      constraints.value = localStorage.getItem("write.constraints") || constraints.value;
    }

    async function runGenerate() {
      persist();
      generate.disabled = true;
      statusEl.textContent = "正在生成，可能需要几十秒...";
      draft.value = "";
      const brief = `主题：${topic.value.trim()}\n主旨：${point.value.trim()}`;
      try {
        const response = await fetch("/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            username: profile.value,
            brief,
            platform: platform.value,
            target_length: length.value,
            extra_constraints: constraints.value
          })
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "生成失败");
        }
        draft.value = data.draft || "";
        statusEl.textContent = "生成完成";
      } catch (error) {
        statusEl.textContent = String(error.message || error);
      } finally {
        generate.disabled = false;
      }
    }

    generate.addEventListener("click", runGenerate);
    copy.addEventListener("click", async () => {
      await navigator.clipboard.writeText(draft.value);
      statusEl.textContent = "已复制";
    });
    for (const el of [profile, platform, length, constraints]) {
      el.addEventListener("change", persist);
    }
    load();
  </script>
</body>
</html>
"""


@app.get("/import", response_class=HTMLResponse, include_in_schema=False)
def import_page() -> str:
    return """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>批量导入样本</title>
  <style>
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2937; background: #f6f7fb; }
    main { max-width: 980px; margin: 0 auto; padding: 36px 20px 64px; }
    h1 { margin: 0 0 8px; font-size: 30px; }
    p, label, li { line-height: 1.7; font-size: 16px; }
    a { color: #0f766e; font-weight: 700; }
    .panel { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 20px; margin-top: 18px; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    label { display: block; font-weight: 700; margin-bottom: 6px; }
    input, select, textarea { width: 100%; box-sizing: border-box; border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px 12px; font: inherit; background: #fff; }
    textarea { min-height: 320px; resize: vertical; }
    button { border: 0; border-radius: 6px; padding: 11px 16px; font: inherit; font-weight: 800; background: #0f766e; color: white; cursor: pointer; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    code, pre { background: #111827; color: #f9fafb; border-radius: 6px; }
    code { padding: 2px 6px; }
    pre { padding: 14px; white-space: pre-wrap; overflow: auto; }
    .muted { color: #6b7280; }
    .actions { display: flex; align-items: center; gap: 12px; margin-top: 16px; flex-wrap: wrap; }
    .hidden { display: none; }
    @media (max-width: 720px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <h1>批量导入样本</h1>
    <p class="muted">不用 X API。把收集好的文本一次性粘贴进来，或者选择一个 .txt/.csv/.json 文件，程序会自动拆成样本。</p>

    <section class="panel">
      <div class="grid">
        <div>
          <label for="username">样本名称</label>
          <input id="username" value="sample_writer" />
        </div>
        <div>
          <label for="displayName">显示名称</label>
          <input id="displayName" value="手动样本" />
        </div>
      </div>

      <div class="grid" style="margin-top:16px;">
        <div>
          <label for="splitMode">拆分方式</label>
          <select id="splitMode">
            <option value="paragraph">按空行拆分</option>
            <option value="line">按每一行拆分</option>
            <option value="separator">按自定义分隔符拆分</option>
          </select>
        </div>
        <div id="separatorWrap" class="hidden">
          <label for="separator">自定义分隔符</label>
          <input id="separator" placeholder="例如：---" />
        </div>
      </div>

      <p>
        <label for="file">可选：读取本地文本文件</label>
        <input id="file" type="file" accept=".txt,.csv,.json,.md,text/plain,text/csv,application/json" />
      </p>

      <p>
        <label for="text">样本文本</label>
        <textarea id="text" placeholder="把很多条推文、短文、文章段落粘贴到这里。默认按空行拆分。"></textarea>
      </p>

      <div class="actions">
        <button id="submit">导入样本</button>
        <span id="status" class="muted"></span>
      </div>
      <pre id="result" class="hidden"></pre>
    </section>

    <section class="panel">
      <p><strong>导入成功后下一步：</strong></p>
      <ol>
        <li>打开 <a href="/docs">/docs</a></li>
        <li>运行 <code>/style/analyze</code>，username 填上面的样本名称</li>
        <li>运行 <code>/generate</code>，username 继续填同一个样本名称</li>
      </ol>
    </section>
  </main>

  <script>
    const splitMode = document.querySelector("#splitMode");
    const separatorWrap = document.querySelector("#separatorWrap");
    const fileInput = document.querySelector("#file");
    const submit = document.querySelector("#submit");
    const statusEl = document.querySelector("#status");
    const result = document.querySelector("#result");

    splitMode.addEventListener("change", () => {
      separatorWrap.classList.toggle("hidden", splitMode.value !== "separator");
    });

    fileInput.addEventListener("change", async () => {
      const file = fileInput.files[0];
      if (!file) return;
      document.querySelector("#text").value = await file.text();
      statusEl.textContent = "已读取文件：" + file.name;
    });

    submit.addEventListener("click", async () => {
      submit.disabled = true;
      statusEl.textContent = "正在导入...";
      result.classList.add("hidden");
      try {
        const payload = {
          username: document.querySelector("#username").value,
          display_name: document.querySelector("#displayName").value,
          text: document.querySelector("#text").value,
          split_mode: splitMode.value,
          separator: document.querySelector("#separator").value
        };
        const response = await fetch("/samples/import-bulk", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        result.textContent = JSON.stringify(data, null, 2);
        result.classList.remove("hidden");
        statusEl.textContent = response.ok ? "导入成功" : "导入失败";
      } catch (error) {
        statusEl.textContent = "导入失败";
        result.textContent = String(error);
        result.classList.remove("hidden");
      } finally {
        submit.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


@app.get("/research", response_class=HTMLResponse, include_in_schema=False)
def research_page() -> str:
    return """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>写作助手</title>
  <style>
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172033; background: #f7f8fb; }
    main { max-width: 1600px; margin: 0 auto; padding: 28px 18px 56px; }
    h1 { margin: 0 0 8px; font-size: 28px; }
    h2 { margin: 0 0 14px; font-size: 18px; }
    label { display: block; margin: 12px 0 6px; font-weight: 750; }
    input, textarea, select { width: 100%; box-sizing: border-box; border: 1px solid #cbd5e1; border-radius: 7px; padding: 10px 12px; font: inherit; background: #fff; }
    textarea { resize: vertical; min-height: 88px; }
    button { border: 0; border-radius: 7px; padding: 11px 15px; font: inherit; font-weight: 800; cursor: pointer; background: #0f766e; color: #fff; }
    button.secondary { background: #334155; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    .grid { display: grid; grid-template-columns: 420px minmax(0, 1fr); gap: 16px; align-items: start; margin-top: 18px; }
    .panel { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; }
    .row { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .row.three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .actions { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; align-items: center; }
    .muted { color: #64748b; line-height: 1.6; }
    .status { color: #475569; min-height: 22px; }
    .brief { min-height: 360px; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 13px; line-height: 1.55; }
    .cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 12px; align-items: stretch; }
    .card { border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; background: #fff; min-width: 0; }
    .card h3 { margin: 0 0 8px; font-size: 16px; }
    .draft { white-space: pre-wrap; line-height: 1.65; font-size: 14px; max-height: 520px; overflow: auto; }
    .final-editor { min-height: 520px; line-height: 1.65; font-size: 14px; }
    .badge { display: inline-block; border-radius: 999px; padding: 3px 8px; font-size: 12px; font-weight: 800; background: #e2e8f0; color: #334155; }
    .badge.ok { background: #dcfce7; color: #166534; }
    .badge.err { background: #fee2e2; color: #991b1b; }
    @media (max-width: 980px) { .grid, .cards, .row { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <h1>写作助手</h1>
    <p class="muted">输入主题后先生成证据简报，再基于证据同时生成 GPT、Grok、DeepSeek 三版文章。</p>
    <div class="grid">
      <section class="panel">
        <h2>研究参数</h2>
        <label for="styleName">风格对象</label>
        <select id="styleName"></select>

        <label for="topic">主题</label>
        <textarea id="topic">Meta 出售 AI 算力，以及这两天全球科技股、存储股大幅回调，说明 AI 算力周期发生了什么变化？</textarea>

        <label for="keywords">关键词</label>
        <input id="keywords" value="Meta AI compute, AI cloud, 科技股回调, 存储股回调" />

        <label for="symbols">美股代码</label>
        <input id="symbols" value="META,NVDA,MU,AVGO,AMD,QQQ,SMH" />

        <label for="jin10Keywords">金十关键词</label>
        <input id="jin10Keywords" value="OpenAI,美光,美联储,科技股" />

        <label for="jin10Codes">金十行情代码</label>
        <input id="jin10Codes" value="XAUUSD,USOIL,USDCNH" />

        <div class="row three">
          <div>
            <label for="platform">平台</label>
            <select id="platform">
              <option value="X / Twitter">X / Twitter</option>
              <option value="小红书">小红书</option>
              <option value="公众号">公众号</option>
              <option value="知乎">知乎</option>
              <option value="即刻">即刻</option>
            </select>
          </div>
          <div>
            <label for="contentMode">篇幅倾向</label>
            <select id="contentMode">
              <option value="标准：结构完整，观点、证据和推论都要有，但不额外铺垫。">标准</option>
              <option value="精简：只保留最关键的一条主线，结论前置，少铺垫，信息密度更高。">精简</option>
              <option value="展开：允许更多背景、解释和转折，但仍然服从目标长度。">展开</option>
            </select>
          </div>
          <div>
            <label for="targetLength">长度</label>
            <input id="targetLength" value="1200-1800字" />
          </div>
        </div>
        <p class="muted">平台决定发布语境；篇幅倾向只决定相对节奏；长度是最终字数约束。</p>

        <label for="constraints">额外要求</label>
        <textarea id="constraints">先讲结论，再讲证据链；区分事实和推论；不要编造没有来源的数据。</textarea>

        <div class="actions">
          <button id="collectBtn">自动研究</button>
          <button id="writeBtn" class="secondary">三模型生成</button>
          <span id="status" class="status"></span>
        </div>
      </section>

      <section class="panel">
        <h2>证据简报</h2>
        <textarea id="brief" class="brief"></textarea>
        <div style="margin-top:16px;">
          <h2>参考文本改写</h2>
          <p class="muted">按风格改写只看参考文本；结合研究生成会以参考文本为主线，并用证据简报补强。</p>
          <label for="referenceText">参考文本</label>
          <textarea id="referenceText" style="min-height:180px;"></textarea>
          <div class="actions">
            <button id="rewriteBtn" class="secondary">按风格改写</button>
            <button id="refWriteBtn" class="secondary">结合研究生成</button>
            <span id="rewriteStatus" class="status"></span>
          </div>
        </div>
      </section>
    </div>

    <section class="panel" style="margin-top:16px;">
      <h2>生成结果</h2>
      <div id="results" class="cards">
        <article class="card">
          <h3>最终稿 <span class="badge">编辑</span></h3>
          <p class="muted">你也可以不等模型生成，直接把终稿贴在这里调整格式。</p>
          <textarea id="inlineFinalText" class="final-editor"></textarea>
          <div class="actions">
            <button id="copyFinalBtn" class="secondary" type="button">复制最终稿</button>
            <button id="formatFinalBtn" class="secondary" type="button">Grok 调整格式</button>
            <button id="saveInlineFinalBtn" type="button">保存到记忆</button>
            <button id="clearFinalBtn" class="secondary" type="button">一键清除</button>
          </div>
        </article>
      </div>
    </section>

    <section class="panel" style="margin-top:16px;">
      <h2>最终稿记忆学习</h2>
      <p class="muted">把你最终确定的文章贴在这里保存，后续生成会把它沉淀成可选的个人风格。</p>
      <div class="row">
        <div>
          <label for="memoryStyleName">保存为风格名</label>
          <input id="memoryStyleName" value="my_style" />
        </div>
        <div>
          <label for="memoryDisplayName">显示名称</label>
          <input id="memoryDisplayName" value="我的风格" />
        </div>
      </div>
      <label for="memoryTitle">标题</label>
      <input id="memoryTitle" />
      <label for="finalText">最终成稿</label>
      <textarea id="finalText" style="min-height:220px;"></textarea>
      <div class="actions">
        <button id="learnBtn">保存到记忆</button>
        <span id="learnStatus" class="status"></span>
      </div>
    </section>
  </main>

  <script>
    const $ = (id) => document.querySelector(id);
    const splitList = (value) => value.split(/[,，\\n]/).map((item) => item.trim()).filter(Boolean);
    const statusEl = $("#status");
    let latestResults = {};

    async function loadStyles() {
      const res = await fetch("/styles/options");
      const data = await res.json();
      const select = $("#styleName");
      select.innerHTML = "";
      for (const item of data.styles || []) {
        const option = document.createElement("option");
        const absorbedCount = Number(item.absorbed_count ?? item.sample_count ?? 0);
        option.value = item.style_name;
        option.textContent = `${item.display_name || item.style_name} (${item.style_name}) · 已吸收 ${absorbedCount} 篇`;
        select.appendChild(option);
      }
    }

    function publicationSettings() {
      const platform = $("#platform").value;
      const contentMode = $("#contentMode").value;
      const targetLength = $("#targetLength").value.trim();
      return {
        platform: `${platform}；篇幅倾向：${contentMode}`,
        targetLength
      };
    }

    async function collectResearch() {
      statusEl.textContent = "研究中...";
      $("#collectBtn").disabled = true;
      try {
        const payload = {
          topic: $("#topic").value,
          keywords: splitList($("#keywords").value),
          symbols: splitList($("#symbols").value),
          jin10_keywords: splitList($("#jin10Keywords").value),
          jin10_quote_codes: splitList($("#jin10Codes").value)
        };
        const res = await fetch("/research/collect", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "研究失败");
        $("#brief").value = data.evidence_brief || "";
        statusEl.textContent = `已整理 ${data.evidence_count || 0} 条证据`;
      } catch (error) {
        statusEl.textContent = String(error.message || error);
      } finally {
        $("#collectBtn").disabled = false;
      }
    }

    function resultCard(name, item) {
      const ok = item && item.ok;
      const draft = ok ? item.draft : (item && item.error ? item.error : "无结果");
      return `<article class="card">
        <h3>${name.toUpperCase()} <span class="badge ${ok ? "ok" : "err"}">${ok ? "完成" : "失败"}</span></h3>
        <p class="muted">${item && item.model ? item.model : ""}</p>
        <div class="actions"><button class="secondary copy-result" data-model="${name}" type="button">复制</button></div>
        <div class="draft">${escapeHtml(draft)}</div>
      </article>`;
    }

    function finalEditorCard() {
      return `<article class="card">
        <h3>最终稿 <span class="badge">编辑</span></h3>
        <p class="muted">对照左边三版，在这里整理你的最终文章。</p>
        <textarea id="inlineFinalText" class="final-editor"></textarea>
        <div class="actions">
          <button id="copyFinalBtn" class="secondary" type="button">复制最终稿</button>
          <button id="formatFinalBtn" class="secondary" type="button">Grok 调整格式</button>
          <button id="saveInlineFinalBtn" type="button">保存到记忆</button>
          <button id="clearFinalBtn" class="secondary" type="button">一键清除</button>
        </div>
      </article>`;
    }

    function renderResults(results) {
      latestResults = results || {};
      $("#results").innerHTML = ["gpt", "grok", "deepseek"].map((name) => resultCard(name, latestResults[name])).join("") + finalEditorCard();
    }

    async function writeThree() {
      const evidenceBrief = $("#brief").value.trim();
      if (!evidenceBrief) {
        statusEl.textContent = "请先生成证据简报";
        return;
      }
      statusEl.textContent = "生成中...";
      $("#writeBtn").disabled = true;
      $("#results").innerHTML = "";
      try {
        const publication = publicationSettings();
        const payload = {
          style_name: $("#styleName").value,
          topic: $("#topic").value,
          evidence_brief: evidenceBrief,
          platform: publication.platform,
          target_length: publication.targetLength,
          extra_constraints: $("#constraints").value
        };
        const res = await fetch("/research/write-three", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "生成失败");
        const results = data.results || {};
        renderResults(results);
        statusEl.textContent = "生成完成";
      } catch (error) {
        statusEl.textContent = String(error.message || error);
      } finally {
        $("#writeBtn").disabled = false;
      }
    }

    async function rewriteThree() {
      const referenceText = $("#referenceText").value.trim();
      const rewriteStatus = $("#rewriteStatus");
      if (!referenceText) {
        rewriteStatus.textContent = "请先粘贴参考文本";
        return;
      }
      rewriteStatus.textContent = "改写中...";
      $("#rewriteBtn").disabled = true;
      $("#results").innerHTML = "";
      try {
        const publication = publicationSettings();
        const payload = {
          style_name: $("#styleName").value,
          reference_text: referenceText,
          platform: publication.platform,
          target_length: publication.targetLength
        };
        const res = await fetch("/research/rewrite-three", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "改写失败");
        const results = data.results || {};
        renderResults(results);
        rewriteStatus.textContent = "改写完成";
      } catch (error) {
        rewriteStatus.textContent = String(error.message || error);
      } finally {
        $("#rewriteBtn").disabled = false;
      }
    }

    async function referenceWriteThree() {
      const referenceText = $("#referenceText").value.trim();
      const evidenceBrief = $("#brief").value.trim();
      const rewriteStatus = $("#rewriteStatus");
      if (!referenceText) {
        rewriteStatus.textContent = "请先粘贴参考文本";
        return;
      }
      if (!evidenceBrief) {
        rewriteStatus.textContent = "请先自动研究，生成证据简报";
        return;
      }
      rewriteStatus.textContent = "结合研究和参考文本生成中...";
      $("#refWriteBtn").disabled = true;
      $("#results").innerHTML = "";
      try {
        const publication = publicationSettings();
        const payload = {
          style_name: $("#styleName").value,
          topic: $("#topic").value,
          evidence_brief: evidenceBrief,
          reference_text: referenceText,
          platform: publication.platform,
          target_length: publication.targetLength,
          extra_constraints: $("#constraints").value
        };
        const res = await fetch("/research/reference-write-three", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "生成失败");
        renderResults(data.results || {});
        rewriteStatus.textContent = "三模型生成完成";
      } catch (error) {
        rewriteStatus.textContent = String(error.message || error);
      } finally {
        $("#refWriteBtn").disabled = false;
      }
    }

    async function learnFinal() {
      const learnStatus = $("#learnStatus");
      const inlineFinal = $("#inlineFinalText");
      const text = (inlineFinal && inlineFinal.value.trim()) || $("#finalText").value.trim();
      if (!text) {
        learnStatus.textContent = "请先粘贴最终成稿";
        return;
      }
      learnStatus.textContent = "保存中...";
      $("#learnBtn").disabled = true;
      try {
        const payload = {
          style_name: $("#memoryStyleName").value || "my_style",
          display_name: $("#memoryDisplayName").value || "我的风格",
          title: $("#memoryTitle").value,
          text,
          source_style: $("#styleName").value
        };
        const res = await fetch("/memory/learn-final", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "保存失败");
        learnStatus.textContent = `已保存，当前累计 ${data.total_samples || 0} 篇`;
        await loadStyles();
      } catch (error) {
        learnStatus.textContent = String(error.message || error);
      } finally {
        $("#learnBtn").disabled = false;
      }
    }

    async function copyFinalDraft() {
      const inlineFinal = $("#inlineFinalText");
      const text = inlineFinal ? inlineFinal.value.trim() : "";
      if (!text) {
        statusEl.textContent = "最终稿还是空的";
        return;
      }
      await navigator.clipboard.writeText(text);
      statusEl.textContent = "已复制最终稿";
    }

    function clearFinalDraft() {
      const inlineFinal = $("#inlineFinalText");
      if (inlineFinal) inlineFinal.value = "";
      const fallbackFinal = $("#finalText");
      if (fallbackFinal) fallbackFinal.value = "";
      statusEl.textContent = "已清空最终稿";
    }

    async function formatFinalDraft() {
      const inlineFinal = $("#inlineFinalText");
      const text = inlineFinal ? inlineFinal.value.trim() : "";
      if (!text) {
        statusEl.textContent = "请先把最终稿粘贴到最右栏";
        return;
      }
      const button = $("#formatFinalBtn");
      statusEl.textContent = "Grok 正在调整推文格式...";
      button.disabled = true;
      try {
        const publication = publicationSettings();
        const payload = {
          style_name: $("#memoryStyleName").value || $("#styleName").value || "my_style",
          text,
          platform: publication.platform,
          target_length: publication.targetLength
        };
        const res = await fetch("/research/format-final", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "格式调整失败");
        inlineFinal.value = data.formatted || "";
        statusEl.textContent = "Grok 已调整格式，你可以继续手动微调";
      } catch (error) {
        statusEl.textContent = String(error.message || error);
      } finally {
        button.disabled = false;
      }
    }

    async function copyResult(modelName) {
      const item = latestResults[modelName];
      const text = item && item.draft ? item.draft : "";
      if (!text) {
        statusEl.textContent = "没有可复制的内容";
        return;
      }
      await navigator.clipboard.writeText(text);
      statusEl.textContent = `已复制 ${modelName.toUpperCase()}`;
    }

    function escapeHtml(text) {
      return String(text).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[ch]));
    }

    $("#collectBtn").addEventListener("click", collectResearch);
    $("#writeBtn").addEventListener("click", writeThree);
    $("#rewriteBtn").addEventListener("click", rewriteThree);
    $("#refWriteBtn").addEventListener("click", referenceWriteThree);
    $("#learnBtn").addEventListener("click", learnFinal);
    $("#results").addEventListener("click", (event) => {
      const button = event.target.closest(".copy-result");
      if (button) copyResult(button.dataset.model);
      if (event.target.closest("#copyFinalBtn")) copyFinalDraft();
      if (event.target.closest("#formatFinalBtn")) formatFinalDraft();
      if (event.target.closest("#saveInlineFinalBtn")) learnFinal();
      if (event.target.closest("#clearFinalBtn")) clearFinalDraft();
    });
    loadStyles().catch((error) => { statusEl.textContent = String(error); });
  </script>
</body>
</html>
"""


@app.get("/health", summary="第 1 步：检查配置是否正常")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "database_path": str(settings.database_path),
        "has_x_token": bool(settings.x_bearer_token),
        "has_openai_key": bool(settings.openai_api_key),
        "has_xai_key": bool(settings.xai_api_key),
        "has_deepseek_key": bool(settings.deepseek_api_key),
        "has_tavily_key": bool(settings.tavily_api_key),
        "has_alpha_vantage_key": bool(settings.alpha_vantage_api_key),
        "has_finnhub_key": bool(settings.finnhub_api_key),
        "has_blockbeats_key": bool(settings.blockbeats_api_key),
        "has_jin10_mcp_token": bool(settings.jin10_mcp_token),
        "models": {
            "gpt": settings.openai_model,
            "grok": settings.xai_model,
            "deepseek": settings.deepseek_model,
        },
    }


def make_jin10_client() -> Jin10McpClient:
    if not settings.jin10_mcp_token:
        raise HTTPException(status_code=400, detail="Missing JIN10_MCP_TOKEN in .env.")
    try:
        client = Jin10McpClient(
            server_url=settings.jin10_mcp_url,
            bearer_token=settings.jin10_mcp_token,
        )
        client.start()
        return client
    except Jin10McpError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/jin10/tools", summary="Jin10 MCP: list available tools")
def jin10_tools() -> dict[str, Any]:
    client = make_jin10_client()
    return {"tools": client.list_tools()}


@app.get("/jin10/resources", summary="Jin10 MCP: list available resources")
def jin10_resources() -> dict[str, Any]:
    client = make_jin10_client()
    return {"resources": client.list_resources()}


@app.get("/jin10/quote-codes", summary="Jin10 MCP: read quote://codes")
def jin10_quote_codes() -> dict[str, Any]:
    client = make_jin10_client()
    return {"data": client.read_resource("quote://codes")}


@app.get("/jin10/quote/{code}", summary="Jin10 MCP: get quote by code")
def jin10_quote(code: str) -> dict[str, Any]:
    client = make_jin10_client()
    return {"data": client.get_quote(code).data}


@app.get("/jin10/kline/{code}", summary="Jin10 MCP: get kline by code")
def jin10_kline(
    code: str,
    time: int | None = Query(None, description="Unix timestamp in seconds."),
    count: int | None = Query(None, ge=1, le=100),
) -> dict[str, Any]:
    client = make_jin10_client()
    return {"data": client.get_kline(code=code, time=time, count=count).data}


@app.get("/jin10/flash", summary="Jin10 MCP: list or search flash news")
def jin10_flash(
    keyword: str | None = Query(None),
    cursor: str | None = Query(None),
) -> dict[str, Any]:
    client = make_jin10_client()
    result = client.search_flash(keyword) if keyword else client.list_flash(cursor)
    return {"data": result.data}


@app.get("/jin10/news", summary="Jin10 MCP: list or search news")
def jin10_news(
    keyword: str | None = Query(None),
    cursor: str | None = Query(None),
) -> dict[str, Any]:
    client = make_jin10_client()
    result = client.search_news(keyword, cursor) if keyword else client.list_news(cursor)
    return {"data": result.data}


@app.get("/jin10/news/{news_id}", summary="Jin10 MCP: get news detail")
def jin10_news_detail(news_id: str) -> dict[str, Any]:
    client = make_jin10_client()
    return {"data": client.get_news(news_id).data}


@app.get("/jin10/calendar", summary="Jin10 MCP: list calendar")
def jin10_calendar() -> dict[str, Any]:
    client = make_jin10_client()
    return {"data": client.list_calendar().data}


@app.post("/research/collect", summary="Collect evidence from web, market data, BlockBeats, and Jin10")
def research_collect(request: ResearchCollectRequest) -> dict[str, Any]:
    if not request.topic.strip():
        raise HTTPException(status_code=400, detail="topic cannot be empty.")
    try:
        research = collect_research(
            ResearchSettings(
                tavily_api_key=settings.tavily_api_key,
                alpha_vantage_api_key=settings.alpha_vantage_api_key,
                finnhub_api_key=settings.finnhub_api_key,
                blockbeats_api_key=settings.blockbeats_api_key,
            ),
            topic=request.topic,
            keywords=request.keywords,
            symbols=request.symbols,
        )
    except ResearchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    _add_jin10_research(research, request)
    evidence_brief = format_evidence_brief(research)
    return {
        **research,
        "evidence_count": len(research.get("evidence", [])),
        "evidence_brief": evidence_brief,
    }


@app.post("/research/write-three", summary="Write three model drafts from an evidence brief")
def research_write_three(request: ResearchWriteThreeRequest) -> dict[str, Any]:
    topic = request.topic.strip()
    evidence_brief = request.evidence_brief.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="topic cannot be empty.")
    if not evidence_brief:
        raise HTTPException(status_code=400, detail="evidence_brief cannot be empty.")

    research_brief = f"""
主题：
{topic}

证据简报：
{evidence_brief}

写作要求：
- 先给出清晰结论，再展开证据链。
- 必须区分事实、数据和推论。
- 只能使用证据简报里已有的数据；没有来源的数据不要编造。
- 正文不要输出来源链接、来源括号或脚注，避免占用推文长度。
- 可以根据证据简报判断事实可信度，但文章里只写必要事实和分析。
- 如果证据不足，要明确写出“不足以确认”的部分。
- 写成一篇完整中文分析文章，不要写成资料清单。
""".strip()
    if request.extra_constraints:
        research_brief += f"\n\n额外要求：\n{request.extra_constraints.strip()}"

    results: dict[str, Any] = {}
    for job in model_jobs_for_style(request.style_name):
        profile_record = db.get_style_profile(job["profile_username"])
        if not profile_record:
            results[job["name"]] = {
                "ok": False,
                "model": job["model"],
                "profile_username": job["profile_username"],
                "error": f"Style profile not found: {job['profile_username']}",
            }
            continue
        try:
            draft = generate_article_with_llm(
                username=job["profile_username"],
                style_profile=profile_record["profile"],
                brief=research_brief,
                platform=request.platform,
                target_length=request.target_length,
                extra_constraints=request.extra_constraints,
                model=job["model"],
                provider=job["provider"],
            )
            generation_id = db.save_generation(
                username=job["profile_username"],
                brief=research_brief,
                platform=request.platform,
                target_length=request.target_length,
                draft=draft,
                model=f"{job['provider']}:{job['model']}",
            )
            results[job["name"]] = {
                "ok": True,
                "id": generation_id,
                "model": job["model"],
                "profile_username": job["profile_username"],
                "draft": draft,
            }
        except LlmError as exc:
            results[job["name"]] = {
                "ok": False,
                "model": job["model"],
                "profile_username": job["profile_username"],
                "error": str(exc),
            }

    return {
        "style_name": request.style_name.strip().lstrip("@").lower(),
        "topic": topic,
        "results": results,
    }


@app.post("/research/rewrite-three", summary="Rewrite a reference text with three model drafts")
def research_rewrite_three(request: ResearchRewriteThreeRequest) -> dict[str, Any]:
    reference_text = request.reference_text.strip()
    if not reference_text:
        raise HTTPException(status_code=400, detail="reference_text cannot be empty.")

    rewrite_brief = f"""
参考文本：
{reference_text}

改写任务：
- 保留参考文本中的核心事实、信息顺序和关键判断。
- 按所选风格对象的结构、节奏、语气和论证方式重写。
- 不要逐句翻译，不要同义词替换式洗稿，要重组表达。
- 不要编造参考文本里没有的新事实或数据。
- 如果参考文本包含不确定表述，改写后也要保留不确定性。
- 写成完整中文文章，不要说明“我是改写”。
""".strip()

    return generate_three(
        GenerateThreeRequest(
            style_name=request.style_name,
            brief=rewrite_brief,
            platform=request.platform,
            target_length=request.target_length,
            extra_constraints=None,
        )
    )


@app.post("/research/reference-write-three", summary="Write three model drafts from research and reference text")
def research_reference_write_three(request: ResearchReferenceWriteThreeRequest) -> dict[str, Any]:
    topic = request.topic.strip()
    evidence_brief = request.evidence_brief.strip()
    reference_text = request.reference_text.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="topic cannot be empty.")
    if not evidence_brief:
        raise HTTPException(status_code=400, detail="evidence_brief cannot be empty.")
    if not reference_text:
        raise HTTPException(status_code=400, detail="reference_text cannot be empty.")

    combined_brief = f"""
主题：
{topic}

参考文本（必须吸收，不是可选素材）：
{reference_text}

研究内容/证据简报（用于校准、补强和更新参考文本）：
{evidence_brief}

写作任务：
- 这不是普通自动研究写作。必须以“参考文本”为文章主线，同时用“研究内容/证据简报”补充事实、数据、市场背景和最新信息。
- 输出必须能看出参考文本的痕迹：保留参考文本的核心观点、问题意识、推进顺序、关键转折和结论倾向；除非与证据冲突，不要把这些内容丢掉。
- 研究内容的作用是补强和校准参考文本：可以把参考文本里的笼统判断改成有数据支撑的判断，也可以补充最新事件，但不要绕开参考文本另写一篇。
- 至少融合参考文本中的 2-5 个可识别要素，例如核心判断、论证角度、对比关系、风险提示、开头钩子或结尾落点。不要在正文里列出这些要素清单。
- 如果参考文本和研究内容冲突，以研究内容/证据简报为准，并自然改写冲突部分。
- 不要逐句洗稿参考文本，也不要写成资料清单；要整合成一篇自然、有观点、有证据链的中文分析文章。
- 只能使用研究内容和参考文本中已有的信息，不要编造新数据、新消息或新来源。
- 正文不要输出来源链接、来源括号或脚注，避免占用推文长度。
- 如果证据不足，要明确写出“不足以确认”的部分。
""".strip()
    if request.extra_constraints:
        combined_brief += f"\n\n额外要求：\n{request.extra_constraints.strip()}"

    return generate_three(
        GenerateThreeRequest(
            style_name=request.style_name,
            brief=combined_brief,
            platform=request.platform,
            target_length=request.target_length,
            extra_constraints=request.extra_constraints,
        )
    )


@app.post("/research/format-final", summary="Use Grok to format a final draft for X/Twitter")
def research_format_final(request: FormatFinalDraftRequest) -> dict[str, Any]:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text cannot be empty.")
    if not settings.xai_api_key:
        raise HTTPException(status_code=400, detail="Missing XAI_API_KEY in .env.")

    style_name = request.style_name.strip().lstrip("@").lower() or "my_style"
    format_examples = _format_reference_examples(style_name)
    prompt = f"""
你是中文 X/Twitter 推文格式编辑。

任务：
把用户给出的最终稿整理成适合发布到 {request.platform or "X / Twitter"} 的格式。

硬性规则：
- 内容完全不要改变。不要改任何字词、标点、数字、公司名、币种、数据、观点或语序。
- 不要新增事实、数据、来源、公司名、观点、标题、编号、emoji 或解释。
- 不要删减任何句子、段落、限定词或不确定表述。
- 只允许做三类格式动作：增加/调整换行、增加/调整空行、在原文已有重点词句两侧添加 Markdown 加粗标记 **。
- 加粗要克制，只加粗核心结论、关键数据、关键资产名或转折句；不要整段加粗。
- 如果文本较长，可以通过空行切成适合 X/Twitter 阅读的段落，但不能为了长度删改内容。
- 输出正文即可，不要解释你做了什么。

目标长度：
{request.target_length or "不限制，优先保留信息完整"}。目标长度只作为分段密度参考，不允许为了长度改写、压缩或删减内容。

用户历史最终稿格式参考：
{format_examples or "暂无历史最终稿格式参考。"}

待整理文本：
{text}
""".strip()
    try:
        formatted = generate_text(
            model=settings.xai_model,
            input_text=prompt,
            provider="xai",
        )
    except LlmError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": True,
        "model": settings.xai_model,
        "style_name": style_name,
        "formatted": formatted,
    }


def _format_reference_examples(style_name: str) -> str:
    posts = db.get_posts(style_name, limit=5)
    if not posts and style_name != "my_style":
        posts = db.get_posts("my_style", limit=5)
    examples: list[str] = []
    for index, post in enumerate(posts[:5], start=1):
        text = " ".join((post.get("text") or "").split())
        if not text:
            continue
        examples.append(f"{index}. {text[:900]}")
    return "\n".join(examples)


def _add_jin10_research(research: dict[str, Any], request: ResearchCollectRequest) -> None:
    if not settings.jin10_mcp_token:
        research.setdefault("sources", {})["jin10"] = {"skipped": True, "reason": "missing token"}
        return

    source_status: dict[str, Any] = {"ok": True, "errors": []}
    research.setdefault("sources", {})["jin10"] = source_status
    evidence = research.setdefault("evidence", [])

    try:
        client = make_jin10_client()
    except HTTPException as exc:
        source_status["ok"] = False
        source_status["errors"].append(str(exc.detail))
        return

    keywords = request.jin10_keywords or request.keywords or [request.topic]
    for keyword in [item.strip() for item in keywords if item.strip()][:5]:
        try:
            flash_data = client.search_flash(keyword).data
            for item in _items_from_jin10_list(flash_data)[:5]:
                evidence.append(
                    {
                        "source": "Jin10",
                        "type": "flash",
                        "title": item.get("title") or _first_nonempty_line(item.get("content")),
                        "time": item.get("time"),
                        "url": item.get("url"),
                        "summary": item.get("content") or item.get("title"),
                        "keyword": keyword,
                    }
                )
        except Exception as exc:
            source_status["errors"].append(f"search_flash({keyword}): {exc}")

        try:
            news_data = client.search_news(keyword).data
            for item in _items_from_jin10_list(news_data)[:3]:
                evidence.append(
                    {
                        "source": "Jin10",
                        "type": "news",
                        "title": item.get("title"),
                        "time": item.get("time"),
                        "url": item.get("url"),
                        "summary": item.get("introduction") or item.get("content") or item.get("title"),
                        "keyword": keyword,
                    }
                )
        except Exception as exc:
            source_status["errors"].append(f"search_news({keyword}): {exc}")

    for code in [item.strip().upper() for item in request.jin10_quote_codes if item.strip()][:8]:
        try:
            quote = client.get_quote(code).data
            if isinstance(quote, dict):
                evidence.append(
                    {
                        "source": "Jin10",
                        "type": "quote",
                        "title": f"{quote.get('name') or code} quote",
                        "time": quote.get("time"),
                        "url": None,
                        "summary": (
                            f"{quote.get('name') or code}: close={quote.get('close')}, "
                            f"change={quote.get('ups_price')}, percent={quote.get('ups_percent')}, "
                            f"high={quote.get('high')}, low={quote.get('low')}"
                        ),
                        "data": quote,
                    }
                )
        except Exception as exc:
            source_status["errors"].append(f"get_quote({code}): {exc}")


def _items_from_jin10_list(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        items = data.get("items", [])
    elif isinstance(data, list):
        items = data
    else:
        items = []
    return [item for item in items if isinstance(item, dict)]


def _first_nonempty_line(value: Any) -> str | None:
    if not value:
        return None
    for line in str(value).splitlines():
        clean = line.strip()
        if clean:
            return clean[:100]
    return None


@app.post("/collect", summary="第 2 步：采集某个账号的公开 posts")
def collect_posts(request: CollectRequest) -> dict[str, Any]:
    try:
        client = XClient(settings.x_bearer_token or "")
        user = client.get_user(request.username)
        username = user["username"].lower()
        posts = client.fetch_recent_posts(
            user_id=user["id"],
            limit=request.limit,
            exclude_replies=request.exclude_replies,
            exclude_retweets=request.exclude_retweets,
        )
        db.upsert_author(username=username, x_user_id=user["id"], display_name=user.get("name"))
        changed = db.upsert_posts(username=username, posts=posts)
        return {
            "username": username,
            "x_user_id": user["id"],
            "fetched": len(posts),
            "rows_changed": changed,
        }
    except XApiError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/samples/import", summary="不用 X API：手动导入文章/推文样本")
def import_samples(request: ImportSamplesRequest) -> dict[str, Any]:
    return save_manual_samples(
        username=request.username,
        samples=request.samples,
        display_name=request.display_name,
    )


@app.post("/samples/import-bulk", summary="不用 X API：批量粘贴或文件导入样本")
def import_bulk_samples(request: ImportBulkSamplesRequest) -> dict[str, Any]:
    samples = split_bulk_text(request.text, request.split_mode, request.separator)
    return save_manual_samples(
        username=request.username,
        samples=samples,
        display_name=request.display_name,
    )


@app.post("/samples/import-one", summary="不用 X API：逐篇保存一个样本")
def import_one_sample(request: ImportOneSampleRequest) -> dict[str, Any]:
    return save_single_sample(
        username=request.username,
        text=request.text,
        title=request.title,
        source_url=request.source_url,
        display_name=request.display_name,
    )


@app.get("/samples/{username}/summary", summary="查看某个样本集的数量和最近样本")
def sample_summary(username: str, limit: int = Query(5, ge=1, le=20)) -> dict[str, Any]:
    clean_username = username.strip().lstrip("@").lower()
    posts = db.get_posts(clean_username, limit=limit)
    return {
        "username": clean_username,
        "count": db.count_posts(clean_username),
        "recent": [
            {
                "id": post["id"],
                "created_at": post.get("created_at"),
                "text_preview": post.get("text", "")[:180],
            }
            for post in posts
        ],
    }


def split_bulk_text(text: str, split_mode: str, separator: str | None) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    if split_mode == "line":
        parts = normalized.split("\n")
    elif split_mode == "separator":
        if not separator:
            raise HTTPException(status_code=400, detail="选择自定义分隔符时，separator 不能为空。")
        parts = normalized.split(separator)
    elif split_mode == "paragraph":
        parts = normalized.split("\n\n")
    else:
        raise HTTPException(status_code=400, detail="split_mode 只能是 paragraph、line 或 separator。")
    return [part.strip() for part in parts if part.strip()]


def save_single_sample(
    username: str,
    text: str,
    title: str | None = None,
    source_url: str | None = None,
    display_name: str | None = None,
) -> dict[str, Any]:
    username = username.strip().lstrip("@").lower()
    text = text.strip()
    if not username:
        raise HTTPException(status_code=400, detail="username 不能为空。")
    if not text:
        raise HTTPException(status_code=400, detail="请粘贴一篇有效样本。")

    db.upsert_author(
        username=username,
        x_user_id=f"manual:{username}",
        display_name=display_name or username,
    )
    digest = hashlib.sha256(
        f"{username}:{source_url or ''}:{title or ''}:{text}".encode("utf-8")
    ).hexdigest()[:16]
    changed = db.upsert_posts(
        username=username,
        posts=[
            {
                "id": f"manual:{username}:{digest}",
                "created_at": utc_now(),
                "text": text,
                "lang": "manual",
                "public_metrics": None,
                "source": "manual_single_import",
                "title": title,
                "source_url": source_url,
            }
        ],
    )
    return {
        "username": username,
        "imported": 1,
        "rows_changed": changed,
        "total_samples": db.count_posts(username),
        "next_step": "继续粘贴下一篇；积累到 10 篇左右后运行 /style/analyze。",
    }


def save_manual_samples(
    username: str,
    samples: list[str],
    display_name: str | None = None,
) -> dict[str, Any]:
    username = username.strip().lstrip("@").lower()
    samples = [sample.strip() for sample in samples if sample.strip()]
    if not username:
        raise HTTPException(status_code=400, detail="username 不能为空。")
    if not samples:
        raise HTTPException(status_code=400, detail="请至少粘贴一条有效样本。")

    db.upsert_author(
        username=username,
        x_user_id=f"manual:{username}",
        display_name=display_name or username,
    )
    posts = []
    for index, text in enumerate(samples, start=1):
        digest = hashlib.sha256(f"{username}:{index}:{text}".encode("utf-8")).hexdigest()[:16]
        posts.append(
            {
                "id": f"manual:{username}:{digest}",
                "created_at": utc_now(),
                "text": text,
                "lang": "manual",
                "public_metrics": None,
                "source": "manual_import",
            }
        )
    changed = db.upsert_posts(username=username, posts=posts)
    return {
        "username": username,
        "imported": len(posts),
        "rows_changed": changed,
        "next_step": "接下来运行第 3 步 /style/analyze，username 填这个返回值。",
    }


def model_jobs_for_style(style_name: str) -> list[dict[str, Any]]:
    base = style_name.strip().lstrip("@").lower()
    jobs = [
        {
            "name": "gpt",
            "provider": "openai",
            "model": settings.openai_model,
            "profile_username": f"{base}__gpt",
        },
        {
            "name": "grok",
            "provider": "xai",
            "model": settings.xai_model,
            "profile_username": f"{base}__grok",
        },
        {
            "name": "deepseek",
            "provider": "deepseek",
            "model": settings.deepseek_model,
            "profile_username": f"{base}__deepseek",
        },
    ]
    for job in jobs:
        if db.get_style_profile(job["profile_username"]):
            continue
        job["profile_username"] = base
    return jobs


def base_style_options() -> list[dict[str, Any]]:
    profiles = db.list_style_profiles()
    authors = {author["username"]: author for author in db.list_authors()}
    display_name_overrides = {
        "yijiangren": "一将人",
        "my_style": "我的风格",
    }
    bases: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        username = profile["username"]
        if username.endswith("__gpt") or username.endswith("__grok") or username.endswith("__deepseek"):
            base, suffix = username.rsplit("__", 1)
        else:
            base, suffix = username, "single"
        author = authors.get(base, {})
        sample_count = db.count_posts(base)
        item = bases.setdefault(
            base,
            {
                "style_name": base,
                "display_name": display_name_overrides.get(base) or author.get("display_name") or base,
                "profiles": [],
                "sample_count": sample_count,
                "absorbed_count": sample_count,
            },
        )
        item["absorbed_count"] = max(item["absorbed_count"], int(profile["sample_size"] or 0))
        item["profiles"].append(
            {
                "kind": suffix,
                "username": username,
                "model": profile["model"],
                "sample_size": profile["sample_size"],
                "updated_at": profile["updated_at"],
            }
        )
    return sorted(bases.values(), key=lambda item: item["style_name"])


@app.get("/posts/{username}", summary="查看已经采集到的 posts")
def list_posts(
    username: str,
    limit: int = Query(50, description="查看多少条，最多 500。"),
) -> dict[str, Any]:
    safe_limit = max(1, min(limit, 500))
    posts = db.get_posts(username, limit=safe_limit)
    return {"username": username.lower(), "count": len(posts), "posts": posts}


@app.post("/style/analyze", summary="第 3 步：分析账号的写作风格")
def analyze_style(request: AnalyzeStyleRequest) -> dict[str, Any]:
    posts = db.get_posts(request.username, limit=request.sample_limit)
    if not posts:
        raise HTTPException(status_code=404, detail="还没有采集到 posts。请先运行第 2 步 /collect。")

    model = request.model or settings.openai_model
    try:
        if request.use_llm:
            profile = analyze_style_with_llm(request.username, posts, model=model)
        else:
            profile = build_heuristic_profile(request.username, posts)
    except LlmError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    db.save_style_profile(
        username=request.username,
        profile=profile,
        sample_size=len(posts),
        model=model if request.use_llm else "heuristic",
    )
    return {
        "username": request.username.lower(),
        "sample_size": len(posts),
        "model": model if request.use_llm else "heuristic",
        "profile": profile,
    }


@app.post("/style/analyze-three", summary="三模型分析：GPT、Grok、DeepSeek 各分析一次")
def analyze_three_models(request: AnalyzeThreeModelsRequest) -> dict[str, Any]:
    posts = db.get_posts(request.username, limit=request.sample_limit)
    if not posts:
        raise HTTPException(status_code=404, detail="还没有样本。请先用 /capture 或 /import 导入样本。")

    jobs = [
        {
            "name": "gpt",
            "provider": "openai",
            "model": settings.openai_model,
            "api_key_present": bool(settings.openai_api_key),
            "saved_username": f"{request.username}__gpt",
        },
        {
            "name": "grok",
            "provider": "xai",
            "model": settings.xai_model,
            "api_key_present": bool(settings.xai_api_key),
            "saved_username": f"{request.username}__grok",
        },
        {
            "name": "deepseek",
            "provider": "deepseek",
            "model": settings.deepseek_model,
            "api_key_present": bool(settings.deepseek_api_key),
            "saved_username": f"{request.username}__deepseek",
        },
    ]
    results: dict[str, Any] = {}
    for job in jobs:
        if not job["api_key_present"]:
            results[job["name"]] = {
                "ok": False,
                "skipped": True,
                "error": f"缺少 {job['name']} 的 API Key。",
                "model": job["model"],
            }
            continue
        try:
            profile = analyze_style_with_llm(
                request.username,
                posts,
                model=job["model"],
                provider=job["provider"],
            )
            if request.save_profiles:
                db.upsert_author(
                    username=job["saved_username"],
                    x_user_id=f"style:{job['saved_username']}",
                    display_name=f"{request.username} {job['name']} 风格画像",
                )
                db.save_style_profile(
                    username=job["saved_username"],
                    profile=profile,
                    sample_size=len(posts),
                    model=f"{job['provider']}:{job['model']}",
                )
            results[job["name"]] = {
                "ok": True,
                "skipped": False,
                "model": job["model"],
                "saved_username": job["saved_username"] if request.save_profiles else None,
                "profile": profile,
            }
        except LlmError as exc:
            results[job["name"]] = {
                "ok": False,
                "skipped": False,
                "model": job["model"],
                "error": str(exc),
            }

    return {
        "username": request.username.lower(),
        "sample_size": len(posts),
        "profiles_saved": request.save_profiles,
        "results": results,
        "next_step": "生成文章时，username 可以填 yijiangren__gpt、yijiangren__grok 或 yijiangren__deepseek。",
    }


@app.get("/styles/options", summary="列出生成工作台可选的风格对象")
def list_style_options() -> dict[str, Any]:
    return {"styles": base_style_options()}


@app.get("/style/{username}", summary="查看已经分析好的风格画像")
def get_style(username: str) -> dict[str, Any]:
    profile = db.get_style_profile(username)
    if not profile:
        raise HTTPException(status_code=404, detail="还没有风格画像。请先运行第 3 步 /style/analyze。")
    return profile


@app.post("/generate-three", summary="同时用 GPT、Grok、DeepSeek 生成三版文章")
def generate_three(request: GenerateThreeRequest) -> dict[str, Any]:
    brief = request.brief.strip()
    if not brief:
        raise HTTPException(status_code=400, detail="brief 不能为空。")

    results: dict[str, Any] = {}
    for job in model_jobs_for_style(request.style_name):
        profile_record = db.get_style_profile(job["profile_username"])
        if not profile_record:
            results[job["name"]] = {
                "ok": False,
                "model": job["model"],
                "profile_username": job["profile_username"],
                "error": f"找不到风格画像：{job['profile_username']}",
            }
            continue
        try:
            draft = generate_article_with_llm(
                username=job["profile_username"],
                style_profile=profile_record["profile"],
                brief=brief,
                platform=request.platform,
                target_length=request.target_length,
                extra_constraints=request.extra_constraints,
                model=job["model"],
                provider=job["provider"],
            )
            generation_id = db.save_generation(
                username=job["profile_username"],
                brief=brief,
                platform=request.platform,
                target_length=request.target_length,
                draft=draft,
                model=f"{job['provider']}:{job['model']}",
            )
            results[job["name"]] = {
                "ok": True,
                "id": generation_id,
                "model": job["model"],
                "profile_username": job["profile_username"],
                "draft": draft,
            }
        except LlmError as exc:
            results[job["name"]] = {
                "ok": False,
                "model": job["model"],
                "profile_username": job["profile_username"],
                "error": str(exc),
            }

    return {
        "style_name": request.style_name.strip().lstrip("@").lower(),
        "results": results,
    }


@app.post("/memory/learn-final", summary="学习你修改后的最终成稿，沉淀为可选风格")
def learn_final_article(request: LearnFinalArticleRequest) -> dict[str, Any]:
    style_name = request.style_name.strip().lstrip("@").lower()
    if not style_name:
        raise HTTPException(status_code=400, detail="style_name 不能为空。")
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="请粘贴最终成稿。")

    save_single_sample(
        username=style_name,
        text=text,
        title=request.title,
        source_url=None,
        display_name=request.display_name or style_name,
    )
    posts = db.get_posts(style_name, limit=200)
    profile = build_heuristic_profile(style_name, posts)
    profile["style_summary"] = (
        f"{request.display_name or style_name} 的个人最终稿风格，"
        f"基于 {len(posts)} 篇你确认后的成稿持续更新。"
    )
    profile["source_style"] = request.source_style
    profile["memory_note"] = "这个画像来自用户最终确认稿，后续生成时应优先吸收其表达偏好。"
    db.save_style_profile(
        username=style_name,
        profile=profile,
        sample_size=len(posts),
        model="user-memory-heuristic",
    )
    return {
        "ok": True,
        "style_name": style_name,
        "total_samples": db.count_posts(style_name),
        "profile_username": style_name,
        "next_step": "下次在生成工作台的风格对象里选择这个名称。",
    }


@app.post("/generate", summary="第 4 步：根据主题生成原创文章")
def generate(request: GenerateRequest) -> dict[str, Any]:
    profile_record = db.get_style_profile(request.username)
    if not profile_record:
        raise HTTPException(status_code=404, detail="还没有风格画像。请先运行第 3 步 /style/analyze。")

    model = request.model or settings.openai_model
    try:
        draft = generate_article_with_llm(
            username=request.username,
            style_profile=profile_record["profile"],
            brief=request.brief,
            platform=request.platform,
            target_length=request.target_length,
            extra_constraints=request.extra_constraints,
            model=model,
        )
    except LlmError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    generation_id = db.save_generation(
        username=request.username,
        brief=request.brief,
        platform=request.platform,
        target_length=request.target_length,
        draft=draft,
        model=model,
    )
    return {"id": generation_id, "username": request.username.lower(), "model": model, "draft": draft}
