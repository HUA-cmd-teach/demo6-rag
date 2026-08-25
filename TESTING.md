# demo6：AI 智能对话（RAG 检索 + 流式输出 + 智能体前端）

前后端联调示例：**一个输入框 + 一个输出框**的豆包风格智能体页面，支持流式打字机输出和知识库 RAG 检索。

## 快速开始

```bash
cd D:\NUC\demo6
D:\NUC\.venv\Scripts\uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

浏览器打开 <http://localhost:8000>

> ⚠️ 必须先启动 OpenSearch？不需要。demo6 用 **FAISS 本地持久化向量库 + jieba/BM25 混合检索**，无需额外服务。
> ⚠️ 需要联网的部分只有回答模型（通义千问 DashScope），嵌入用本地 BGE 模型。

## 功能验证清单

| 功能 | 验证方式 | 结果 |
|------|----------|------|
| RAG 检索 | 提问"DeepSeek 的核心技术是什么？"，回答下方出现「📄 参考 N 篇文档」来源引用 | ✅ |
| 流式输出 | 发送后回答逐字逐段出现（打字机效果 + 闪烁光标） | ✅ |
| 停止生成 | 生成中点输入框右侧红色停止按钮，立即中断 | ✅ |
| 前端页面 | 居中布局：顶部标题栏 / 中部输出区 / 底部输入框，空态有引导建议词 | ✅ |
| 短期记忆 | 连续对话，回答上方显示「会话第 N 轮」逐步递增；服务端自动注入上下文 | ✅ |
| 长期记忆 | 先说"我叫张三，我喜欢做后端"，新会话后再问相关话题，能召回「🧠 长期记忆」 | ✅ |
| 会话/用户隔离 | 换 `X-Session-Id` 历史清零但长期记忆保留；换 `X-Client-Id` 全部不可见 | ✅ |

## 项目结构

```
demo6/
├── main.py        # FastAPI 后端：启动时构建 FAISS + jieba/BM25 索引 + /api/chat 流式接口
├── index.html     # 豆包风格前端：一个输入框 + 一个输出框（fetch 流式解析 SSE）
├── data/          # 知识库文档（txt/md/docx），可自行增删
└── faiss_index/   # （自动生成）FAISS 持久化向量索引
```

## 接口说明

### POST /api/chat — 流式问答（SSE）

请求头（无登录的用户/会话隔离）：
```
X-Client-Id: <匿名用户ID>   前端首访生成，localStorage 保存
X-Session-Id: <会话ID>      每次对话一个，前端「新会话」按钮重置
```
请求体：
```json
{ "query": "DeepSeek 的核心技术是什么？", "top_k": 3 }
```
`history` 字段已废弃：多轮历史由服务端按 `(client_id, session_id)` 保存（短期记忆）并自动注入。

响应为 SSE 流，六种事件：
```
event: session  # data: {session_id, history_turns}             会话确认 + 短期记忆轮数
event: memory   # data: {facts:[{text, score}]}                 长期记忆召回（BGE 向量相似度）
event: source   # data: {sources:[{score, source, semantic, bm25, text}]}  检索来源
event: token    # data: {text: "..."}                            回答增量（逐字渲染）
event: done     # data: {answer: "..."}                          完整回答
event: error    # data: {message: "..."}                         出错信息
```

### GET /api/health — 健康检查
### GET / — 前端页面

## 技术栈

| 组件 | 方案 | 说明 |
|------|------|------|
| 语义检索 | FAISS（本地持久化） | 无需启动外部服务，索引在 `faiss_index/` |
| 关键词检索 | jieba 分词 + BM25 | `rank-bm25`，与语义结果 RRF 融合 |
| 记忆 | 长短期记忆（JSON 持久化） | 会话历史 + 用户事实抽取，BGE 向量召回长期记忆 |
| 用户/会话 | 匿名 ID（无登录） | `X-Client-Id`/`X-Session-Id` 请求头，localStorage 保存 |
| 嵌入模型 | 本地 BGE-base-zh-v1.5 | 离线，`demo03/local_models/`；查询自动加 BGE 检索前缀 |
| 回答模型 | 通义千问（DashScope） | OpenAI 兼容接口，`ChatOpenAI.stream()` 流式 |
| 前端 | 原生 HTML/JS | `fetch` + `ReadableStream` 按字节累积解析 SSE，无需构建工具 |

## 常用配置（main.py 顶部）

- `REBUILD_ON_START = True`：每次启动重建索引，保证 索引 == data 目录。数据多时可改为 False 复用旧索引
- `LLM_MODEL`：默认 `qwen3.7-plus`，可设环境变量 `QWEN_MODEL` 覆盖
- `API_KEY`：读 `DASHSCOPE_API_KEY`（其次 `API_KEY`）；未配置时接口退化为直接返回检索到的原文

## 测试要点（本次联调已实测）

1. 启动日志应看到 `已加载 DeepSeek 技术介绍.docx` / `索引就绪：N 个文本块 -> FAISS(重建) + jieba/BM25`
2. `curl -s -N -X POST .../api/chat -d '{"query":"DeepSeek 的核心技术是什么？"}'` 应依次收到 `event: session` → `event: memory` → `event: source` → 多个 `event: token` → `event: done`
3. 前端 SSE 解析逻辑对真实字节验证通过：token 拼接结果 == done 的完整回答
4. Windows 中文终端显示乱码属 GBK 控制台编码问题，实际传输是 UTF-8，不影响浏览器
5. **记忆验证（2026-08-25 实测）**：user-A 先发"我叫张三。我喜欢做后端开发。"→ `memory_store.json` 出现 2 条 facts；换 `X-Session-Id: session-2` 问"我的兴趣是什么"→ `memory` 事件召回"用户提到：我喜欢做后端开发"(score≈0.41)；换 `X-Client-Id: user-B` 问同一问题 → `memory` 事件为空（用户隔离）。换回 session-1 → `session` 事件 `history_turns` 递增（短期记忆保留）。
6. **启动命令**：`LANGCHAIN_OPENAI_TCP_KEEPALIVE=0` 前缀抑制 langchain-openai transport 注入警告；不配置 key 时接口退化为 fallback 秒回（记忆逻辑同样生效，方便离线测记忆）。
