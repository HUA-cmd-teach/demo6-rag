"""
demo6：AI 智能对话 + RAG 检索 + 流式输出（前后端联调）
======================================================
后端：FastAPI
  * 启动时扫描 data/ 目录 → 本地 BGE 模型向量化 → 写入 Chroma 持久化索引
  * POST /api/chat  SSE 流式接口
      event: source  检索到的知识来源（前端展示引用）
      event: token   回答内容增量（前端逐字渲染，实现流式打字机效果）
      event: done    回答结束
      event: error   出错信息
  * GET  /           返回豆包风格前端页面（一个输入框 + 一个输出框）

前端：index.html（fetch + ReadableStream 解析 SSE）

运行：
    cd D:\\NUC\\demo6
    D:\\NUC\\.venv\\Scripts\\uvicorn main:app --reload --host 0.0.0.0 --port 8000
浏览器打开 http://localhost:8000

技术栈（离线优先）：
  向量库   = Chroma（持久化，无需额外启动服务）
  嵌入模型 = 本地 BGE-base-zh-v1.5（不联网）
  回答模型 = 通义千问（DashScope，OpenAI 兼容接口，支持流式）
"""
import json
import os
import shutil
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

# ==================== 配置 ====================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"                              # 知识库文档目录
CHROMA_DIR = BASE_DIR / "chroma_db"                       # 向量库持久化目录
COLLECTION = "demo6_rag"                                  # Chroma 集合名
# BGE 模型：本机优先用本地路径，否则从 HuggingFace Hub 下载同名模型（云部署场景）
_BGE_LOCAL = r"D:\NUC\demo03\local_models\bge-base-zh-v1.5"
BGE_MODEL_PATH = os.environ.get("BGE_MODEL_PATH") or (
    _BGE_LOCAL if os.path.exists(_BGE_LOCAL) else "BAAI/bge-base-zh-v1.5"
)
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文档："          # BGE 检索前缀
REBUILD_ON_START = True                                  # 每次启动重建索引，保证 索引 == data 目录

# 回答模型（DashScope 通义千问，OpenAI 兼容接口）
API_KEY = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("API_KEY")
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = os.environ.get("QWEN_MODEL", "qwen3.7-plus")

# ==================== 嵌入模型（本地 BGE）====================
from sentence_transformers import SentenceTransformer


class BGEEmbeddings:
    """本地 BGE 嵌入（langchain 嵌入接口的鸭子类型）。
    BGE 官方要求：文档侧不加前缀，查询侧加检索前缀，这里在 embed_query 中处理。"""

    def __init__(self):
        print("正在加载本地 BGE 模型，首次可能需要几秒...")
        self.model = SentenceTransformer(BGE_MODEL_PATH, device="cpu")
        print("BGE 模型加载完成")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode([QUERY_PREFIX + text], normalize_embeddings=True)
        return vector.tolist()[0]


_embeddings = None


def get_embeddings() -> BGEEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = BGEEmbeddings()
    return _embeddings


# ==================== 文档加载与拆分 ====================
def load_and_split() -> list:
    """加载 data/ 目录下所有支持的文档（txt/md/docx）并拆分成文本块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "，", "、"],
    )
    docs = []
    for path in sorted(DATA_DIR.iterdir()):
        suffix = path.suffix.lower()
        if suffix in (".txt", ".md"):
            loader = TextLoader(str(path), encoding="utf-8")
        elif suffix == ".docx":
            loader = Docx2txtLoader(str(path))
        else:
            continue
        if path.stat().st_size == 0:          # 跳过空文件
            continue
        docs.extend(splitter.split_documents(loader.load()))
        print(f"已加载 {path.name}")
    return docs


def build_index() -> Chroma:
    """重建向量索引：先删旧库再写入，保证索引内容 == data 目录（与 demo03/demo04 思路一致）"""
    docs = load_and_split()
    if CHROMA_DIR.exists():                   # 删除旧索引，全新重建
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    vectorstore = Chroma.from_documents(
        docs,
        get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION,
    )
    print(f"索引构建完成：{len(docs)} 个文本块 -> {COLLECTION}")
    return vectorstore


# ==================== FastAPI 应用 ====================
app = FastAPI(title="demo6 AI 智能对话", description="RAG 检索 + 流式输出 + 智能体前端", version="1.0.0")

# CORS：开发环境放开，方便前端单独打开页面调试
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 启动时构建索引（若无需每次重建可把 REBUILD_ON_START 改为 False）
print("正在初始化知识库索引...")
vectorstore = build_index()


# ==================== SSE 辅助 ====================
def sse(event: str, data: dict) -> str:
    """把事件拼成 SSE 文本（data 是 JSON，ensure_ascii=False 保证中文可见）"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ==================== 接口 1：POST /api/chat（流式对话）====================
class ChatRequest(BaseModel):
    query: str = Field(..., description="用户问题")
    top_k: int = Field(3, ge=1, le=10, description="检索返回的文档块数量")
    history: list = Field(default_factory=list, description="多轮对话历史：[{role: user/assistant, content}]")


@app.post("/api/chat")
def chat(request: ChatRequest):
    """RAG 流式问答：向量检索 → SSE 流式输出 LLM 回答"""
    query = request.query.strip()
    if not query:
        return StreamingResponse(iter([sse("error", {"message": "问题不能为空"})]), media_type="text/event-stream")

    def gen():
        # 1. 向量检索知识库
        try:
            hits = vectorstore.similarity_search_with_score(query, k=request.top_k)
        except Exception as e:
            yield sse("error", {"message": f"检索失败：{e}"})
            return

        if not hits:
            msg = "知识库暂无相关内容。请往 data/ 目录添加 txt/md/docx 文档后重启服务。"
            yield sse("token", {"text": msg})
            yield sse("done", {"answer": msg})
            return

        # 2. 先发检索来源，前端展示引用
        sources = [
            {
                "score": round(score, 4),
                "source": Path(doc.metadata.get("source", "")).name,
                "text": doc.page_content[:200],
            }
            for doc, score in hits
        ]
        yield sse("source", {"sources": sources})

        context = "\n\n".join(f"[文档{i + 1}] {doc.page_content}" for i, (doc, _) in enumerate(hits))

        # 3. 组装对话消息（系统提示 + 多轮历史 + 当前问题）
        system_prompt = (
            "你是知识库智能助手。请优先依据提供的文档上下文回答；"
            "若上下文与问题无关或不足以回答，则根据自己的知识直接回答。"
            "回答要简洁、准确、有条理。\n\n"
            f"参考文档上下文：\n{context}"
        )
        messages = [SystemMessage(content=system_prompt)]
        for h in request.history:
            if h.get("role") == "user":
                messages.append(HumanMessage(content=h.get("content", "")))
            elif h.get("role") == "assistant":
                messages.append(AIMessage(content=h.get("content", "")))
        messages.append(HumanMessage(content=query))

        # 4. 流式调用通义千问
        if not API_KEY:
            fallback = "未配置 DASHSCOPE_API_KEY，无法调用大模型。以上是检索到的知识原文：\n\n" + context
            yield sse("token", {"text": fallback})
            yield sse("done", {"answer": fallback})
            return

        try:
            llm = ChatOpenAI(
                api_key=API_KEY,
                base_url=LLM_BASE_URL,
                model=LLM_MODEL,
                temperature=0,
                streaming=True,
            )
            full = ""
            for chunk in llm.stream(messages):
                token = chunk.content
                if not token:
                    continue
                full += token
                yield sse("token", {"text": token})
            yield sse("done", {"answer": full})
        except Exception as e:
            yield sse("error", {"message": f"模型调用失败：{e}"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ==================== 接口 2：GET /api/health（健康检查）====================
@app.get("/api/health")
def health():
    return {"code": 200, "message": "demo6 服务运行正常", "model": LLM_MODEL}


# ==================== 接口 3：GET /（前端页面）====================
@app.get("/")
def index():
    return FileResponse(BASE_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
