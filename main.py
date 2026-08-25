"""
demo6：AI 智能对话 + RAG 混合检索 + 流式输出（前后端联调）
======================================================
后端：FastAPI
  * 启动时扫描 data/ 目录 → 本地 BGE 模型向量化 → 写入 FAISS 持久化索引；
    同时用 jieba 分词 + BM25 构建关键词索引（启动重建，保证索引 == data）
  * 检索 = FAISS 语义召回 + jieba/BM25 关键词召回，RRF（Reciprocal Rank Fusion）融合
  * POST /api/chat  SSE 流式接口
      event: source  检索到的知识来源（含语义分/BM25分/融合分，前端展示引用）
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
  语义检索   = FAISS（持久化，无需额外启动服务）
  关键词检索 = jieba 分词 + BM25（rank-bm25）
  融合策略   = RRF（Reciprocal Rank Fusion，k=60）
  嵌入模型   = 本地 BGE-base-zh-v1.5（不联网）
  回答模型   = 通义千问（DashScope，OpenAI 兼容接口，支持流式）
"""
import asyncio
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import jieba
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("demo6")

# ==================== 配置 ====================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"                              # 知识库文档目录
FAISS_DIR = BASE_DIR / "faiss_index"                      # FAISS 持久化目录
COLLECTION = "demo6_rag"                                  # 集合名（FAISS 内部使用）
HYBRID_CANDIDATES = 20   # 每路检索器各取 top20 候选，融合后再取 top_k
RRF_K = 60               # RRF 常数，越小越看重头部排名
# BGE 模型：本机优先用本地路径，否则从 HuggingFace Hub 下载同名模型（云部署场景）
_BGE_LOCAL = r"D:\NUC\demo03\local_models\bge-base-zh-v1.5"
BGE_MODEL_PATH = os.environ.get("BGE_MODEL_PATH") or (
    _BGE_LOCAL if os.path.exists(_BGE_LOCAL) else "BAAI/bge-base-zh-v1.5"
)
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文档："          # BGE 检索前缀
REBUILD_ON_START = True                                  # True=启动重建索引；False=加载已有 FAISS 索引

# 用户/会话记忆（无登录：前端自生成匿名 client_id，见 index.html）
STORE_PATH = BASE_DIR / "memory_store.json"   # 记忆持久化文件（运行时生成，勿提交）
SHORT_MEMORY_TURNS = 12                       # 每个会话保留最近 N 轮（短期记忆）
LONG_MEMORY_TOP_K = 3                         # 每次提问召回的长期记忆条数
LONG_MEMORY_THRESHOLD = 0.15                  # 长期记忆相似度低于此值视为无关

# 并发与限流（防公网滥用 / 线程池耗尽）
MAX_CONCURRENT_CHATS = 10                     # 同时进行的对话数上限
RATE_LIMIT_MAX = 30                           # 每个 client_id 每窗口最大请求数
RATE_LIMIT_WINDOW = 60                        # 限流窗口（秒）

# 回答模型（DashScope 通义千问，OpenAI 兼容接口）
API_KEY = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("API_KEY")
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = os.environ.get("QWEN_MODEL", "qwen3.7-plus")

# 绕开 Windows 系统代理（本机 127.0.0.1:26561 是 MITM 代理，证书不信任导致
# httpx 走代理时报 SSL CERTIFICATE_VERIFY_FAILED）。trust_env=False 直连 DashScope，
# 模块级复用同一 Client。Docker 云部署无该系统代理，此设置只是保持直连，同样安全。
HTTPX_CLIENT = httpx.Client(trust_env=False)

# ==================== 嵌入模型（本地 BGE）====================
from sentence_transformers import SentenceTransformer


class BGEEmbeddings(Embeddings):
    """本地 BGE 嵌入（继承 langchain Embeddings 抽象类，FAISS 需要 isinstance 判断）。
    BGE 官方要求：文档侧不加前缀，查询侧加检索前缀，这里在 embed_query 中处理。"""

    def __init__(self):
        logger.info("正在加载本地 BGE 模型，首次可能需要几秒...")
        self.model = SentenceTransformer(BGE_MODEL_PATH, device="cpu")
        logger.info("BGE 模型加载完成")

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
        logger.info(f"已加载 {path.name}")
    return docs


# ==================== 混合检索索引（FAISS + jieba/BM25）====================
DOCS: list = []          # 文本块（BM25 语料，列表下标即语料索引）
vectorstore = None       # FAISS 语义索引
bm25 = None              # jieba 分词 + BM25 关键词索引


def build_index():
    """重建混合检索索引：保证索引内容 == data 目录（与 demo03/demo04 思路一致）。
    语义索引存 FAISS，关键词索引由 jieba+BM25 构建，查询时用 RRF 融合两者排名。"""
    global DOCS, vectorstore, bm25
    docs = load_and_split()
    embeddings = get_embeddings()

    if REBUILD_ON_START or not FAISS_DIR.exists():
        if FAISS_DIR.exists():                            # 删除旧索引，全新重建
            shutil.rmtree(FAISS_DIR, ignore_errors=True)
        # 给每个文本块打上语料序号，融合阶段靠它回对齐 BM25 分数
        for i, doc in enumerate(docs):
            doc.metadata["_idx"] = i
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(str(FAISS_DIR))
        faiss_note = "重建"
    else:
        vectorstore = FAISS.load_local(
            str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True
        )
        faiss_note = "加载"

    # 关键词索引：jieba 分词 + BM25（每次启动重建，开销很小）
    tokenized_corpus = [jieba.lcut(doc.page_content) for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    DOCS = docs
    logger.info(f"索引就绪：{len(docs)} 个文本块 -> FAISS({faiss_note}) + jieba/BM25")


def hybrid_search(query: str, top_k: int = 3) -> list[dict]:
    """混合召回：FAISS 语义 + jieba/BM25 关键词，RRF 融合后取 top_k。

    RRF 对每个检索器结果的排名求 1/(k+rank) 再相加，天然规避两种分数
    量纲不一致（FAISS 是 L2 距离、BM25 是词频分）的问题，无需归一化。
    """
    # 1) FAISS 语义召回（score 为 L2 距离，越小越相似）
    faiss_hits = vectorstore.similarity_search_with_score(query, k=HYBRID_CANDIDATES)

    # 2) jieba 分词 + BM25 关键词召回
    bm25_scores = bm25.get_scores(jieba.lcut(query))
    bm25_rank = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)

    # 3) RRF 融合
    fusion: dict[int, dict] = {}
    for rank, (doc, score) in enumerate(faiss_hits):
        idx = doc.metadata["_idx"]
        fusion[idx] = {"doc": doc, "rrf": 1.0 / (RRF_K + rank), "semantic": score, "bm25": 0.0}
    for rank, idx in enumerate(bm25_rank):
        entry = fusion.setdefault(
            idx, {"doc": DOCS[idx], "rrf": 0.0, "semantic": None, "bm25": 0.0}
        )
        entry["rrf"] += 1.0 / (RRF_K + rank)
        entry["bm25"] = float(bm25_scores[idx])

    # 4) 按融合分排序，取 top_k
    return sorted(fusion.values(), key=lambda e: e["rrf"], reverse=True)[:top_k]


# ==================== 用户/会话记忆（长短期记忆 + 用户隔离）====================
# 无登录方案：前端首访生成匿名 client_id（用户维度）+ 每次对话一个 session_id（会话维度），
# 经请求头传入。服务端按 (client_id, session_id) 隔离存储，互不可见。

# 长期记忆抽取模式：命中任一模式的句子记为"用户事实"入库
FACT_PATTERNS = [
    r"我叫\S{1,10}", r"我名字叫\S{1,10}", r"我今年\S{1,10}岁",
    r"我喜欢\S{1,30}", r"我不喜欢\S{1,30}", r"我讨厌\S{1,30}",
    r"我擅长\S{1,20}", r"我在\S{1,20}工作", r"我住在\S{1,20}", r"我住\S{1,20}",
    r"我的职业\S{1,20}", r"我的公司\S{1,30}", r"我的项目\S{1,30}",
    r"我的爱好\S{1,20}", r"我的目标\S{1,30}", r"我的梦想\S{1,20}",
]


class MemoryStore:
    """按 (client_id, session_id) 隔离的轻量记忆库，JSON 文件持久化，重启不丢。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        # RLock（可重入）：append_turn/add_facts 在持锁状态下再调 _save() 会二次加锁，
        # 普通 Lock 会死锁并卡死整个请求线程（表现为服务"假死"、memory_store.json 永不生成）。
        self._lock = threading.RLock()
        self.data = {"users": {}}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {"users": {}}

    def _save(self):
        with self._lock:
            self.path.write_text(
                json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def _user(self, client_id: str) -> dict:
        return self.data["users"].setdefault(client_id, {"sessions": {}, "facts": []})

    def get_history(self, client_id: str, session_id: str) -> list:
        """返回该会话已存的短期记忆（不含当前这一条）"""
        with self._lock:
            return self._user(client_id)["sessions"].setdefault(session_id, {"history": []})["history"]

    def append_turn(self, client_id: str, session_id: str, user_msg: str, assistant_msg: str):
        """保存一轮到会话短期记忆，只保留最近 SHORT_MEMORY_TURNS 轮"""
        with self._lock:
            hist = self._user(client_id)["sessions"].setdefault(session_id, {"history": []})["history"]
            hist.append({"role": "user", "content": user_msg})
            if assistant_msg:
                hist.append({"role": "assistant", "content": assistant_msg})
            if len(hist) > SHORT_MEMORY_TURNS * 2:
                del hist[: len(hist) - SHORT_MEMORY_TURNS * 2]
            self._save()

    def add_facts(self, client_id: str, sentences: list) -> list:
        """把抽取的用户事实加入长期记忆（去重、限条数）"""
        added = []
        with self._lock:
            facts = self._user(client_id)["facts"]
            existing = {f["text"] for f in facts}
            for s in sentences:
                if s not in existing and len(facts) < 50:
                    facts.append({"text": s})
                    existing.add(s)
                    added.append(s)
            if added:
                self._save()
        return added

    def get_facts(self, client_id: str) -> list:
        with self._lock:
            return [f["text"] for f in self._user(client_id)["facts"]]

    def stats(self) -> dict:
        with self._lock:
            users = len(self.data["users"])
            sessions = sum(len(u.get("sessions", {})) for u in self.data["users"].values())
            facts = sum(len(u.get("facts", [])) for u in self.data["users"].values())
        return {"users": users, "sessions": sessions, "facts": facts}


store = MemoryStore(STORE_PATH)

# ==================== 并发与限流 ====================
# 并发信号量：超过 MAX_CONCURRENT_CHATS 的对话直接拒绝，避免慢请求长期占满线程池
_chat_slots = threading.BoundedSemaphore(MAX_CONCURRENT_CHATS)

# 滑动窗口限流：{client_id: [请求时间戳]}，防止公网接口被刷爆 API key 预算
_ratelimit: dict[str, list[float]] = {}
_ratelimit_lock = threading.Lock()


def rate_limited(client_id: str) -> bool:
    """返回 True 表示该 client_id 在窗口内请求超限。"""
    now = time.monotonic()
    with _ratelimit_lock:
        ts = [t for t in _ratelimit.get(client_id, []) if now - t < RATE_LIMIT_WINDOW]
        if len(ts) >= RATE_LIMIT_MAX:
            _ratelimit[client_id] = ts
            return True
        ts.append(now)
        _ratelimit[client_id] = ts
        return False


def extract_facts(text: str) -> list:
    """从一句话/一段话里抽取"用户事实"句：按标点切句后逐句匹配模式"""
    facts = []
    for sentence in re.split(r"[。！？!?\n]+", text):
        sentence = sentence.strip()
        if len(sentence) < 4 or len(sentence) > 60:
            continue
        if any(re.search(p, sentence) for p in FACT_PATTERNS):
            facts.append(f"用户提到：{sentence}")
    return facts


def recall_facts(client_id: str, query: str) -> list:
    """长期记忆召回：用 BGE 向量算查询与已存事实的余弦相似度，取 top_k"""
    facts = store.get_facts(client_id)
    if not facts:
        return []
    emb = get_embeddings()
    qvec = emb.embed_query(query)
    fvecs = emb.embed_documents(facts)
    # BGE 输出已归一化，余弦相似度 == 点积
    sims = [sum(a * b for a, b in zip(qvec, fvec)) for fvec in fvecs]
    order = sorted(range(len(facts)), key=lambda i: sims[i], reverse=True)
    return [
        {"text": facts[i], "score": round(sims[i], 3)}
        for i in order[:LONG_MEMORY_TOP_K]
        if sims[i] >= LONG_MEMORY_THRESHOLD
    ]


# ==================== FastAPI 应用 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时构建混合检索索引（改到 lifespan，避免 import 时阻塞）。"""
    logger.info("正在初始化知识库索引...")
    await asyncio.to_thread(build_index)
    yield


app = FastAPI(title="demo6 AI 智能对话", description="RAG 混合检索 + 流式输出 + 智能体前端", version="1.0.0", lifespan=lifespan)

# CORS：仅本地把前端单独打开时调试用；同源部署不生效。
# 注意：allow_origins=["*"] 与 allow_credentials=True 浏览器规范冲突，去掉 credentials。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== SSE 辅助 ====================
def sse(event: str, data: dict) -> str:
    """把事件拼成 SSE 文本（data 是 JSON，ensure_ascii=False 保证中文可见）"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ==================== 接口 1：POST /api/chat（流式对话）====================
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="用户问题")
    top_k: int = Field(3, ge=1, le=10, description="融合后返回的文档块数量")


@app.post("/api/chat")
def chat(request: Request, body: ChatRequest):
    """RAG 流式问答：会话隔离 + 长短期记忆 + 混合检索 → SSE 流式输出。

    请求头：
      X-Client-Id   用户维度匿名 ID（前端首访生成，localStorage 保存）
      X-Session-Id  会话维度 ID（每次对话一个，前端「新会话」按钮重置）
    SSE 事件：session / memory / source / token / done / error
    """
    client_id = (request.headers.get("X-Client-Id") or "").strip()
    session_id = (request.headers.get("X-Session-Id") or "").strip() or uuid.uuid4().hex
    query = body.query.strip()

    if not client_id:
        return StreamingResponse(
            iter([sse("error", {"message": "缺少 X-Client-Id 请求头，请在请求头中带上客户端标识"})]),
            media_type="text/event-stream",
        )
    if not query:
        return StreamingResponse(iter([sse("error", {"message": "问题不能为空"})]), media_type="text/event-stream")

    # 限流：同一 client_id 窗口内请求超限直接拒绝（防公网刷 API key 预算）
    if rate_limited(client_id):
        return StreamingResponse(
            iter([sse("error", {"message": "请求过于频繁，请稍后再试"})]),
            media_type="text/event-stream",
            status_code=429,
        )

    # 并发上限：防止慢请求长期占满线程池（拿不到信号量就立刻拒绝）
    if not _chat_slots.acquire(blocking=False):
        return StreamingResponse(
            iter([sse("error", {"message": "当前同时对话数过多，请稍后再试"})]),
            media_type="text/event-stream",
            status_code=429,
        )

    def _gen_inner():
        # 0) 会话事件：确认当前会话，并告知前端短期记忆轮数
        history = store.get_history(client_id, session_id)
        yield sse("session", {"session_id": session_id, "history_turns": len(history) // 2})

        # 1) 长期记忆召回（按查询向量相似度取 top_k）
        facts = recall_facts(client_id, query)
        yield sse("memory", {"facts": facts})

        # 1.5) 抽取本条消息中的用户事实并立即入库（长期记忆）。
        #      放在 LLM 调用之前：即使模型超时/失败/客户端断开，事实也已持久化，
        #      下次提问即可跨会话召回。fallback 路径（无 API_KEY）同样生效。
        store.add_facts(client_id, extract_facts(query))

        # 2) 混合检索知识库（FAISS 语义 + jieba/BM25 关键词，RRF 融合）
        try:
            hits = hybrid_search(query, top_k=body.top_k)
        except Exception as e:
            yield sse("error", {"message": f"检索失败：{e}"})
            return

        if not hits:
            msg = "知识库暂无相关内容。请往 data/ 目录添加 txt/md/docx 文档后重启服务。"
            yield sse("token", {"text": msg})
            yield sse("done", {"answer": msg})
            store.append_turn(client_id, session_id, query, msg)
            return

        # 3) 先发检索来源，前端展示引用（含语义分/BM25分/融合分）
        sources = [
            {
                "score": round(float(h["rrf"]), 4),
                # FAISS 返回的 L2 距离是 numpy.float32，先转 float 才能 JSON 序列化
                "semantic": round(float(h["semantic"]), 4) if h["semantic"] is not None else None,
                "bm25": round(float(h["bm25"]), 4),
                "source": Path(h["doc"].metadata.get("source", "")).name,
                "text": h["doc"].page_content[:200],
            }
            for h in hits
        ]
        yield sse("source", {"sources": sources})

        context = "\n\n".join(f"[文档{i + 1}] {h['doc'].page_content}" for i, h in enumerate(hits))

        # 4) 组装消息：系统提示（长期记忆 + 文档上下文）+ 会话短期历史 + 当前问题
        system_prompt = (
            "你是知识库智能助手。请优先依据提供的文档上下文回答；"
            "若上下文与问题无关或不足以回答，则根据自己的知识直接回答。"
            "回答要简洁、准确、有条理。"
        )
        memory_block = "\n".join(f"- {f['text']}" for f in facts)
        if memory_block:
            system_prompt += f"\n\n关于用户的长久记忆（供回答参考，不要向用户复述）：\n{memory_block}"
        system_prompt += f"\n\n参考文档上下文：\n{context}"

        messages = [SystemMessage(content=system_prompt)]
        for h in history:  # 服务端保存的会话短期记忆，自动注入多轮上下文
            if h["role"] == "user":
                messages.append(HumanMessage(content=h["content"]))
            else:
                messages.append(AIMessage(content=h["content"]))
        messages.append(HumanMessage(content=query))

        # 5) 流式调用通义千问
        if not API_KEY:
            fallback = "未配置 DASHSCOPE_API_KEY，无法调用大模型。以上是检索到的知识原文：\n\n" + context
            yield sse("token", {"text": fallback})
            yield sse("done", {"answer": fallback})
            store.append_turn(client_id, session_id, query, fallback)
            return

        full = ""
        try:
            llm = ChatOpenAI(
                api_key=API_KEY,
                base_url=LLM_BASE_URL,
                model=LLM_MODEL,
                temperature=0,
                streaming=True,
                http_client=HTTPX_CLIENT,
                # 分阶段超时：连接/写入/连接池 20s，读取（含流式逐块）180s。
                # 防止模型调用挂起时整个 SSE 请求永不返回、长期占死线程池线程。
                timeout=httpx.Timeout(connect=20, write=20, pool=20, read=180),
            )
            for chunk in llm.stream(messages):
                token = chunk.content
                if not token:
                    continue
                full += token
                yield sse("token", {"text": token})
            yield sse("done", {"answer": full})
        except Exception as e:
            yield sse("error", {"message": f"模型调用失败：{e}"})
        finally:
            # 保存本轮进会话短期记忆（长期记忆已在 LLM 调用前抽取入库）
            if full:
                store.append_turn(client_id, session_id, query, full)

    def gen():
        """外层生成器：负责在对话结束后释放并发信号量（异常/客户端断开也会走 finally）。"""
        try:
            yield from _gen_inner()
        finally:
            _chat_slots.release()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ==================== 接口 2：GET /api/history（恢复历史对话）====================
@app.get("/api/history")
def history(request: Request):
    """返回当前会话的短期记忆，供前端刷新页面时恢复对话显示。"""
    client_id = (request.headers.get("X-Client-Id") or "").strip()
    session_id = (request.headers.get("X-Session-Id") or "").strip()
    if not client_id:
        return {"code": 400, "message": "缺少 X-Client-Id 请求头"}
    return {"code": 200, "session_id": session_id, "turns": store.get_history(client_id, session_id)}


# ==================== 接口 3：GET /api/health（健康检查）====================
@app.get("/api/health")
async def health():
    # async def：直接跑在事件循环上，不占线程池。
    # 即使 chat 的线程池被慢请求占满，健康检查也始终即时响应。
    return {"code": 200, "message": "demo6 服务运行正常", "model": LLM_MODEL, "memory": store.stats()}


# ==================== 接口 4：GET /（前端页面）====================
@app.get("/")
def index():
    return FileResponse(BASE_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
