# demo6 RAG 对话服务 —— Hugging Face Spaces (Docker SDK) 部署镜像
FROM python:3.10-slim

WORKDIR /app

# 模型/缓存放镜像内固定位置，运行时直接命中；HOME 也归到 /app 避免权限问题
ENV HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers \
    HF_HUB_DISABLE_XET=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/app

# 先装 CPU 版 torch：HF 免费 CPU 环境用不上 CUDA，省 2GB+ 构建时间
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 构建期预下载本地 BGE 模型，运行时秒级加载，避免首次请求超时
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-zh-v1.5')"

COPY . .

# HF 容器可能以非 root 用户运行，放开写权限（运行时需写 chroma_db/）
RUN chmod -R a+rwX /app

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
