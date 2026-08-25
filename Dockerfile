# demo6 RAG 对话服务 —— 国内轻量云部署镜像（Docker）
FROM python:3.10-slim

WORKDIR /app

# 国内网络优化：pip 走清华源；BGE 模型直接 COPY 进镜像（ModelScope 预下载），不走 HF
ENV HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers \
    HF_HUB_DISABLE_XET=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PYTHONUNBUFFERED=1 \
    HOME=/app \
    BGE_MODEL_PATH=/app/bge-model

# CPU 版 torch：从上交大镜像直接下载 wheel 本地安装（约184MB，实测 15MB/s+）。
# 不能用 pip --index-url 走该镜像：其 S3 后端返回 application/octet-stream，
# pip 拒绝解析该 Content-Type 索引。清华 PyPI 虽有 torch 但是 2.3GB 的 CUDA 版，
# 这里用 CPU 版更省（依赖在清华源自动解析）。
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://mirror.sjtu.edu.cn/pytorch-wheels/cpu/torch-2.9.1%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl', '/tmp/torch-2.9.1+cpu-cp310-cp310-manylinux_2_28_x86_64.whl')" && \
    pip install --no-cache-dir /tmp/torch-2.9.1+cpu-cp310-cp310-manylinux_2_28_x86_64.whl && \
    rm -f /tmp/torch-2.9.1+cpu-cp310-cp310-manylinux_2_28_x86_64.whl

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# BGE 模型：由部署方预下载到 bge-model/（ModelScope 实测 ~190MB/s），COPY . . 会自动带入，
# 运行时 main.py 读 BGE_MODEL_PATH=/app/bge-model 本地加载，不依赖 HuggingFace 网络。

COPY . .

# 运行时需写 chroma_db/，放开写权限
RUN chmod -R a+rwX /app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
