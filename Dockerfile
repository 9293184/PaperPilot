# PaperPilot 单服务镜像：后端 FastAPI + 托管前端构建产物
#
# 架构说明：
# - 前端用 Vite 构建出 frontend/dist；
# - 后端启动时会自动托管该目录（见 backend/app/main.py 的 SPA 兜底路由），
#   因此一个容器即可同时提供页面与 /api/v1 接口。
# - 运行时数据（SQLite、上传的 PDF、日志、api_config.json）默认在 /data，
#   云平台需把持久化卷挂到该路径（环境变量 PAPERPILOT_WORKSPACE_DIR）。

# ============ 阶段 1：构建前端 ============
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
# 先只拷贝依赖清单，利用 Docker 层缓存
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ============ 阶段 2：运行时 ============
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PAPERPILOT_WORKSPACE_DIR=/data

# tesseract 供 pytesseract 使用（扫描版 PDF 的 OCR 兜底）；
# 只装必要语言包，避免镜像过大。
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 后端依赖（单独一层，改动少时命中缓存）
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# 后端代码与内置数据（Data/ 含 schema.sql、seed.sql、seed_pdfs/）
COPY backend/ /app/backend/
COPY Data/ /app/Data/

# 前端构建产物：放到后端默认查找的位置（project_root/frontend/dist）
COPY --from=frontend-build /app/frontend/dist /app/frontend/dist

# 运行时数据目录（挂载卷后会被卷覆盖）
RUN mkdir -p /data

EXPOSE 8000

WORKDIR /app/backend
# 云平台通过 PORT 注入端口；本地默认 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
