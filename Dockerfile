# syntax=docker/dockerfile:1

FROM node:20-alpine AS web-build
WORKDIR /web
COPY frontend/web/package.json frontend/web/package-lock.json* ./
RUN npm install
COPY frontend/web/ ./
RUN npm run build

FROM python:3.11-slim AS runtime
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY frontend/local_rag/requirements.txt /app/frontend/local_rag/requirements.txt
RUN pip install -r /app/frontend/local_rag/requirements.txt

COPY . /app
COPY --from=web-build /web/dist /app/frontend/web/dist

EXPOSE 8000
CMD ["python", "run.py"]
