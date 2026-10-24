FROM python:3.13-slim

# uv for installing the locked dependencies
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# system libraries that OpenCV needs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_HTTP_TIMEOUT=300 \
    UV_LINK_MODE=copy \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# 1. install dependencies first, so this layer is cached between builds
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# 2. add the code and the trained weights
COPY src ./src
COPY models/best.pt ./models/best.pt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "brain_tumor.api.main:app", "--host", "0.0.0.0", "--port", "8000"]