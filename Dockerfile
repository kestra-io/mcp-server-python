FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /app
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt
COPY src/ ./src/
CMD ["uv", "run", "src/server.py"]