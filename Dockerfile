FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /app
COPY requirements.txt .
RUN uv pip install --system httpx pydantic pyyaml python-dotenv fastmcp
COPY src/ ./src/
CMD ["uv", "run", "src/server.py"]

# Build using: docker build -t kestra-mcp-python .