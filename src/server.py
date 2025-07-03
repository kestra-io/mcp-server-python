from dotenv import load_dotenv
import httpx
from mcp.server.fastmcp import FastMCP
import os

from tools.backfill import register_backfill_tools
from tools.ee import register_ee_tools
from tools.execution import register_execution_tools
from tools.files import register_files_tools
from tools.flow import register_flow_tools
from tools.kv import register_kv_tools
from tools.namespace import register_namespace_tools
from tools.resume import register_resume_tools
from tools.replay import register_replay_tools
from tools.restart import register_restart_tools

load_dotenv()


def make_kestra_client() -> httpx.AsyncClient:
    """
    Builds an AsyncClient that supports:
      • OSS (no auth)
      • OSS with BasicAuth
      • EE with Bearer token
      • EE multi‑tenant via path prefix + header
    """

    # 1) Base URL (may include tenant)
    base = os.getenv("KESTRA_BASE_URL", "http://localhost:8080/api/v1")
    if tenant := os.getenv("KESTRA_TENANT_ID"):
        base = f"{base.rstrip('/')}/{tenant}"

    # 2) Auth + headers
    headers: dict[str, str] = {}
    auth = None

    # 3) Basic auth (OSS or EE)
    if (user := os.getenv("KESTRA_USERNAME")) and (pwd := os.getenv("KESTRA_PASSWORD")):
        auth = httpx.BasicAuth(user, pwd)
    # 4) Or Bearer token (EE)
    elif token := os.getenv("KESTRA_API_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    if tenant:
        headers["X-Kestra-Tenant"] = tenant

    return httpx.AsyncClient(base_url=base, auth=auth, headers=headers)


mcp = FastMCP(
    "Kestra MCP",
    instructions="""
        Help users interact with Kestra. For every user query:
        1. Select the single most relevant tool for the question.
        2. Invoke only that one tool; do not call any others.
        """,
)
client = make_kestra_client()

# e.g. KESTRA_MCP_DISABLED_TOOLS=ee
DISABLED_TOOLS = os.getenv("KESTRA_MCP_DISABLED_TOOLS", "").split(",")
DISABLED_TOOLS = [tool.strip() for tool in DISABLED_TOOLS if tool.strip()]

if "backfill" not in DISABLED_TOOLS:
    register_backfill_tools(mcp, client)
if "ee" not in DISABLED_TOOLS:
    register_ee_tools(mcp, client)
if "execution" not in DISABLED_TOOLS:
    register_execution_tools(mcp, client)
if "files" not in DISABLED_TOOLS:
    register_files_tools(mcp, client)
if "flow" not in DISABLED_TOOLS:
    register_flow_tools(mcp, client)
if "kv" not in DISABLED_TOOLS:
    register_kv_tools(mcp, client)
if "namespace" not in DISABLED_TOOLS:
    register_namespace_tools(mcp, client)
if "replay" not in DISABLED_TOOLS:
    register_replay_tools(mcp, client)
if "restart" not in DISABLED_TOOLS:
    register_restart_tools(mcp, client)
if "resume" not in DISABLED_TOOLS:
    register_resume_tools(mcp, client)

if __name__ == "__main__":
    mcp.run(transport="stdio")
