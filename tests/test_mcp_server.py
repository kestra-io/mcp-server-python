"""
uv run pytest src/tests/test_mcp_server.py

config = {
  "mcpServers": {
    "kestra": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-e",
        "KESTRA_MCP_DISABLED_TOOLS",
        "-e",
        "KESTRA_BASE_URL",
        "-e",
        "KESTRA_API_TOKEN",
        "-e",
        "KESTRA_TENANT_ID",
        "-e",
        "-e",
        "GOOGLE_API_KEY",
        "-e",
        "HELICONE_API_KEY",
        "ghcr.io/kestra-io/mcp-server-python:latest"
      ],
      "env": {
        "KESTRA_MCP_DISABLED_TOOLS": "",
        "KESTRA_BASE_URL": "http://host.docker.internal:38080/api/v1",
        "KESTRA_API_TOKEN": os.getenv("KESTRA_API_TOKEN"),
        "KESTRA_TENANT_ID": os.getenv("KESTRA_TENANT_ID")
      }
    }
  }
}
"""

import pytest
from fastmcp import Client
from dotenv import load_dotenv
from pathlib import Path
from test_utils import mcp_server_config
import os

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)


@pytest.mark.asyncio
async def test_list_tools():
    async with Client(mcp_server_config) as client:
        tools = await client.list_tools()
        tool_names = [tool.name for tool in tools]

        # Base tools that are always available
        base_tools = {
            "backfill_executions",
            "execute_flow",
            "add_execution_labels",
            "list_executions",
            "namespace_file_action",
            "namespace_directory_action",
            "search_flows",
            "list_flows_with_triggers",
            "create_flow_from_yaml",
            "manage_flow",
            "manage_kv_store",
            "list_namespaces",
            "list_flows_in_namespace",
            "list_namespace_dependencies",
            "replay_execution",
            "restart_execution",
            "change_taskrun_state",
            "resume_execution",
            "force_run_execution",
            "manage_executions",
        }

        # EE-specific tools
        ee_tools = {
            "get_instance_info",
            "invite_user",
            "search_apps",
            "manage_announcements",
            "manage_apps",
            "manage_group",
            "manage_invitations",
            "manage_maintenance_mode",
            "manage_tests",
            "manage_worker_groups",
            "search_users",
        }

        # Check if EE tools are disabled
        disabled_tools = os.getenv("KESTRA_MCP_DISABLED_TOOLS", "").split(",")
        disabled_tools = [tool.strip() for tool in disabled_tools if tool.strip()]

        if "ee" in disabled_tools:
            expected_tools = base_tools
        else:
            expected_tools = base_tools | ee_tools

        actual_tools = set(tool_names)
        print(f"Actual tools: {actual_tools}")
        print(f"Expected tools: {expected_tools}")
        print(f"Disabled tools: {disabled_tools}")

        assert (
            actual_tools == expected_tools
        ), f"Missing tools: {expected_tools - actual_tools}, Extra tools: {actual_tools - expected_tools}"


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_list_tools())
