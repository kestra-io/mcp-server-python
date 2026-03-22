"""
uv run pytest tests/test_mcp_server.py
"""

import pytest
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)


@pytest.mark.asyncio
async def test_list_tools(kestra_client):
    tools = await kestra_client.list_tools()
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
        "generate_flow",
        "generate_dashboard",
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
        # Logs tools
        "get_execution_logs",
        "download_execution_logs",
        "search_logs",
        "delete_execution_logs",
        "delete_flow_logs",
        "follow_execution_logs",
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
        "manage_tests",
        "generate_app",
        "generate_test"
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
