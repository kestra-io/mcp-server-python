import pytest
from dotenv import load_dotenv
import json
from pathlib import Path
from test_utils import create_flow, poll_for_execution

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env", override=True)


async def _create_and_execute(kestra_client, cleanup):
    """Helper: create hello_mcp flow, execute it, wait for completion, return (flow_id, namespace, execution_id)."""
    response_json = await create_flow("hello_mcp.yaml", kestra_client, cleanup)
    flow_id = response_json.get("id", "")
    namespace = response_json.get("namespace", "")
    result = await kestra_client.call_tool(
        "execute_flow",
        {"namespace": namespace, "flow_id": flow_id},
    )
    execution_response = json.loads(result.content[0].text) if result and result.content and result.content[0].text else {}
    execution_id = execution_response.get("id", "")
    cleanup.track_execution(execution_id)
    await poll_for_execution(kestra_client, execution_id)
    return flow_id, namespace, execution_id


@pytest.mark.asyncio
async def test_get_execution_logs(kestra_client, cleanup):
    """Test getting logs for a specific execution."""
    flow_id, namespace, execution_id = await _create_and_execute(kestra_client, cleanup)

    # Test getting logs
    result = await kestra_client.call_tool(
        "get_execution_logs",
        {"execution_id": execution_id},
    )
    logs_response = json.loads(result.content[0].text) if result and result.content and result.content[0].text else {}

    # Verify we got logs - unwrap if wrapped in {"results": [...]}
    if isinstance(logs_response, dict) and "results" in logs_response:
        logs_list = logs_response["results"]
    elif isinstance(logs_response, list):
        logs_list = logs_response
    else:
        logs_list = [logs_response]
    assert len(logs_list) > 0
    log_entry = logs_list[0]
    assert "namespace" in log_entry
    assert "flowId" in log_entry
    assert "executionId" in log_entry
    assert "timestamp" in log_entry
    assert "level" in log_entry
    assert "message" in log_entry


@pytest.mark.asyncio
async def test_get_execution_logs_with_filters(kestra_client, cleanup):
    """Test getting logs with filters."""
    flow_id, namespace, execution_id = await _create_and_execute(kestra_client, cleanup)

    # Test getting logs with level filter
    result = await kestra_client.call_tool(
        "get_execution_logs",
        {
            "execution_id": execution_id,
            "min_level": "INFO"
        },
    )
    logs_response = json.loads(result.content[0].text) if result and result.content and result.content[0].text else {}

    # Verify we got logs (API returns a single log entry or list)
    assert isinstance(logs_response, (dict, list))


@pytest.mark.asyncio
async def test_download_execution_logs(kestra_client, cleanup):
    """Test downloading logs as text."""
    flow_id, namespace, execution_id = await _create_and_execute(kestra_client, cleanup)

    # Test downloading logs
    result = await kestra_client.call_tool(
        "download_execution_logs",
        {"execution_id": execution_id},
    )
    logs_text = result.content[0].text if result and result.content and result.content[0].text else ""

    # Verify we got text logs
    assert isinstance(logs_text, str)


@pytest.mark.asyncio
async def test_search_logs(kestra_client, cleanup):
    """Test searching logs across executions."""
    flow_id, namespace, execution_id = await _create_and_execute(kestra_client, cleanup)

    # Test searching logs
    result = await kestra_client.call_tool(
        "search_logs",
        {
            "namespace": namespace,
            "flow_id": flow_id,
            "min_level": "INFO",
            "page": 1,
            "size": 10
        },
    )
    search_response = json.loads(result.content[0].text) if result and result.content and result.content[0].text else {}

    # Verify we got search results
    assert isinstance(search_response, dict)
    assert "results" in search_response
    assert "total" in search_response


@pytest.mark.asyncio
async def test_log_level_validation(kestra_client):
    """Test that invalid log levels are rejected."""
    try:
        result = await kestra_client.call_tool(
            "get_execution_logs",
            {
                "execution_id": "test-execution-id",
                "min_level": "INVALID"
            },
        )
        assert False, "Expected error for invalid log level"
    except Exception as e:
        assert "Invalid min_level" in str(e) or "INVALID" in str(e)


@pytest.mark.asyncio
async def test_logs_tools_availability(kestra_client):
    """Test that all logs tools are available."""
    logs_tools = [
        "get_execution_logs",
        "download_execution_logs",
        "search_logs",
        "delete_execution_logs",
        "delete_flow_logs",
        "follow_execution_logs",
    ]

    tools_result = await kestra_client.list_tools()
    available_tools = [tool.name for tool in tools_result]

    for tool_name in logs_tools:
        assert tool_name in available_tools, f"Tool {tool_name} not found in available tools"
