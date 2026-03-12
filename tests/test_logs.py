import pytest
from fastmcp import Client
from dotenv import load_dotenv
import json
from pathlib import Path
from test_utils import mcp_server_config, create_flow, poll_for_execution
import time
import asyncio

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env", override=True)


@pytest.mark.asyncio
async def test_get_execution_logs():
    """Test getting logs for a specific execution."""
    async with Client(mcp_server_config) as client:
        # First create a flow and execute it to have logs to retrieve
        response_json = await create_flow("hello_mcp.yaml", client)
        flow_id = response_json.get("id", "")
        namespace = response_json.get("namespace", "")
        
        # Execute the flow
        result = await client.call_tool(
            "execute_flow",
            {"namespace": namespace, "flow_id": flow_id},
        )
        execution_response = json.loads(result.content[0].text) if result and result.content and result.content[0].text else {}
        execution_id = execution_response.get("id", "")
        
        # Wait for execution to complete
        await poll_for_execution(client, execution_id)
        
        # Test getting logs
        result = await client.call_tool(
            "get_execution_logs",
            {"execution_id": execution_id},
        )
        logs_response = json.loads(result.content[0].text) if result and result.content and result.content[0].text else {}
        
        # Verify we got logs (API returns a single log entry or list)
        if isinstance(logs_response, dict):
            # Single log entry
            log_entry = logs_response
            assert "namespace" in log_entry
            assert "flowId" in log_entry
            assert "executionId" in log_entry
            assert "timestamp" in log_entry
            assert "level" in log_entry
            assert "message" in log_entry
        elif isinstance(logs_response, list):
            # List of log entries
            assert len(logs_response) > 0
            log_entry = logs_response[0]
            assert "namespace" in log_entry
            assert "flowId" in log_entry
            assert "executionId" in log_entry
            assert "timestamp" in log_entry
            assert "level" in log_entry
            assert "message" in log_entry


@pytest.mark.asyncio
async def test_get_execution_logs_with_filters():
    """Test getting logs with filters."""
    async with Client(mcp_server_config) as client:
        # First create a flow and execute it to have logs to retrieve
        response_json = await create_flow("hello_mcp.yaml", client)
        flow_id = response_json.get("id", "")
        namespace = response_json.get("namespace", "")
        
        # Execute the flow
        result = await client.call_tool(
            "execute_flow",
            {"namespace": namespace, "flow_id": flow_id},
        )
        execution_response = json.loads(result.content[0].text) if result and result.content and result.content[0].text else {}
        execution_id = execution_response.get("id", "")
        
        # Wait for execution to complete
        await poll_for_execution(client, execution_id)
        
        # Test getting logs with level filter
        result = await client.call_tool(
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
async def test_download_execution_logs():
    """Test downloading logs as text."""
    async with Client(mcp_server_config) as client:
        # First create a flow and execute it to have logs to retrieve
        response_json = await create_flow("hello_mcp.yaml", client)
        flow_id = response_json.get("id", "")
        namespace = response_json.get("namespace", "")
        
        # Execute the flow
        result = await client.call_tool(
            "execute_flow",
            {"namespace": namespace, "flow_id": flow_id},
        )
        execution_response = json.loads(result.content[0].text) if result and result.content and result.content[0].text else {}
        execution_id = execution_response.get("id", "")
        
        # Wait for execution to complete
        await poll_for_execution(client, execution_id)
        
        # Test downloading logs
        result = await client.call_tool(
            "download_execution_logs",
            {"execution_id": execution_id},
        )
        logs_text = result.content[0].text if result and result.content and result.content[0].text else ""
        
        # Verify we got text logs
        assert isinstance(logs_text, str)


@pytest.mark.asyncio
async def test_search_logs():
    """Test searching logs across executions."""
    async with Client(mcp_server_config) as client:
        # First create a flow and execute it to have logs to search
        response_json = await create_flow("hello_mcp.yaml", client)
        flow_id = response_json.get("id", "")
        namespace = response_json.get("namespace", "")
        
        # Execute the flow
        result = await client.call_tool(
            "execute_flow",
            {"namespace": namespace, "flow_id": flow_id},
        )
        execution_response = json.loads(result.content[0].text) if result and result.content and result.content[0].text else {}
        execution_id = execution_response.get("id", "")
        
        # Wait for execution to complete
        await poll_for_execution(client, execution_id)
        
        # Test searching logs
        result = await client.call_tool(
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
        # Note: API might not return page/size in response


@pytest.mark.asyncio
async def test_log_level_validation():
    """Test that invalid log levels are rejected."""
    async with Client(mcp_server_config) as client:
        # Test with invalid log level
        try:
            result = await client.call_tool(
                "get_execution_logs",
                {
                    "execution_id": "test-execution-id",
                    "min_level": "INVALID"
                },
            )
            # If we get here, the tool should have returned an error
            assert False, "Expected error for invalid log level"
        except Exception as e:
            # This is expected - the tool should reject invalid log levels
            assert "Invalid min_level" in str(e) or "INVALID" in str(e)


@pytest.mark.asyncio
async def test_logs_tools_availability():
    """Test that all logs tools are available."""
    async with Client(mcp_server_config) as client:
        # Test that all logs tools are registered and callable
        logs_tools = [
            "get_execution_logs",
            "download_execution_logs", 
            "search_logs",
            "delete_execution_logs",
            "delete_flow_logs",
            "follow_execution_logs",
        ]
        
        # Get available tools
        tools_result = await client.list_tools()
        available_tools = [tool.name for tool in tools_result]
        
        # Verify all logs tools are available
        for tool_name in logs_tools:
            assert tool_name in available_tools, f"Tool {tool_name} not found in available tools"