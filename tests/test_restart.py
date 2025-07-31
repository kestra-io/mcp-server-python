import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from dotenv import load_dotenv
from pathlib import Path
from test_utils import mcp_server_config, create_flow, poll_for_execution
import asyncio
import json


load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env", override=True)


@pytest.mark.asyncio
async def test_restart_execution():
    """Test restart_execution tool with successful cases."""
    async with Client(mcp_server_config) as client:
        flow_response_json = await create_flow("failure.yaml", client)
        assert flow_response_json["id"] == "failure"
        assert flow_response_json["namespace"] == "company.team"
        assert not flow_response_json["disabled"]
        assert not flow_response_json["deleted"]

        # Execute the flow first time
        first_execution = await client.call_tool(
            "execute_flow", {"namespace": "company.team", "flow_id": "failure"}
        )
        first_execution_json = json.loads(first_execution[0].text)
        first_execution_id = first_execution_json["id"]
        print(f"First execution response: {json.dumps(first_execution_json, indent=2)}")
        assert first_execution_json["flowId"] == "failure"
        assert first_execution_json["namespace"] == "company.team"

        # Execute the flow second time
        second_execution = await client.call_tool(
            "execute_flow", {"namespace": "company.team", "flow_id": "failure"}
        )
        second_execution_json = json.loads(second_execution[0].text)
        second_execution_id = second_execution_json["id"]
        print(
            f"Second execution response: {json.dumps(second_execution_json, indent=2)}"
        )
        assert second_execution_json["flowId"] == "failure"
        assert second_execution_json["namespace"] == "company.team"

        # Verify IDs are different
        assert first_execution_id != second_execution_id

        # Wait for executions to fail
        await poll_for_execution(
            client, first_execution_id, max_retries=10, retry_interval=1
        )
        await poll_for_execution(
            client, second_execution_id, max_retries=10, retry_interval=1
        )

        # Restart the most recent execution
        restart_result = await client.call_tool(
            "restart_execution", {"namespace": "company.team", "flow_id": "failure"}
        )
        restart_json = json.loads(restart_result[0].text)
        print(f"Restart result: {json.dumps(restart_json, indent=2)}")
        restarted_execution_id = restart_json["id"]

        # Verify the restart response structure
        assert "id" in restart_json
        assert "state" in restart_json
        assert restart_json["state"]["current"] == "RESTARTED"
        assert restart_json["flowId"] == "failure"
        assert restart_json["namespace"] == "company.team"

        # Wait for the restarted execution to fail again before attempting another restart
        await poll_for_execution(
            client, restarted_execution_id, max_retries=10, retry_interval=1
        )
        get_execution = await client.call_tool(
            "manage_executions", {"action": "get", "execution_id": first_execution_id}
        )
        get_execution_json = json.loads(get_execution[0].text)
        assert get_execution_json["state"]["current"] == "FAILED"

        # Now restart the first execution again
        restart_result = await client.call_tool(
            "restart_execution",
            {
                "namespace": "company.team",
                "flow_id": "failure",
                "execution_id": first_execution_id,
            },
        )
        restart_json = json.loads(restart_result[0].text)
        print(f"Restart first execution result: {json.dumps(restart_json, indent=2)}")
        assert "id" in restart_json
        assert "state" in restart_json
        assert restart_json["state"]["current"] == "RESTARTED"


@pytest.mark.asyncio
async def test_restart_from_failed_task():
    """Test restarting execution from a failed task."""
    async with Client(mcp_server_config) as client:
        # Create and execute flow
        await create_flow("failure.yaml", client)
        execution = await client.call_tool(
            "execute_flow", {"namespace": "company.team", "flow_id": "failure"}
        )
        execution_json = json.loads(execution[0].text)
        execution_id = execution_json["id"]
        print(f"Execution response: {json.dumps(execution_json, indent=2)}")

        # Wait for execution to fail
        await poll_for_execution(client, execution_id, max_retries=10, retry_interval=1)

        # Restart from failed task
        restart_result = await client.call_tool(
            "change_taskrun_state", {"execution_id": execution_id}
        )
        restart_json = json.loads(restart_result[0].text)
        print(f"Restart from failed task result: {json.dumps(restart_json, indent=2)}")
        assert restart_json["id"] == execution_id  # Same execution, different state


@pytest.mark.asyncio
async def test_change_taskrun_state():
    """Test changing task state to WARNING."""
    async with Client(mcp_server_config) as client:
        # Create and execute flow
        await create_flow("failure.yaml", client)
        execution = await client.call_tool(
            "execute_flow", {"namespace": "company.team", "flow_id": "failure"}
        )
        execution_json = json.loads(execution[0].text)
        execution_id = execution_json["id"]
        print(f"Execution response: {json.dumps(execution_json, indent=2)}")

        # Wait for execution to fail
        await poll_for_execution(client, execution_id, max_retries=10, retry_interval=1)

        # Change task state to WARNING
        change_result = await client.call_tool(
            "change_taskrun_state", {"execution_id": execution_id, "state": "WARNING"}
        )
        change_json = json.loads(change_result[0].text)
        print(f"Change task state result: {json.dumps(change_json, indent=2)}")
        assert change_json["id"] == execution_id


@pytest.mark.asyncio
async def test_restart_invalid_execution():
    """Test restart_execution tool with invalid execution."""
    async with Client(mcp_server_config) as client:
        with pytest.raises(ToolError):
            await client.call_tool(
                "restart_execution",
                {
                    "namespace": "company.team",
                    "flow_id": "failure",
                    "execution_id": "invalid_id",
                },
            )


@pytest.mark.asyncio
async def test_restart_missing_params():
    """Test restart_execution tool with missing parameters."""
    async with Client(mcp_server_config) as client:
        with pytest.raises(ToolError):
            await client.call_tool(
                "restart_execution", {"flow_id": "failure"}  # Missing namespace
            )


if __name__ == "__main__":
    asyncio.run(test_restart_execution())
    asyncio.run(test_restart_from_failed_task())
    asyncio.run(test_change_taskrun_state())
    asyncio.run(test_restart_invalid_execution())
    asyncio.run(test_restart_missing_params())
