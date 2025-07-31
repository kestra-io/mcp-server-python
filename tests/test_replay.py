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
async def test_replay_execution():
    """Test replay_execution tool with successful cases."""
    async with Client(mcp_server_config) as client:
        flow_response_json = await create_flow("fail_randomly.yaml", client)
        assert flow_response_json["id"] == "fail_randomly"
        assert flow_response_json["namespace"] == "company.team"
        assert not flow_response_json["disabled"]
        assert not flow_response_json["deleted"]

        # Execute the flow first time
        first_execution = await client.call_tool(
            "execute_flow", {"namespace": "company.team", "flow_id": "fail_randomly"}
        )
        first_execution_json = json.loads(first_execution[0].text)
        first_execution_id = first_execution_json["id"]
        assert first_execution_json["flowId"] == "fail_randomly"
        assert first_execution_json["namespace"] == "company.team"

        # Execute the flow second time
        second_execution = await client.call_tool(
            "execute_flow", {"namespace": "company.team", "flow_id": "fail_randomly"}
        )
        second_execution_json = json.loads(second_execution[0].text)
        second_execution_id = second_execution_json["id"]
        assert second_execution_json["flowId"] == "fail_randomly"
        assert second_execution_json["namespace"] == "company.team"

        # Verify IDs are different
        assert first_execution_id != second_execution_id

        # Wait for executions to complete
        await poll_for_execution(
            client, first_execution_id, max_retries=10, retry_interval=1
        )
        await poll_for_execution(
            client, second_execution_id, max_retries=10, retry_interval=1
        )

        # Replay the most recent execution
        replay_result = await client.call_tool(
            "replay_execution",
            {"flow_id": "fail_randomly", "namespace": "company.team"},
        )
        replay_text = replay_result[0].text
        print(f"Replay result: {replay_text}")
        assert "Replayed execution" in replay_text
        assert "fail_randomly" in replay_text
        assert "company.team" in replay_text
        assert "Result: {'count': 1}" in replay_text

        # Replay the first execution by ID
        replay_result = await client.call_tool(
            "replay_execution", {"ids": [first_execution_id]}
        )
        replay_json = json.loads(replay_result[0].text)
        print(f"Replay by ID result: {json.dumps(replay_json, indent=2)}")
        assert replay_json["count"] == 1


@pytest.mark.asyncio
async def test_replay_execution_invalid_flow():
    """Test replay_execution tool with invalid flow."""
    async with Client(mcp_server_config) as client:
        with pytest.raises(ToolError, match="No executions found for"):
            await client.call_tool(
                "replay_execution",
                {"flow_id": "non_existent_flow", "namespace": "company.team"},
            )


@pytest.mark.asyncio
async def test_replay_execution_missing_params():
    """Test replay_execution tool with missing parameters."""
    async with Client(mcp_server_config) as client:
        with pytest.raises(
            ToolError,
            match="If ids is not provided, both namespace and flow_id are required",
        ):
            await client.call_tool(
                "replay_execution", {"flow_id": "fail_randomly"}  # Missing namespace
            )


@pytest.mark.asyncio
async def test_replay_execution_invalid_ids():
    """Test replay_execution tool with invalid execution IDs."""
    async with Client(mcp_server_config) as client:
        # Create flow first
        await create_flow("fail_randomly.yaml", client)

        # Try to replay with non-existent ID
        with pytest.raises(ToolError):  # The exact error might vary based on the API
            await client.call_tool("replay_execution", {"ids": ["non_existent_id"]})


if __name__ == "__main__":
    asyncio.run(test_replay_execution())
    asyncio.run(test_replay_execution_invalid_flow())
    asyncio.run(test_replay_execution_missing_params())
    asyncio.run(test_replay_execution_invalid_ids())
