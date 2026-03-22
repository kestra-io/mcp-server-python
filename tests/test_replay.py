import pytest
from fastmcp.exceptions import ToolError
from dotenv import load_dotenv
from pathlib import Path
from test_utils import create_flow, poll_for_execution
import json

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env", override=True)


@pytest.mark.asyncio
async def test_replay_execution(kestra_client, cleanup):
    """Test replay_execution tool with successful cases."""
    flow_response_json = await create_flow("fail_randomly.yaml", kestra_client, cleanup)
    assert flow_response_json["id"] == "fail_randomly"
    assert flow_response_json["namespace"] == "company.team"
    assert not flow_response_json["disabled"]
    assert not flow_response_json["deleted"]

    # Execute the flow first time
    first_execution = await kestra_client.call_tool(
        "execute_flow", {"namespace": "company.team", "flow_id": "fail_randomly"}
    )
    first_execution_json = json.loads(first_execution.content[0].text)
    first_execution_id = first_execution_json["id"]
    cleanup.track_execution(first_execution_id)
    assert first_execution_json["flowId"] == "fail_randomly"
    assert first_execution_json["namespace"] == "company.team"

    # Execute the flow second time
    second_execution = await kestra_client.call_tool(
        "execute_flow", {"namespace": "company.team", "flow_id": "fail_randomly"}
    )
    second_execution_json = json.loads(second_execution.content[0].text)
    second_execution_id = second_execution_json["id"]
    cleanup.track_execution(second_execution_id)
    assert second_execution_json["flowId"] == "fail_randomly"
    assert second_execution_json["namespace"] == "company.team"

    # Verify IDs are different
    assert first_execution_id != second_execution_id

    # Wait for executions to complete
    await poll_for_execution(
        kestra_client, first_execution_id, max_retries=10, retry_interval=1
    )
    await poll_for_execution(
        kestra_client, second_execution_id, max_retries=10, retry_interval=1
    )

    # Replay the most recent execution
    replay_result = await kestra_client.call_tool(
        "replay_execution",
        {"flow_id": "fail_randomly", "namespace": "company.team"},
    )
    replay_json = json.loads(replay_result.content[0].text)
    print(f"Replay result: {replay_json}")
    assert "Replayed execution" in replay_json["message"]
    assert "fail_randomly" in replay_json["message"]
    assert "company.team" in replay_json["message"]
    assert replay_json["result"]["count"] == 1

    # Replay the first execution by ID
    replay_result = await kestra_client.call_tool(
        "replay_execution", {"ids": [first_execution_id]}
    )
    replay_json = json.loads(replay_result.content[0].text)
    print(f"Replay by ID result: {json.dumps(replay_json, indent=2)}")
    assert replay_json["count"] == 1


@pytest.mark.asyncio
async def test_replay_execution_invalid_flow(kestra_client):
    """Test replay_execution tool with invalid flow."""
    with pytest.raises(ToolError, match="No executions found for"):
        await kestra_client.call_tool(
            "replay_execution",
            {"flow_id": "non_existent_flow", "namespace": "company.team"},
        )


@pytest.mark.asyncio
async def test_replay_execution_missing_params(kestra_client):
    """Test replay_execution tool with missing parameters."""
    with pytest.raises(
        ToolError,
        match="If ids is not provided, both namespace and flow_id are required",
    ):
        await kestra_client.call_tool(
            "replay_execution", {"flow_id": "fail_randomly"}  # Missing namespace
        )


@pytest.mark.asyncio
async def test_replay_execution_invalid_ids(kestra_client, cleanup):
    """Test replay_execution tool with invalid execution IDs."""
    await create_flow("fail_randomly.yaml", kestra_client, cleanup)

    # Try to replay with non-existent ID
    with pytest.raises(ToolError):
        await kestra_client.call_tool("replay_execution", {"ids": ["non_existent_id"]})
