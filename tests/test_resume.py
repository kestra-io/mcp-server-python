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
async def test_resume_flow_with_pause_task():
    """Test resume functionality with pause demo flow."""
    async with Client(mcp_server_config) as client:
        # Create flow
        flow_response_json = await create_flow("pause_demo.yaml", client)
        assert flow_response_json["id"] == "pause_demo"
        assert flow_response_json["namespace"] == "company.team"

        # Execute the flow first time
        first_execution = await client.call_tool(
            "execute_flow", {"namespace": "company.team", "flow_id": "pause_demo"}
        )
        first_execution_json = json.loads(first_execution[0].text)
        first_execution_id = first_execution_json["id"]
        print(f"First execution response: {json.dumps(first_execution_json, indent=2)}")

        # Execute the flow second time
        second_execution = await client.call_tool(
            "execute_flow", {"namespace": "company.team", "flow_id": "pause_demo"}
        )
        second_execution_json = json.loads(second_execution[0].text)
        second_execution_id = second_execution_json["id"]
        print(
            f"Second execution response: {json.dumps(second_execution_json, indent=2)}"
        )

        await poll_for_execution(
            client,
            first_execution_id,
            desired_state="PAUSED",
            max_retries=10,
            retry_interval=1,
        )
        await poll_for_execution(
            client,
            second_execution_id,
            desired_state="PAUSED",
            max_retries=10,
            retry_interval=1,
        )

        # Resume the first execution
        resume_result = await client.call_tool(
            "resume_execution", {"execution_ids": [first_execution_id]}
        )
        resume_json = json.loads(resume_result[0].text)
        print(f"Resume first execution result: {json.dumps(resume_json, indent=2)}")
        assert "count" in resume_json

        # Resume the most recent paused execution
        resume_result = await client.call_tool(
            "resume_execution", {"namespace": "company.team", "flow_id": "pause_demo"}
        )
        resume_json = json.loads(resume_result[0].text)
        print(f"Resume latest execution result: {json.dumps(resume_json, indent=2)}")
        assert isinstance(resume_json, dict)


@pytest.mark.asyncio
async def test_resume_flow_with_on_resume_inputs():
    """Test resume functionality with approval flow."""
    async with Client(mcp_server_config) as client:
        # Create flow
        flow_response_json = await create_flow("approval.yaml", client)
        assert flow_response_json["id"] == "approval"
        assert flow_response_json["namespace"] == "company.team"

        # Execute the flow first time
        first_execution = await client.call_tool(
            "execute_flow", {"namespace": "company.team", "flow_id": "approval"}
        )
        first_execution_json = json.loads(first_execution[0].text)
        first_execution_id = first_execution_json["id"]
        print(f"First execution response: {json.dumps(first_execution_json, indent=2)}")

        # Execute the flow second time
        second_execution = await client.call_tool(
            "execute_flow", {"namespace": "company.team", "flow_id": "approval"}
        )
        second_execution_json = json.loads(second_execution[0].text)
        second_execution_id = second_execution_json["id"]
        print(
            f"Second execution response: {json.dumps(second_execution_json, indent=2)}"
        )

        # Wait for executions to pause
        await poll_for_execution(
            client,
            first_execution_id,
            desired_state="PAUSED",
            max_retries=10,
            retry_interval=1,
        )
        await poll_for_execution(
            client,
            second_execution_id,
            desired_state="PAUSED",
            max_retries=10,
            retry_interval=1,
        )

        # Resume the latest execution with default onResume inputs
        resume_result = await client.call_tool(
            "resume_execution", {"namespace": "company.team", "flow_id": "approval"}
        )
        resume_json = json.loads(resume_result[0].text)
        print(f"Resume latest execution result: {json.dumps(resume_json, indent=2)}")
        assert isinstance(resume_json, dict)

        # Resume the first execution with custom onResume inputs
        resume_result = await client.call_tool(
            "resume_execution",
            {
                "execution_ids": [first_execution_id],
                "on_resume": {"reason": "no more PTO available"},
            },
        )
        resume_json = json.loads(resume_result[0].text)
        print(f"Resume with custom reason result: {json.dumps(resume_json, indent=2)}")
        assert isinstance(resume_json, dict)


@pytest.mark.asyncio
async def test_resume_invalid_execution():
    """Test resume_execution tool with invalid execution id."""
    async with Client(mcp_server_config) as client:
        with pytest.raises(ToolError):
            await client.call_tool(
                "resume_execution", {"execution_ids": ["invalid_id"]}
            )


@pytest.mark.asyncio
async def test_resume_missing_params():
    """Test resume_execution tool with missing parameters."""
    async with Client(mcp_server_config) as client:
        with pytest.raises(ToolError):
            await client.call_tool(
                "resume_execution", {"flow_id": "approval"}  # Missing namespace
            )


if __name__ == "__main__":
    asyncio.run(test_resume_flow_with_pause_task())
    asyncio.run(test_resume_flow_with_on_resume_inputs())
    asyncio.run(test_resume_invalid_execution())
    asyncio.run(test_resume_missing_params())
