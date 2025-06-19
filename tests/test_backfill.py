import pytest
from fastmcp import Client
from dotenv import load_dotenv
import json
from pathlib import Path
from test_utils import mcp_server_config, create_flow
import asyncio

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env", override=True)


@pytest.mark.asyncio
async def test_backfill():
    """Test backfill tool."""
    async with Client(mcp_server_config) as client:
        response_json = await create_flow("scheduled_flow.yaml", client)
        flow_id = response_json.get("id", "")
        namespace = response_json.get("namespace", "")
        trigger_id = response_json.get("triggers")[0].get("id", "")
        print(f"Namespace: {namespace}, Flow ID: {flow_id}, Trigger ID: {trigger_id}")
        await asyncio.sleep(1)  # # wait for the flow to be created

        result = await client.call_tool(
            "backfill_executions",
            {
                "namespace": namespace,
                "flow_id": flow_id,
                "hours": 1,
                "labels": [{"key": "backfillFrom", "value": "unitTest"}],
            },
        )
        response_json = json.loads(result[0].text)
        assert response_json["namespace"] == namespace
        assert response_json["flowId"] == flow_id
        assert "backfill" in response_json
        assert "start" in response_json["backfill"]
        assert "end" in response_json["backfill"]
        assert response_json["triggerId"] == trigger_id
        assert response_json["disabled"] is False
        assert response_json["backfill"]["paused"] is False

        await asyncio.sleep(3)  # wait for the backfill to complete
        await create_flow("scheduled_flow_disabled_trigger.yaml", client)


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_backfill())
