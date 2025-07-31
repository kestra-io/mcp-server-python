import pytest
from fastmcp import Client
from dotenv import load_dotenv
import json
from pathlib import Path
from test_utils import mcp_server_config, create_flow

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)


@pytest.mark.asyncio
async def test_create_flow_from_yaml():
    async with Client(mcp_server_config) as client:
        yaml_path = Path(__file__).parent / "code" / "hello_mcp.yaml"
        with open(yaml_path, "r") as f:
            expected_yaml = f.read()

        result = await client.call_tool(
            "create_flow_from_yaml", {"yaml_definition": expected_yaml}
        )
        response_json = json.loads(result[0].text)
        returned_source = response_json.get("source", "")
        assert expected_yaml.strip() == returned_source.strip()


@pytest.mark.asyncio
async def test_disable_flow():
    async with Client(mcp_server_config) as client:
        flow = await create_flow("hello_mcp.yaml", client)
        print(f"Create flow response: {flow}")
        result = await client.call_tool(
            "manage_flow",
            {"action": "disable", "namespace": "company.team", "flow_id": "hello_mcp"},
        )
        response_json = json.loads(result[0].text)
        # The response should contain a 'count' field indicating how many flows were disabled
        assert "count" in response_json
        assert response_json["count"] >= 1


@pytest.mark.asyncio
async def test_enable_flow():
    async with Client(mcp_server_config) as client:
        # First create the flow
        flow = await create_flow("hello_mcp.yaml", client)
        print(f"Create flow response: {flow}")

        # First disable the flow
        disable_result = await client.call_tool(
            "manage_flow",
            {"action": "disable", "namespace": "company.team", "flow_id": "hello_mcp"},
        )
        disable_response = json.loads(disable_result[0].text)
        assert "count" in disable_response
        assert disable_response["count"] >= 1

        # Then try to enable it
        result = await client.call_tool(
            "manage_flow",
            {"action": "enable", "namespace": "company.team", "flow_id": "hello_mcp"},
        )
        response_json = json.loads(result[0].text)
        # The response should contain a 'count' field indicating how many flows were enabled
        assert "count" in response_json
        assert response_json["count"] >= 1


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_create_flow_from_yaml())
    asyncio.run(test_disable_flow())
    asyncio.run(test_enable_flow())
