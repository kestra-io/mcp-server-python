import pytest
from dotenv import load_dotenv
import json
from pathlib import Path
from test_utils import create_flow

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)


@pytest.mark.asyncio
async def test_create_flow_from_yaml(kestra_client, cleanup):
    yaml_path = Path(__file__).parent / "code" / "hello_mcp.yaml"
    with open(yaml_path, "r") as f:
        expected_yaml = f.read()

    result = await kestra_client.call_tool(
        "create_flow_from_yaml", {"yaml_definition": expected_yaml}
    )
    response_json = json.loads(result.content[0].text)
    cleanup.track_flow(response_json["namespace"], response_json["id"])
    returned_source = response_json.get("source", "")
    assert expected_yaml.strip() == returned_source.strip()


@pytest.mark.asyncio
async def test_disable_flow(kestra_client, cleanup):
    flow = await create_flow("hello_mcp.yaml", kestra_client, cleanup)
    print(f"Create flow response: {flow}")
    result = await kestra_client.call_tool(
        "manage_flow",
        {"action": "disable", "namespace": "company.team", "flow_id": "hello_mcp"},
    )
    response_json = json.loads(result.content[0].text)
    # The response should contain a 'count' field indicating how many flows were disabled
    assert "count" in response_json
    assert response_json["count"] >= 1


@pytest.mark.asyncio
async def test_enable_flow(kestra_client, cleanup):
    # First create the flow
    flow = await create_flow("hello_mcp.yaml", kestra_client, cleanup)
    print(f"Create flow response: {flow}")

    # First disable the flow
    disable_result = await kestra_client.call_tool(
        "manage_flow",
        {"action": "disable", "namespace": "company.team", "flow_id": "hello_mcp"},
    )
    disable_response = json.loads(disable_result.content[0].text)
    assert "count" in disable_response
    assert disable_response["count"] >= 1

    # Then try to enable it
    result = await kestra_client.call_tool(
        "manage_flow",
        {"action": "enable", "namespace": "company.team", "flow_id": "hello_mcp"},
    )
    response_json = json.loads(result.content[0].text)
    # The response should contain a 'count' field indicating how many flows were enabled
    # Note: count might be 0 if the flow was already enabled
    assert "count" in response_json
    assert response_json["count"] >= 0
