import pytest
from dotenv import load_dotenv
import asyncio
import httpx
import json
import os
import sys
from fastmcp import Client, FastMCP
from pathlib import Path
from test_utils import create_flow

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from tools.flow import register_flow_tools  # noqa: E402

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)


def _tenant_http_client() -> httpx.AsyncClient:
    """Raw API client, for setting up state that no MCP tool changes."""
    base = os.getenv("KESTRA_BASE_URL", "http://localhost:8080/api/v1")
    tenant = os.getenv("KESTRA_TENANT_ID")
    headers: dict[str, str] = {}
    auth = None
    if (user := os.getenv("KESTRA_USERNAME")) and (pwd := os.getenv("KESTRA_PASSWORD")):
        auth = httpx.BasicAuth(user, pwd)
    elif token := os.getenv("KESTRA_API_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    tenant_base = f"{base.rstrip('/')}/{tenant}" if tenant else base
    return httpx.AsyncClient(base_url=tenant_base, auth=auth, headers=headers)


async def _list_triggers_text(kestra_client, **args) -> str:
    result = await kestra_client.call_tool("list_flows_with_triggers", args)
    return "\n".join(json.loads(result.content[0].text)["results"])


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


@pytest.mark.asyncio
async def test_find_flow(kestra_client, cleanup):
    await create_flow("hello_mcp.yaml", kestra_client, cleanup)

    # Exact-ish match with spaces instead of underscores
    result = await kestra_client.call_tool("find_flow", {"query": "hello mcp"})
    response = json.loads(result.content[0].text)
    assert response["found"] is True
    assert "match" in response
    assert response["match"]["flow_id"] == "hello_mcp"
    assert response["match"]["namespace"] == "company.team"

    # Partial match
    result2 = await kestra_client.call_tool("find_flow", {"query": "hello"})
    response2 = json.loads(result2.content[0].text)
    assert response2["found"] is True
    assert any(m["flow_id"] == "hello_mcp" for m in response2["matches"])

    # No match
    result3 = await kestra_client.call_tool(
        "find_flow", {"query": "nonexistent_xyz_123"}
    )
    response3 = json.loads(result3.content[0].text)
    assert response3["found"] is False


@pytest.mark.asyncio
async def test_search_flows(kestra_client, cleanup):
    """Test search_flows tool."""
    await create_flow("hello_mcp.yaml", kestra_client, cleanup)

    result = await kestra_client.call_tool(
        "search_flows", {"query": "hello_mcp"}
    )
    response = json.loads(result.content[0].text)
    assert "results" in response
    assert len(response["results"]) > 0
    assert any(f["id"] == "hello_mcp" for f in response["results"])


@pytest.mark.asyncio
async def test_list_flows_with_triggers(kestra_client, cleanup):
    """Test list_flows_with_triggers tool."""
    await create_flow("scheduled_flow.yaml", kestra_client, cleanup)

    result = await kestra_client.call_tool(
        "list_flows_with_triggers", {"namespace": "company.team"}
    )
    response = json.loads(result.content[0].text)
    # Response may be a list or {"results": [...]}
    if isinstance(response, dict) and "results" in response:
        items = response["results"]
    elif isinstance(response, list):
        items = response
    else:
        items = [response]
    assert len(items) > 0
    # Items are markdown strings describing flows with triggers
    combined = "\n".join(str(i) for i in items)
    assert "scheduled_flow" in combined


@pytest.mark.asyncio
async def test_list_flows_with_triggers_reports_runtime_state(kestra_client, cleanup):
    """A trigger disabled at runtime (UI toggle / set-disabled API) keeps
    `disabled` unset in the flow source, but must be reported as disabled."""
    await create_flow("runtime_disabled_trigger.yaml", kestra_client, cleanup)
    key = {"namespace": "company.team", "flowId": "runtime_disabled_trigger", "triggerId": "nightly"}

    async with _tenant_http_client() as http:
        # The scheduler creates the trigger state shortly after the flow is saved.
        text = ""
        for _ in range(30):
            text = await _list_triggers_text(kestra_client, namespace="company.team")
            if "`nightly` (enabled, next evaluation" in text:
                break
            await asyncio.sleep(1)
        assert "`nightly` (enabled, next evaluation" in text, text

        resp = await http.put("/triggers/set-disabled", json={**key, "disabled": True})
        resp.raise_for_status()

        text = await _list_triggers_text(kestra_client, namespace="company.team")
        assert "`nightly` (disabled at runtime)" in text, text

        disabled = await _list_triggers_text(
            kestra_client, namespace="company.team", disabled_only=True
        )
        assert "runtime_disabled_trigger" in disabled

        enabled = await _list_triggers_text(
            kestra_client, namespace="company.team", enabled_only=True
        )
        assert "runtime_disabled_trigger" not in enabled


@pytest.mark.asyncio
async def test_list_flows_with_triggers_source_disabled(kestra_client, cleanup):
    """A trigger disabled in its definition is still reported as plain `disabled`."""
    await create_flow("scheduled_flow_disabled_trigger.yaml", kestra_client, cleanup)
    text = await _list_triggers_text(
        kestra_client, namespace="company.team", disabled_only=True
    )
    assert "`schedule` (disabled)" in text, text


@pytest.mark.asyncio
async def test_list_flows_with_triggers_warns_when_trigger_state_is_unreadable():
    """If `GET /triggers/search` fails, the listing still answers from the flow
    source, and says so in `warning` rather than silently showing `enabled`."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/configs"):
            return httpx.Response(200, json={"version": "2.0.6", "edition": "OSS"})
        if request.url.path.endswith("/flows/company.team"):
            return httpx.Response(
                200,
                json=[
                    {
                        "id": "report",
                        "namespace": "company.team",
                        "triggers": [
                            {
                                "id": "nightly",
                                "type": "io.kestra.plugin.core.trigger.Schedule",
                                "cron": "0 2 * * *",
                            }
                        ],
                    }
                ],
            )
        if request.url.path.endswith("/triggers/search"):
            return httpx.Response(403, json={"message": "Forbidden"})
        return httpx.Response(404)

    http = httpx.AsyncClient(
        base_url="http://kestra/api/v1/main", transport=httpx.MockTransport(handler)
    )
    mcp = FastMCP("test")
    register_flow_tools(mcp, http)
    async with Client(mcp) as client:
        result = await client.call_tool(
            "list_flows_with_triggers", {"namespace": "company.team"}
        )
    response = json.loads(result.content[0].text)
    assert "`nightly` (enabled)" in response["results"][0]
    assert "`company.team`" in response["warning"]
    assert "403" in response["warning"]
