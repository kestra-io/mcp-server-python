"""Tests for the flow-validation tools.

The offline tests need no running Kestra and pin the request shape and response
parsing of `_validate_flow_source`. The live test exercises the registered
`validate_flow` tool against a real Kestra instance.
"""

import sys
import json
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tools.validate import _validate_flow_source  # noqa: E402


@pytest.fixture(autouse=True)
def _no_version_override(monkeypatch):
    monkeypatch.delenv("KESTRA_API_VERSION", raising=False)


VALID_YAML = (
    "id: hello\n"
    "namespace: company.team\n"
    "tasks:\n"
    "  - id: log\n"
    "    type: io.kestra.plugin.core.log.Log\n"
    "    message: hi\n"
)


@pytest.mark.asyncio
async def test_validate_flow_source_valid_returns_true():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url.path).endswith("/flows/validate")
        assert request.headers["Content-Type"].startswith("application/x-yaml")
        assert "id: hello" in request.content.decode()
        return httpx.Response(204, request=request)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://kestra/api/v1"
    )
    result = await _validate_flow_source(client, VALID_YAML)

    assert result == {"valid": True, "errors": []}


@pytest.mark.asyncio
async def test_validate_flow_source_invalid_returns_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "title": "Validation failed",
                "detail": "Invalid type: io.kestra.core.tasks.log.Log",
                "details": [
                    {"message": "Invalid type: io.kestra.core.tasks.log.Log"}
                ],
            },
            request=request,
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://kestra/api/v1"
    )
    result = await _validate_flow_source(client, VALID_YAML)

    assert result["valid"] is False
    assert result["errors"]
    assert result["message"] == "Invalid type: io.kestra.core.tasks.log.Log"


@pytest.mark.asyncio
async def test_validate_flow_tool(kestra_client):
    result = await kestra_client.call_tool("validate_flow", {"source": VALID_YAML})
    response = json.loads(result.content[0].text)
    assert response["valid"] is True

    invalid_yaml = VALID_YAML.replace("id: hello\n", "")
    result = await kestra_client.call_tool("validate_flow", {"source": invalid_yaml})
    response = json.loads(result.content[0].text)
    assert response["valid"] is False
