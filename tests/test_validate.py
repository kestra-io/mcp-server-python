"""Tests for the flow-validation tools.

The offline tests need no running Kestra and pin the request shape and response
parsing of `_validate_flow_source`, using bodies captured from Kestra 2.0 (errors
only in `constraints`) and from a newer build (`errors` entries as well). The
live tests exercise the registered tools against a real Kestra instance.
"""

import json
import sys
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

INVALID_TYPE_YAML = (
    "id: hello\n"
    "namespace: company.team\n"
    "inputs:\n"
    "  - id: flag\n"
    "    type: NOT_A_TYPE\n"
    "tasks:\n"
    "  - id: log\n"
    "    type: io.kestra.plugin.core.log.Log\n"
    "    message: hi\n"
)

SYNTAX_ERROR_YAML = VALID_YAML.replace("    message: hi\n", "   message: [unclosed\n")

UNKNOWN_TASK_YAML = VALID_YAML.replace(
    "io.kestra.plugin.core.log.Log", "io.kestra.plugin.core.log.DoesNotExist"
).replace("id: hello\n", "id: hello2\n")


def _client(status: int, body) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url.path).endswith("/flows/validate")
        assert request.headers["Content-Type"].startswith("application/x-yaml")
        return httpx.Response(status, json=body, request=request)

    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://kestra/api/v1"
    )


@pytest.mark.asyncio
async def test_validate_flow_source_valid_returns_true():
    body = [
        {
            "index": 0,
            "namespace": "company.team",
            "flow": "hello",
            "outdated": False,
            "deprecationPaths": [],
            "warnings": [],
            "infos": [],
        }
    ]
    result = await _validate_flow_source(_client(200, body), VALID_YAML)

    assert result == {
        "valid": True,
        "documents": [
            {
                "index": 0,
                "namespace": "company.team",
                "flow": "hello",
                "valid": True,
                "errors": [],
            }
        ],
    }


@pytest.mark.asyncio
async def test_validate_flow_source_200_with_constraints_is_invalid():
    # Kestra 2.0 answers 200 and reports the error only in `constraints`.
    body = [
        {
            "index": 0,
            "constraints": "Validation error: Invalid type: NOT_A_TYPE",
            "outdated": False,
        }
    ]
    result = await _validate_flow_source(_client(200, body), INVALID_TYPE_YAML)

    assert result["valid"] is False
    doc = result["documents"][0]
    assert doc["valid"] is False
    assert doc["constraints"] == "Validation error: Invalid type: NOT_A_TYPE"
    assert doc["errors"] == [{"detail": "Validation error: Invalid type: NOT_A_TYPE"}]


@pytest.mark.asyncio
async def test_validate_flow_source_200_with_errors_is_invalid():
    # Newer Kestra versions add located `errors` entries next to `constraints`.
    error = {
        "detail": "Invalid type: NOT_A_TYPE",
        "pointer": "/inputs/0/type",
        "path": "inputs[0].type",
    }
    body = [
        {
            "index": 0,
            "namespace": "company.team",
            "flow": "hello",
            "outdated": False,
            "deprecationPaths": [],
            "warnings": [],
            "infos": [],
            "errors": [error],
            "constraints": "inputs[0].type: Invalid type: NOT_A_TYPE",
        }
    ]
    result = await _validate_flow_source(_client(200, body), INVALID_TYPE_YAML)

    assert result["valid"] is False
    assert result["documents"][0]["errors"] == [error]


@pytest.mark.asyncio
async def test_validate_flow_source_keeps_non_fatal_notices():
    body = [
        {
            "index": 0,
            "namespace": "company.team",
            "flow": "hello",
            "outdated": False,
            "deprecationPaths": ["tasks[0].delay"],
            "warnings": ["The subflow 'other' not found in namespace 'company.team'."],
            "infos": [],
        }
    ]
    result = await _validate_flow_source(_client(200, body), VALID_YAML)

    assert result["valid"] is True
    doc = result["documents"][0]
    assert doc["deprecationPaths"] == ["tasks[0].delay"]
    assert doc["warnings"] == [
        "The subflow 'other' not found in namespace 'company.team'."
    ]
    assert "infos" not in doc


@pytest.mark.asyncio
async def test_validate_flow_source_unexpected_status_raises():
    with pytest.raises(httpx.HTTPStatusError):
        await _validate_flow_source(_client(204, None), VALID_YAML)


async def _call(kestra_client, tool: str, args: dict) -> dict:
    result = await kestra_client.call_tool(tool, args)
    return json.loads(result.content[0].text)


@pytest.mark.asyncio
async def test_validate_flow_tool(kestra_client):
    response = await _call(kestra_client, "validate_flow", {"source": VALID_YAML})
    assert response["valid"] is True
    assert response["documents"][0]["errors"] == []

    invalid_yaml = VALID_YAML.replace("id: hello\n", "")
    response = await _call(kestra_client, "validate_flow", {"source": invalid_yaml})
    assert response["valid"] is False


@pytest.mark.asyncio
async def test_validate_flow_tool_invalid_type(kestra_client):
    response = await _call(
        kestra_client, "validate_flow", {"source": INVALID_TYPE_YAML}
    )

    assert response["valid"] is False
    doc = response["documents"][0]
    assert doc["valid"] is False
    assert "Invalid type: NOT_A_TYPE" in doc["constraints"]
    assert doc["errors"]


@pytest.mark.asyncio
async def test_validate_flow_tool_syntax_error(kestra_client):
    response = await _call(
        kestra_client, "validate_flow", {"source": SYNTAX_ERROR_YAML}
    )

    assert response["valid"] is False
    assert "YAML parsing error" in response["documents"][0]["constraints"]


@pytest.mark.asyncio
async def test_validate_flow_tool_multi_document(kestra_client):
    source = VALID_YAML + "---\n" + UNKNOWN_TASK_YAML
    response = await _call(kestra_client, "validate_flow", {"source": source})

    assert response["valid"] is False
    first, second = response["documents"]
    assert (first["index"], first["valid"]) == (0, True)
    assert (second["index"], second["valid"]) == (1, False)
    assert "io.kestra.plugin.core.log.DoesNotExist" in second["constraints"]


@pytest.mark.asyncio
async def test_validate_namespace_files_tool(kestra_client):
    response = await _call(
        kestra_client,
        "validate_namespace_files",
        {"files": {"good.yaml": VALID_YAML, "bad.yaml": INVALID_TYPE_YAML}},
    )

    assert response["valid"] is False
    assert response["files"]["good.yaml"]["valid"] is True
    assert response["files"]["bad.yaml"]["valid"] is False
