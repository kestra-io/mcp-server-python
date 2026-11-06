from typing import Annotated, Any

import httpx
from fastmcp import FastMCP
from pydantic import Field


async def _validate_flow_source(client: httpx.AsyncClient, source: str) -> dict:
    """Validate a single flow source and normalise the response.

    Kestra's `POST /flows/validate` answers `204` (valid) or `422` with a
    `application/problem+json` body. Everything else is an error.
    """
    headers = {"Content-Type": "application/x-yaml"}
    resp = await client.post("/flows/validate", content=source, headers=headers)

    if resp.status_code in (200, 204):
        return {"valid": True, "errors": []}

    if resp.status_code == 422:
        body = resp.json() if resp.content else {}
        return {
            "valid": False,
            "errors": _extract_errors(body),
            "message": _validation_message(body),
        }

    resp.raise_for_status()
    return {"valid": True, "errors": []}


def _extract_errors(body: Any) -> list:
    if not isinstance(body, dict):
        return []
    for key in ("details", "violations", "errors"):
        value = body.get(key)
        if isinstance(value, list):
            return value
    detail = body.get("detail")
    return [detail] if detail else []


def _validation_message(body: Any) -> str:
    if isinstance(body, dict):
        return body.get("detail") or body.get("title") or "Validation failed"
    return "Validation failed"


def register_validate_tools(mcp: FastMCP, client: httpx.AsyncClient) -> None:
    @mcp.tool()
    async def validate_flow(
        source: Annotated[
            str,
            Field(
                description="The flow YAML source to validate. Must include 'id' and 'namespace'."
            ),
        ],
    ) -> dict:
        """Validate a flow's YAML source without creating it.

        Use this before create_flow_from_yaml to catch syntax, type and
        constraint errors while the flow can still be fixed for free. Returns
        {"valid": true, "errors": []} when valid, or
        {"valid": false, "errors": [...], "message": ...} when not.
        """
        return await _validate_flow_source(client, source)

    @mcp.tool()
    async def validate_namespace_files(
        files: Annotated[
            dict,
            Field(
                description="A map of file path to flow YAML source to validate before writing to a namespace."
            ),
        ],
    ) -> dict:
        """Validate several flow YAML files at once.

        Each entry in `files` is validated independently and the result is
        reported per path, plus an overall `valid` flag.
        """
        results = {
            str(path): await _validate_flow_source(client, str(src))
            for path, src in (files or {}).items()
        }
        return {
            "valid": all(r.get("valid", False) for r in results.values()),
            "files": results,
        }
