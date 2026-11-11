from typing import Annotated, Any

import httpx
from fastmcp import FastMCP
from pydantic import Field

# Non-fatal findings Kestra reports next to a document; surfaced only when set.
_NOTICE_KEYS = ("warnings", "deprecationPaths", "infos")


async def _validate_flow_source(client: httpx.AsyncClient, source: str) -> dict:
    """Validate a flow source and normalise the response.

    Kestra's `POST /flows/validate` answers `200` with one entry per YAML
    document, valid or not. A document is invalid when its `constraints` text
    or, on newer Kestra versions, its `errors` list is non-empty. Any other
    status is an error.
    """
    headers = {"Content-Type": "application/x-yaml"}
    resp = await client.post("/flows/validate", content=source, headers=headers)
    resp.raise_for_status()
    if resp.status_code != 200:
        raise httpx.HTTPStatusError(
            f"Unexpected status {resp.status_code} from {resp.request.url}",
            request=resp.request,
            response=resp,
        )

    documents = [_document_result(entry) for entry in resp.json()]
    result: dict[str, Any] = {
        "valid": bool(documents) and all(doc["valid"] for doc in documents),
        "documents": documents,
    }
    if not documents:
        result["message"] = "No flow found in the source."
    return result


def _document_result(entry: dict) -> dict:
    constraints = (entry.get("constraints") or "").strip()
    errors = entry.get("errors") or []
    if constraints and not errors:
        # Older Kestra versions report errors only as the `constraints` text.
        errors = [{"detail": constraints}]

    doc: dict[str, Any] = {"index": entry.get("index")}
    for key in ("namespace", "flow"):
        if entry.get(key):
            doc[key] = entry[key]
    doc["valid"] = not errors
    doc["errors"] = errors
    if constraints:
        doc["constraints"] = constraints
    for key in _NOTICE_KEYS:
        if entry.get(key):
            doc[key] = entry[key]
    if entry.get("outdated"):
        doc["outdated"] = True
    return doc


def register_validate_tools(mcp: FastMCP, client: httpx.AsyncClient) -> None:
    @mcp.tool()
    async def validate_flow(
        source: Annotated[
            str,
            Field(
                description="The flow YAML source to validate. Must include 'id' and 'namespace'. "
                "Several flows may be separated by '---'."
            ),
        ],
    ) -> dict:
        """Validate a flow's YAML source without creating it.

        Use this before create_flow_from_yaml to catch syntax, type and
        constraint errors while the flow can still be fixed for free.

        Returns {"valid": bool, "documents": [...]} with one entry per YAML
        document. Each entry has "index", "namespace" and "flow" when known,
        "valid", and "errors" (a list of {"detail", "pointer", "path"}; older
        Kestra versions give only "detail") plus the raw "constraints" text
        when invalid. Non-fatal "warnings", "deprecationPaths", "infos" and
        "outdated" are included when Kestra reports them. "valid" is true only
        when every document is valid.
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

        Each entry in `files` is validated independently and reported per
        path in the validate_flow shape, plus an overall `valid` flag that is
        true only when every file is valid.
        """
        results = {
            str(path): await _validate_flow_source(client, str(src))
            for path, src in (files or {}).items()
        }
        return {
            "valid": all(r["valid"] for r in results.values()),
            "files": results,
        }
