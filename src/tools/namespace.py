from mcp.server.fastmcp import FastMCP
import httpx
from typing import Annotated
from pydantic import Field
from kestra.utils import _render_dependencies


def register_namespace_tools(mcp: FastMCP, client: httpx.AsyncClient) -> None:
    @mcp.tool()
    async def list_namespaces(
        query: Annotated[
            str,
            Field(
                description="The query string for full-text search in namespace names"
            ),
        ] = "",
        page_size: Annotated[
            int,
            Field(
                description="The number of namespaces to return per page. Default is 100."
            ),
        ] = 100,
        with_flows_only: Annotated[
            bool,
            Field(
                description="Whether to only include namespaces with at least one flow. Default is False."
            ),
        ] = False,
    ) -> list:
        """List all existing namespaces, with an option to only include those with flows.
        Optionally filter by a text query. Handles pagination automatically.

        If with_flows_only is True, only namespaces with at least one flow are returned (as a list of strings).
        Otherwise, outputs each namespace as plain text in the format:
        <id> (created)     if disabled == false
        <id> (not created) if disabled == true
        """
        if with_flows_only:
            resp = await client.get("/flows/distinct-namespaces")
            resp.raise_for_status()
            return resp.json()

        all_namespaces: list[str] = []
        page = 1

        while True:
            params = {"page": page, "size": page_size}
            if query:
                params["q"] = query

            resp = await client.get("/namespaces/search", params=params)
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("content", data.get("results", data))
            if not isinstance(batch, list) or not batch:
                break

            for ns in batch:
                ns_id = ns.get("id", "<unknown>")
                disabled = ns.get("disabled", False)
                status = "created" if disabled is False else "not created"
                all_namespaces.append(f"{ns_id} ({status})")

            if len(batch) < page_size:
                break
            page += 1

        return all_namespaces

    @mcp.tool()
    async def list_flows_in_namespace(
        namespace: Annotated[
            str, Field(description="The namespace to list flows from")
        ],
    ) -> list:
        """Retrieve all flows in a given namespace."""
        resp = await client.get(f"/flows/{namespace}")
        resp.raise_for_status()
        return resp.json()

    @mcp.tool()
    async def list_namespace_dependencies(
        namespace: Annotated[
            str, Field(description="The namespace to list dependencies from")
        ],
    ) -> str:
        """Retrieve all flow-to-flow dependencies within a given namespace and render them as an ASCII dependency graph **using only the flow IDs**. Always return the legend after the graph."""
        resp = await client.get(f"/namespaces/{namespace}/dependencies")
        resp.raise_for_status()
        return await _render_dependencies(
            resp.json(),
            "Flows listed without arrows have no dependencies within this namespace.",
        )
