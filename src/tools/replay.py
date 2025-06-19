from mcp.server.fastmcp import FastMCP
import httpx
from typing import Annotated
from pydantic import Field
from kestra.utils import get_latest_execution


def register_replay_tools(mcp: FastMCP, client: httpx.AsyncClient) -> None:
    @mcp.tool()
    async def replay_execution(
        ids: Annotated[
            list,
            Field(
                description="List of execution IDs to replay. If not provided, namespace and flow_id are required to fetch the latest execution."
            ),
        ] = None,
        namespace: Annotated[
            str,
            Field(
                description="Namespace of the flow (required if ids is not provided)"
            ),
        ] = None,
        flow_id: Annotated[
            str, Field(description="Flow ID (required if ids is not provided)")
        ] = None,
        latest_revision: Annotated[
            bool, Field(description="Use the latest flow revision (optional)")
        ] = False,
    ) -> dict:
        """
        Replay (NOT RESTART, BUT REPLAY) one or more past executions by ID, or the latest execution for a given flow if a list of ids is not provided. Returns a summary string if replaying the latest execution, otherwise the API JSON response.
        """
        if not ids:
            if not (namespace and flow_id):
                raise ValueError(
                    "If ids is not provided, both namespace and flow_id are required."
                )
            latest = await get_latest_execution(client, namespace, flow_id)
            exec_id = latest.get("id")
            if not exec_id:
                raise ValueError(f"No executions found for {namespace}/{flow_id}.")
            ids = [exec_id]
            params: dict[str, bool] = {}
            if latest_revision:
                params["latestRevision"] = True
            resp = await client.post(
                "/executions/replay/by-ids", params=params or None, json=ids
            )
            resp.raise_for_status()
            result = resp.json()
            return f"Replayed execution {exec_id} for flow '{flow_id}' in namespace '{namespace}'.\nResult: {result}"
        else:
            params: dict[str, bool] = {}
            if latest_revision:
                params["latestRevision"] = True
            resp = await client.post(
                "/executions/replay/by-ids", params=params or None, json=ids
            )
            resp.raise_for_status()
            return resp.json()
