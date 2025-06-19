from mcp.server.fastmcp import FastMCP
import httpx
from datetime import datetime, timedelta, timezone
import os
from typing import Annotated
from pydantic import Field


def ensure_int(val, name: str) -> int:
    if val is None:
        return 0
    try:
        return int(val)
    except (ValueError, TypeError):
        raise ValueError(f"`{name}` must be an integer. Got: {val}")


def register_backfill_tools(mcp: FastMCP, client: httpx.AsyncClient) -> None:
    @mcp.tool()
    async def backfill_executions(
        namespace: Annotated[str, Field(description="The namespace of the flow")],
        flow_id: Annotated[str, Field(description="The id of the flow")],
        start: Annotated[
            str, Field(description="The start date of the backfill")
        ] = None,
        end: Annotated[str, Field(description="The end date of the backfill")] = None,
        days: Annotated[int, Field(description="The number of days to backfill")] = 0,
        hours: Annotated[int, Field(description="The number of hours to backfill")] = 0,
        minutes: Annotated[
            int, Field(description="The number of minutes to backfill")
        ] = 0,
        trigger_id: Annotated[str, Field(description="The id of the trigger")] = None,
        inputs: Annotated[
            dict, Field(description="The inputs for the backfill")
        ] = None,
        labels: Annotated[
            list, Field(description="The labels for the backfill")
        ] = None,
    ) -> dict:
        """Backfill executions for a flow within a time window. You can provide explicit start/end dates or use the `days`/`hours`/`minutes` parameters to calculate the start/end dates relative to the current time. Provide those values as integers, e.g., 2 to backfill executions for the last 2 days/hours/minutes."""

        days_int = ensure_int(days, "days")
        hours_int = ensure_int(hours, "hours")
        minutes_int = ensure_int(minutes, "minutes")

        if days_int or hours_int or minutes_int:
            now = datetime.now(timezone.utc)
            delta = timedelta(days=days_int, hours=hours_int, minutes=minutes_int)
            start = (now - delta).isoformat().replace("+00:00", "Z")
            end = now.isoformat().replace("+00:00", "Z")

        if not start:
            raise ValueError(
                "Either `start` or `days`/`hours`/`minutes` must be provided"
            )

        # Find Schedule trigger ID if not given
        if not trigger_id:
            flow_resp = await client.get(f"flows/{namespace}/{flow_id}")
            flow_resp.raise_for_status()
            for t in flow_resp.json().get("triggers", []):
                if t.get("type", "").endswith("Schedule"):
                    trigger_id = t.get("id")
                    break
            if not trigger_id:
                raise ValueError(f"No Schedule trigger found in {namespace}/{flow_id}")

        if inputs is None:
            inputs = {}
        if labels is None:
            labels = []

        payload = {
            "namespace": namespace,
            "flowId": flow_id,
            "triggerId": trigger_id,
            "backfill": {
                "start": start,
                "end": end,
                "inputs": inputs,
                "labels": labels,
            },
        }
        if tenant := os.getenv("KESTRA_TENANT_ID"):
            payload["tenantId"] = tenant

        resp = await client.put("triggers", json=payload)
        resp.raise_for_status()
        return resp.json()
