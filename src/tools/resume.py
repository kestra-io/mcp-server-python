from mcp.server.fastmcp import FastMCP
import httpx
from typing import Annotated, List, Any, Dict
from pydantic import Field
from kestra.utils import _parse_iso
from kestra.constants import (
    _VALID_FORCE_STATES,
)


def register_resume_tools(mcp: FastMCP, client: httpx.AsyncClient) -> None:
    @mcp.tool()
    async def resume_execution(
        execution_ids: Annotated[
            List[str],
            Field(
                description="List of execution IDs to resume. If not provided, namespace and flow_id are required to fetch the most recent paused execution."
            ),
        ] = None,
        namespace: Annotated[
            str,
            Field(
                description="Namespace of the flow (required if execution_ids is not provided)"
            ),
        ] = None,
        flow_id: Annotated[
            str,
            Field(description="Flow ID (required if execution_ids is not provided)"),
        ] = None,
        on_resume: Annotated[
            dict,
            Field(
                description="Key-value pairs to send as multipart/form-data. If not provided, the bulk resume endpoint is used or the Pause task defaults are used if available."
            ),
        ] = None,
    ) -> Dict[str, Any]:
        """
        Resume one or more paused executions by ID, or the most recent paused execution for a given namespace and flow.
        If execution_ids are not provided, requires namespace and flow_id, fetches the most recent paused execution for the given flow, and resumes it.
        If `on_resume` is provided, its key→value pairs are sent as multipart/form-data; otherwise, uses the bulk resume endpoint or fetches defaults from the Pause task if available.

        Returns:
        - BulkResponse JSON when resuming multiple executions or when on_resume is None, or
        - A dict mapping each executionId to {} (204) or its JSON (200).
        """
        results: Dict[str, Any] = {}
        if execution_ids:
            if on_resume is None:
                on_resume = {}
            if not on_resume:
                resp = await client.post("executions/resume/by-ids", json=execution_ids)
                resp.raise_for_status()
                return resp.json()
            for exec_id in execution_ids:
                files = [(key, (None, str(value))) for key, value in on_resume.items()]
                resp = await client.post(f"executions/{exec_id}/resume", files=files)
                if resp.status_code == 204:
                    results[exec_id] = {}
                else:
                    resp.raise_for_status()
                    results[exec_id] = resp.json()
            return results
        else:
            if not (namespace and flow_id):
                raise ValueError(
                    "If execution_ids is not provided, both namespace and flow_id are required."
                )
            params = {
                "namespace": namespace,
                "flowId": flow_id,
                "state": "PAUSED",
                "size": 100,
                "sort": "state.startDate,desc",
            }
            resp = await client.get("/executions", params=params)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                executions = data.get("results") or data.get("content") or []
            elif isinstance(data, list):
                executions = data
            else:
                executions = []
            if not executions:
                raise ValueError(
                    f"No paused executions found for {namespace}/{flow_id}. "
                    f"Checked with params: {params}"
                )
            latest = max(executions, key=lambda e: _parse_iso(e["state"]["startDate"]))
            exec_id = latest["id"]
            use_defaults = on_resume is None or (
                isinstance(on_resume, dict) and len(on_resume) == 0
            )
            if on_resume is None:
                on_resume = {}
            if use_defaults:
                flow_resp = await client.get(f"/flows/{namespace}/{flow_id}")
                flow_resp.raise_for_status()
                flow_data = flow_resp.json()
                pause_task = None
                for task in flow_data.get("tasks", []):
                    if task.get("type", "").endswith("Pause") and "onResume" in task:
                        pause_task = task
                        break
                defaults = {}
                if pause_task:
                    for item in pause_task.get("onResume", []):
                        if "id" in item and "defaults" in item:
                            defaults[item["id"]] = item["defaults"]
                if defaults:
                    files = [
                        (key, (None, str(value))) for key, value in defaults.items()
                    ]
                    resp = await client.post(
                        f"/executions/{exec_id}/resume", files=files
                    )
                else:
                    resp = await client.post(f"/executions/{exec_id}/resume")
            else:
                files = [(key, (None, str(value))) for key, value in on_resume.items()]
                resp = await client.post(f"/executions/{exec_id}/resume", files=files)
            if resp.status_code == 204:
                results[exec_id] = {}
            else:
                resp.raise_for_status()
                results[exec_id] = resp.json()
            return results

    @mcp.tool()
    async def force_run_execution(
        execution_id: Annotated[
            str, Field(description="The ID of the execution to force run")
        ],
    ) -> dict:
        """Force run an execution only if it is in CREATED, PAUSED, or QUEUED state.
        Otherwise, raises a ValueError with the current execution state."""
        get_resp = await client.get(f"/executions/{execution_id}")
        get_resp.raise_for_status()
        exec_data = get_resp.json()
        current_state = exec_data.get("state", {}).get("current")

        if current_state not in _VALID_FORCE_STATES:
            raise ValueError(
                f"Cannot force-run execution `{execution_id}` because it's currently in `{current_state}` state."
            )

        run_resp = await client.post(f"/executions/{execution_id}/force-run")
        run_resp.raise_for_status()
        return run_resp.json()
