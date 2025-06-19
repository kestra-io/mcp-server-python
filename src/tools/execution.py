from mcp.server.fastmcp import FastMCP
import httpx
from typing import Annotated, Any, Literal, List
from pydantic import Field
from datetime import datetime, timedelta, timezone
from kestra.utils import _parse_iso
from kestra.constants import (
    _VALID_STATES,
    _TERMINAL_STATES,
    _RESERVED_FLOW_IDS,
)


def register_execution_tools(mcp: FastMCP, client: httpx.AsyncClient) -> None:
    @mcp.tool()
    async def execute_flow(
        namespace: Annotated[
            str, Field(description="The namespace of the flow to execute")
        ],
        flow_id: Annotated[str, Field(description="The ID of the flow to execute")],
        revision: Annotated[
            int, Field(description="The revision of the flow to execute. Default is 0.")
        ] = 0,
        wait: Annotated[
            bool,
            Field(
                description="Whether to wait for the execution to complete. Default is False."
            ),
        ] = False,
        inputs: Annotated[
            dict,
            Field(
                description="The inputs to the flow which must be provided as a flat dictionary, e.g. {'user': 'Harry Potter'}. Do NOT use a list of {'key': ..., 'value': ...} objects for inputs. Default is an empty dictionary."
            ),
        ] = None,
        labels: Annotated[
            list,
            Field(
                description="The labels to the flow. Must be provided as a list of 'key:value' strings, e.g. ['project:hogwarts']. Default is an empty list."
            ),
        ] = None,
        schedule_date: Annotated[
            str,
            Field(
                description="The date to schedule the execution for, which must be provided as a string in ISO 8601 format, e.g. '2025-12-28T12:00:00Z'. Default is an empty string."
            ),
        ] = "",
    ) -> str:
        """Start a flow execution for a given namespace and flow_id."""
        if flow_id in _RESERVED_FLOW_IDS:
            not_allowed = ", ".join(sorted(_RESERVED_FLOW_IDS))
            raise ValueError(
                f"The flow ID `{flow_id}` is reserved and cannot be used. "
                f"Please rename your flow to something other than: {not_allowed}."
            )

        params: dict[str, str | int | bool | list[str]] = {}
        if revision:
            params["revision"] = revision
        if wait:
            params["wait"] = True
        if labels:
            # Convert list of dicts with 'key' and 'value' to list of 'key:value' strings
            if isinstance(labels, list) and all(
                isinstance(l, dict) and "key" in l and "value" in l for l in labels
            ):
                params["labels"] = [f"{l['key']}:{l['value']}" for l in labels]
            else:
                params["labels"] = labels
        if schedule_date:
            params["scheduleDate"] = schedule_date

        if inputs is None:
            inputs = {}
        # Support both dict and list-of-dict (with 'key' and 'value') formats for inputs
        if isinstance(inputs, list) and all(
            isinstance(i, dict) and "key" in i and "value" in i for i in inputs
        ):
            inputs = {i["key"]: i["value"] for i in inputs}

        files = [(k, (None, str(v))) for k, v in inputs.items()]

        resp = await client.post(
            f"/executions/{namespace}/{flow_id}", params=params, files=files or None
        )
        resp.raise_for_status()
        return resp.json()

    @mcp.tool()
    async def manage_executions(
        action: Annotated[
            Literal["pause", "kill", "delete", "get", "change_status"],
            Field(
                description="The action to perform: pause, kill, delete, get, change_status. Use 'pause' to pause a running execution, 'kill' to kill an execution (and its subflows if cascade=True), 'delete' to delete an execution by ID, 'get' to retrieve full details for a single execution by its ID, 'change_status' to change the state of an execution (requires 'status')."
            ),
        ],
        execution_id: Annotated[
            str, Field(description="The execution ID to operate on")
        ],
        status: Annotated[
            str,
            Field(
                description="The new status for 'change_status' action. Required for 'change_status' action."
            ),
        ] = None,
        cascade: Annotated[
            bool,
            Field(
                description="Whether to cascade kill to subflows. Only for 'kill', default True."
            ),
        ] = True,
    ) -> dict:
        """Manage an execution by action. Base don the action, the tool returns:
        - For 'pause': {"status": "paused"}
        - For 'kill': {"executionId": ..., "status": ...}
        - For 'delete': {} on successful deletion
        - For 'get': the execution object as JSON
        - For 'change_status': the updated execution object as JSON"""
        if action == "pause":
            resp = await client.post(f"/executions/{execution_id}/pause")
            resp.raise_for_status()
            return {"status": "paused"}
        elif action == "kill":
            resp = await client.delete(
                f"/executions/{execution_id}/kill", params={"isOnKillCascade": cascade}
            )
            status_map = {
                202: "kill_requested",
                404: "not_found",
                409: "already_finished",
            }
            return {
                "executionId": execution_id,
                "status": status_map.get(resp.status_code, f"error_{resp.status_code}"),
            }
        elif action == "delete":
            resp = await client.delete(f"/executions/{execution_id}")
            if resp.status_code == 204:
                return {}
            resp.raise_for_status()
            return resp.json()
        elif action == "get":
            resp = await client.get(f"/executions/{execution_id}")
            resp.raise_for_status()
            return resp.json()
        elif action == "change_status":
            if not status:
                raise ValueError("'status' is required for change_status action.")
            if status not in _VALID_STATES:
                allowed = ", ".join(sorted(_VALID_STATES))
                raise ValueError(
                    f"Invalid status `{status}`. Must be one of: {allowed}"
                )
            resp = await client.post(
                f"/executions/{execution_id}/change-status", params={"status": status}
            )
            resp.raise_for_status()
            return resp.json()
        else:
            raise ValueError(
                "Action must be one of: pause, kill, delete, get, change_status"
            )

    @mcp.tool()
    async def add_execution_labels(
        execution_id: Annotated[
            str, Field(description="The execution ID to add labels to")
        ],
        labels: Annotated[
            list[dict],
            Field(
                description="The labels to add to the execution. Must be provided as a list of objects with 'key' and 'value' fields."
            ),
        ],
    ) -> dict:
        """Add or update labels of a terminated execution. The execution must be in one of the terminal states:
        SUCCESS, WARNING, FAILED, KILLED, CANCELLED, or SKIPPED. Raises a ValueError otherwise.

        Parse any free‑form label text from the user prompt (e.g. "replay:true project:mcp" or "team:datateam") into a list of {"key": ..., "value": ...} objects and pass it here as `labels`.
        """
        get_resp = await client.get(f"/executions/{execution_id}")
        get_resp.raise_for_status()
        exec_data = get_resp.json()
        current_state = exec_data.get("state", {}).get("current")
        existing = exec_data.get("labels", [])

        if current_state not in _TERMINAL_STATES:
            allowed = ", ".join(sorted(_TERMINAL_STATES))
            raise ValueError(
                f"Cannot add labels to execution `{execution_id}` because its current state is `{current_state}`. "
                f"Labels may only be added when execution is in one of: {allowed}."
            )

        if not isinstance(labels, list) or not all(
            isinstance(l, dict) and "key" in l and "value" in l for l in labels
        ):
            raise ValueError(
                "`labels` must be a list of objects with 'key' and 'value' fields"
            )

        label_map = {lbl["key"]: lbl["value"] for lbl in existing}
        for l in labels:
            label_map[l["key"]] = l["value"]
        merged_labels = [{"key": k, "value": v} for k, v in label_map.items()]

        resp = await client.post(
            f"/executions/{execution_id}/labels", json=merged_labels
        )
        resp.raise_for_status()
        return resp.json()

    @mcp.tool()
    async def list_executions(
        namespace: Annotated[
            str, Field(description="The namespace to list executions from")
        ],
        flow_id: Annotated[
            str, Field(description="Optional flow ID to filter executions")
        ] = "",
        count: Annotated[
            int,
            Field(
                description="Optional number of most recent executions to return. If None, returns all executions"
            ),
        ] = None,
        minutes: Annotated[
            int,
            Field(
                description="Optional time window in minutes to filter executions. If None, no time filtering"
            ),
        ] = None,
        page_size: Annotated[
            int,
            Field(
                description="Number of executions to fetch per page. Default is 100."
            ),
        ] = 100,
    ) -> List[dict]:
        """List executions in a namespace (and optional flow) with flexible filtering options.

        Returns a JSON array of execution objects, sorted by startDate in descending order (newest first). Each returned object includes at least:
        - id, namespace, flowId
        - state.current, state.startDate, state.endDate
        - durationSeconds
        - inputs (dict)
        - labels (dict)

        Format the result as markdown. For each execution, output:
        - a heading with `### Execution ID {{id}}`
        - a list with:
        - **State**: the current state
        - **Start Date**: from state.startDate
        - **End Date**: from state.endDate
        - **Duration**: from durationSeconds (in seconds)
        - **Inputs**: key1: val1, key2: val2 (comma-separated)
        - **Labels**: key1: val1, key2: val2 (comma-separated)

        Only include fields that exist in the object."""
        all_execs: List[dict] = []
        page = 1

        # If we only need one execution, we can optimize by using a smaller page size
        if count == 1:
            page_size = 10  # Small page size is sufficient for finding the latest

        while True:
            params: dict[str, Any] = {
                "namespace": namespace,
                "page": page,
                "size": page_size,
            }
            if flow_id:
                params["flowId"] = flow_id

            resp = await client.get("/executions", params=params)
            resp.raise_for_status()
            data = resp.json()

            batch = data.get("results", data.get("content", []))
            if not batch:
                break

            all_execs.extend(batch)

            # If we have enough executions to satisfy the count requirement, we can stop
            if count is not None and len(all_execs) >= count:
                break

            # If we got fewer results than requested, we've reached the end
            if len(batch) < page_size:
                break

            page += 1

        # Apply time filtering if requested
        if minutes is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
            all_execs = [
                e for e in all_execs if _parse_iso(e["state"]["startDate"]) >= cutoff
            ]

        # Sort by startDate in descending order (newest first)
        all_execs.sort(key=lambda e: _parse_iso(e["state"]["startDate"]), reverse=True)

        # Return only the requested number of executions
        if count is not None:
            all_execs = all_execs[:count]

        return all_execs
