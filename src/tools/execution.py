from fastmcp import FastMCP
import asyncio
import httpx
import logging
from typing import Annotated, Any, Literal, List
from pydantic import Field
import re
from datetime import datetime, timedelta, timezone
from kestra.compat import (
    attach_execution_outputs,
    attach_task_outputs,
    execution_action_path,
    executions_filter_params,
    executions_search_params,
)
from kestra.utils import _parse_iso
from kestra.constants import (
    _VALID_STATES,
    _TERMINAL_STATES,
    _RESERVED_FLOW_IDS,
)

logger = logging.getLogger(__name__)


def _parse_iso_duration(duration: str) -> timedelta:
    """Parse a simple ISO 8601 duration (PnD, PTnH, PTnM, PnDTnHnM) into a timedelta."""
    m = re.fullmatch(
        r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?",
        duration.strip(),
    )
    if not m or not any(m.groups()):
        raise ValueError(
            f"Unsupported ISO 8601 duration: {duration!r}. "
            "Use formats like 'P30D', 'P7D', 'PT5M', 'PT1H'."
        )
    days = int(m.group(1) or 0)
    hours = int(m.group(2) or 0)
    minutes = int(m.group(3) or 0)
    seconds = int(m.group(4) or 0)
    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


def _resolve_start_date(
    start_date: str, time_range: str, now: datetime | None = None
) -> str:
    """The absolute startDate to send to `/executions/search`.

    An explicit `start_date` wins. Otherwise `time_range` is subtracted from now,
    and without either the window is the last 90 days. Day-only durations snap
    back to midnight so "last 10 days" includes the whole of the boundary day,
    not just the past 240 hours. An explicit date is always sent rather than the
    API's `timeRange`, whose exact-hour arithmetic excludes that boundary day.
    """
    if start_date:
        return start_date
    now = now or datetime.now(timezone.utc)
    delta = _parse_iso_duration(time_range) if time_range else timedelta(days=90)
    point = now - delta
    # Whole-day durations, including PT24H, which normalizes to one day.
    if delta.seconds == 0 and delta.microseconds == 0:
        point = point.replace(hour=0, minute=0, second=0, microsecond=0)
    return point.strftime("%Y-%m-%dT%H:%M:%SZ")


# Upper bound on `search_executions(limit=...)`. Every page is one request, so
# this also bounds the size of a single tool response.
_SEARCH_LIMIT_MAX = 100
# Parallel `GET /executions/{id}` reads used for `failedTasks` on Kestra 2.x.
_FAILED_TASKS_CONCURRENCY = 8


def _validate_states(states: list[str] | None) -> list[str]:
    if not states:
        return []
    normalized = [s.strip().upper() for s in states if s and s.strip()]
    invalid = sorted(set(normalized) - _VALID_STATES)
    if invalid:
        raise ValueError(
            f"Invalid state(s): {', '.join(invalid)}. "
            f"Must be any of: {', '.join(sorted(_VALID_STATES))}"
        )
    return list(dict.fromkeys(normalized))


def _failed_task_ids(execution: dict) -> list[str]:
    """IDs of the tasks whose run ended FAILED, in run order, without repeats."""
    return list(
        dict.fromkeys(
            task_run.get("taskId")
            for task_run in execution.get("taskRunList") or []
            if isinstance(task_run, dict)
            and (task_run.get("state") or {}).get("current") == "FAILED"
            and task_run.get("taskId")
        )
    )


def _search_summary(execution: dict, include_inputs: bool) -> dict[str, Any]:
    """The fields of an execution `search_executions` returns.

    Labels are flattened to a dict, without the `system.*` labels Kestra adds
    to every execution, which carry no information for a search result.
    """
    state = execution.get("state") or {}
    summary: dict[str, Any] = {
        "id": execution.get("id"),
        "namespace": execution.get("namespace"),
        "flowId": execution.get("flowId"),
        "state": {
            "current": state.get("current"),
            "startDate": state.get("startDate"),
            "endDate": state.get("endDate"),
            "duration": state.get("duration"),
        },
    }
    labels = {
        label["key"]: label.get("value")
        for label in execution.get("labels") or []
        if isinstance(label, dict)
        and label.get("key")
        and not str(label["key"]).startswith("system.")
    }
    if labels:
        summary["labels"] = labels
    if include_inputs and execution.get("inputs"):
        summary["inputs"] = execution["inputs"]
    return summary


def _normalize_labels(labels) -> list[dict]:
    """Normalize labels from any format into list of {"key": ..., "value": ...} dicts.

    Accepts:
      - dict: {"env": "prod", "team": "data"} → [{"key": "env", "value": "prod"}, ...]
      - list of "key:value" strings: ["env:prod"] → [{"key": "env", "value": "prod"}]
      - list of {"key": ..., "value": ...} dicts: passed through as-is
    """
    if isinstance(labels, dict):
        return [{"key": str(k), "value": str(v)} for k, v in labels.items()]
    if isinstance(labels, list):
        result = []
        for item in labels:
            if isinstance(item, dict) and "key" in item and "value" in item:
                result.append({"key": str(item["key"]), "value": str(item["value"])})
            elif isinstance(item, str) and ":" in item:
                key, _, value = item.partition(":")
                result.append({"key": key, "value": value})
            else:
                raise ValueError(
                    f"Invalid label format: {item!r}. "
                    "Labels must be a dict like {\"env\": \"prod\"}, "
                    "a list of \"key:value\" strings, or a list of "
                    "{\"key\": ..., \"value\": ...} objects."
                )
        return result
    raise ValueError(
        f"Invalid labels type: {type(labels).__name__}. "
        "Labels must be a dict like {\"env\": \"prod\"}, "
        "a list of \"key:value\" strings, or a list of "
        "{\"key\": ..., \"value\": ...} objects."
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
            dict | None,
            Field(
                description="The inputs to the flow which must be provided as a flat dictionary, e.g. {'user': 'Harry Potter'}. Do NOT use a list of {'key': ..., 'value': ...} objects for inputs. Default is an empty dictionary."
            ),
        ] = None,
        labels: Annotated[
            dict | None,
            Field(
                description="The labels for the execution as a flat dictionary, e.g. {'environment': 'test', 'project': 'hogwarts'}. Default is no labels."
            ),
        ] = None,
        schedule_date: Annotated[
            str,
            Field(
                description="The date to schedule the execution for, which must be provided as a string in ISO 8601 format, e.g. '2025-12-28T12:00:00Z'. Default is an empty string."
            ),
        ] = "",
    ) -> dict:
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
            normalized = _normalize_labels(labels)
            params["labels"] = [f"{l['key']}:{l['value']}" for l in normalized]
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
        return await attach_task_outputs(
            client, await attach_execution_outputs(client, resp.json())
        )

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
            str | None,
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
            resp = await client.post(
                await execution_action_path(client, execution_id, "pause")
            )
            resp.raise_for_status()
            return {"status": "paused"}
        elif action == "kill":
            resp = await client.delete(
                await execution_action_path(client, execution_id, "kill"),
                params={"isOnKillCascade": cascade},
            )
            # 1.x answers a kill with 202, 2.x with 200.
            status_map = {
                200: "kill_requested",
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
            return await attach_task_outputs(
                client, await attach_execution_outputs(client, resp.json())
            )
        elif action == "change_status":
            if not status:
                raise ValueError("'status' is required for change_status action.")
            if status not in _VALID_STATES:
                allowed = ", ".join(sorted(_VALID_STATES))
                raise ValueError(
                    f"Invalid status `{status}`. Must be one of: {allowed}"
                )
            resp = await client.post(
                await execution_action_path(client, execution_id, "change-status"),
                params={"status": status},
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
            dict,
            Field(
                description="The labels to add as a flat dictionary, e.g. {'team': 'datateam', 'replay': 'true'}."
            ),
        ],
    ) -> dict:
        """Add or update labels of a terminated execution. The execution must be in one of the terminal states:
        SUCCESS, WARNING, FAILED, KILLED, CANCELLED, or SKIPPED. Raises a ValueError otherwise.
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

        normalized = _normalize_labels(labels)
        label_map = {lbl["key"]: lbl["value"] for lbl in existing}
        for l in normalized:
            label_map[l["key"]] = l["value"]
        merged_labels = [{"key": k, "value": v} for k, v in label_map.items()]

        resp = await client.post(
            await execution_action_path(client, execution_id, "labels"),
            json=merged_labels,
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
            int | None,
            Field(
                description="Optional number of most recent executions to return. If None, returns all executions within the date range."
            ),
        ] = None,
        time_range: Annotated[
            str,
            Field(
                description=(
                    "Relative time range as an ISO 8601 duration, e.g. 'P30D' for the last 30 days, "
                    "'P7D' for the last 7 days, 'PT5M' for the last 5 minutes, 'PT1H' for the last hour. "
                    "Defaults to 'P90D' (last 90 days) when no date filter is provided. "
                    "Use start_date/end_date instead for an absolute date range."
                )
            ),
        ] = "",
        start_date: Annotated[
            str,
            Field(
                description="Absolute start of the date range in ISO 8601 format, e.g. '2025-04-01T00:00:00Z'. Takes precedence over time_range."
            ),
        ] = "",
        end_date: Annotated[
            str,
            Field(
                description="Absolute end of the date range in ISO 8601 format, e.g. '2025-04-30T23:59:59Z'. Only used with start_date."
            ),
        ] = "",
        page_size: Annotated[
            int,
            Field(
                description="Number of executions to fetch per page. Default is 100."
            ),
        ] = 100,
    ) -> dict:
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
        effective_start_date = _resolve_start_date(start_date, time_range)

        while True:
            params = await executions_search_params(
                client,
                namespace=namespace,
                flow_id=flow_id or None,
                start_date=effective_start_date,
                end_date=end_date or None,
                page=page,
                size=page_size,
            )

            resp = await client.get("/executions/search", params=params)
            resp.raise_for_status()
            data = resp.json()

            batch = data.get("results", data.get("content", []))
            if not batch:
                break

            all_execs.extend(batch)

            # If we got fewer results than requested, we've reached the end
            if len(batch) < page_size:
                break

            page += 1

        # Sort client-side — the search endpoint doesn't accept a sort param reliably
        all_execs.sort(key=lambda e: _parse_iso(e["state"]["startDate"]), reverse=True)

        # Return only the requested number of executions
        if count is not None:
            all_execs = all_execs[:count]

        # Trim each execution to only the fields needed — full objects include
        # taskRunList, outputs, metadata, etc. which bloat the response significantly.
        def _slim(e: dict) -> dict:
            state = e.get("state", {})
            slim: dict[str, Any] = {
                "id": e.get("id"),
                "namespace": e.get("namespace"),
                "flowId": e.get("flowId"),
                "state": {
                    "current": state.get("current"),
                    "startDate": state.get("startDate"),
                    "endDate": state.get("endDate"),
                },
            }
            if "durationSeconds" in e:
                slim["durationSeconds"] = e["durationSeconds"]
            if e.get("inputs"):
                slim["inputs"] = e["inputs"]
            if e.get("labels"):
                slim["labels"] = e["labels"]
            return slim

        return {"results": [_slim(e) for e in all_execs]}

    @mcp.tool()
    async def search_executions(
        state: Annotated[
            list[str] | None,
            Field(
                description=(
                    "Execution states to match, any of them, e.g. ['FAILED'] or "
                    "['FAILED', 'KILLED']. Leave empty to match every state. Valid: "
                    + ", ".join(sorted(_VALID_STATES))
                )
            ),
        ] = None,
        namespace: Annotated[
            str | None,
            Field(description="Namespace to search in. Leave empty to search all namespaces."),
        ] = None,
        include_child_namespaces: Annotated[
            bool,
            Field(
                description=(
                    "Also match the child namespaces of `namespace`, e.g. 'company' then "
                    "matches 'company.team' too. Default false: the namespace must match exactly."
                )
            ),
        ] = False,
        flow_id: Annotated[
            str | None, Field(description="Flow ID to match. Usually combined with `namespace`.")
        ] = None,
        labels: Annotated[
            dict[str, str] | None,
            Field(description="Labels the execution must all carry, e.g. {'env': 'prod'}."),
        ] = None,
        query: Annotated[
            str | None,
            Field(description="Free-text query, matched by Kestra against fields such as the namespace and flow ID."),
        ] = None,
        time_range: Annotated[
            str | None,
            Field(
                description=(
                    "Relative window as an ISO 8601 duration ending now, e.g. 'PT1H', 'P1D', "
                    "'P7D'. Whole-day durations start at midnight. Defaults to 'P90D' when "
                    "neither this nor start_date is given."
                )
            ),
        ] = None,
        start_date: Annotated[
            str | None,
            Field(
                description=(
                    "Absolute start of the window in ISO 8601, e.g. '2026-04-01T00:00:00Z'. "
                    "Takes precedence over time_range."
                )
            ),
        ] = None,
        end_date: Annotated[
            str | None,
            Field(description="Absolute end of the window in ISO 8601, e.g. '2026-04-30T23:59:59Z'."),
        ] = None,
        limit: Annotated[
            int,
            Field(
                ge=1,
                le=_SEARCH_LIMIT_MAX,
                description=f"Executions per page, 1 to {_SEARCH_LIMIT_MAX}. Default 25.",
            ),
        ] = 25,
        page: Annotated[
            int,
            Field(ge=1, description="Page to return, starting at 1. Use when `has_more` is true."),
        ] = 1,
        include_inputs: Annotated[
            bool,
            Field(description="Include each execution's inputs. Off by default, as they can be large."),
        ] = False,
    ) -> dict:
        """Search executions by state, namespace, flow, labels, text and time window, newest first.

        All filtering, sorting and paging happen on the Kestra server, and each call
        reads a single page, so it stays fast and small on instances with millions of
        executions. Use it, for instance, to find the latest failures:
        `search_executions(state=['FAILED'], time_range='P1D', limit=10)`.

        Returns:
        - total: how many executions match, across all pages
        - page, limit: the page returned and its size
        - has_more: true when more matching executions exist on later pages
        - results: executions, each with id, namespace, flowId, state (current,
          startDate, endDate, duration), labels (without the system.* ones), inputs
          (only with include_inputs) and, for FAILED executions, failedTasks: the IDs
          of the tasks that failed
        - warning: present only when failedTasks could not be read for some executions

        Report the total, and say so when has_more is true rather than presenting the
        page as the complete result. Format each execution as a markdown list item
        with its ID, flow (namespace.flowId), state, start date and, when present,
        failed tasks."""
        states = _validate_states(state)
        params = executions_filter_params(
            states=states,
            namespace=namespace or None,
            include_child_namespaces=include_child_namespaces,
            flow_id=flow_id or None,
            labels=labels or None,
            query=query or None,
            start_date=_resolve_start_date(start_date or "", time_range or ""),
            end_date=end_date or None,
            page=page,
            size=limit,
            sort="state.startDate:desc",
        )
        resp = await client.get("/executions/search", params=params)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("results") or []
        total = data.get("total")
        if not batch and page > 1:
            # Past the last page Kestra answers `total: 0`, which would read as
            # "nothing matches". Ask for the first page to learn the real count.
            first = await client.get(
                "/executions/search", params={**params, "page": 1, "size": 1}
            )
            first.raise_for_status()
            total = first.json().get("total")
        if total is None:
            total = (page - 1) * limit + len(batch)

        results = [_search_summary(e, include_inputs) for e in batch]

        # 1.x carries taskRunList in search results; 2.x drops it there, so the
        # failed executions of the page are read one by one instead.
        failed = [
            (summary, execution)
            for summary, execution in zip(results, batch)
            if summary["state"]["current"] == "FAILED"
        ]
        semaphore = asyncio.Semaphore(_FAILED_TASKS_CONCURRENCY)

        async def read_task_runs(execution: dict) -> dict | None:
            if "taskRunList" in execution:
                return execution
            async with semaphore:
                try:
                    r = await client.get(f"/executions/{execution['id']}")
                    if r.status_code == 200:
                        return r.json()
                    logger.warning(
                        "GET /executions/%s answered %s", execution["id"], r.status_code
                    )
                except (httpx.HTTPError, ValueError) as e:
                    logger.warning("Execution %s could not be read: %s", execution["id"], e)
                return None

        full = await asyncio.gather(*(read_task_runs(e) for _, e in failed))
        unread = []
        for (summary, execution), detail in zip(failed, full):
            if detail is None:
                unread.append(execution["id"])
            else:
                summary["failedTasks"] = _failed_task_ids(detail)

        response: dict[str, Any] = {
            "total": total,
            "page": page,
            "limit": limit,
            "has_more": page * limit < total,
            "results": results,
        }
        if unread:
            response["warning"] = (
                "failedTasks could not be read for executions "
                f"{', '.join(unread)}; use manage_executions with action 'get' on them."
            )
        return response
