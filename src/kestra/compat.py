"""Request translation between the Kestra 1.x and 2.x REST APIs.

Kestra 2.0 moved the execution action endpoints under `/actions/`, replaced the
per-endpoint search query parameters with the unified `filters[field][OPERATION]`
model, moved backfill creation to `/triggers/backfill/create` and dropped the
per-namespace KV listing in favour of `GET /kv`.

A 1.x query parameter sent to a 2.x server is not rejected, it is ignored, so
`/executions/search?namespace=x` answers 200 with every execution on the
instance. Every request whose shape differs between the two majors is therefore
built here from the version the server reports, rather than being sent in one
dialect and hoped for.
"""

import asyncio
import logging
import os
import re
from typing import Any, Optional

import httpx

from kestra.utils import _root_api_url

logger = logging.getLogger(__name__)

# Assumed when the server version cannot be read. A 1.x request against a 2.x
# server is accepted and silently unfiltered, while a 2.x request against a 1.x
# server fails loudly, so guessing the newer major fails in the safer direction.
_FALLBACK_MAJOR = 2

_CACHE_ATTR = "_kestra_api_major"


def _parse_major(version: Any) -> Optional[int]:
    match = re.match(r"\s*v?(\d+)", str(version or ""))
    return int(match.group(1)) if match else None


def configured_major() -> Optional[int]:
    """The major version forced through `KESTRA_API_VERSION`, if it is set."""
    raw = os.getenv("KESTRA_API_VERSION", "").strip()
    if not raw:
        return None
    major = _parse_major(raw)
    if major is None:
        raise ValueError(
            f"KESTRA_API_VERSION must start with a major version number, "
            f"for example '1', '2', '1.3.3' or '2.0.1'. Got: {raw!r}"
        )
    return major


def reset_version_cache(client: Optional[httpx.AsyncClient] = None) -> None:
    """Forget the version detected for `client`. Intended for tests."""
    if client is not None:
        try:
            delattr(client, _CACHE_ATTR)
        except AttributeError:
            pass


async def api_major(client: httpx.AsyncClient) -> int:
    """The major version of the Kestra API behind `client`.

    Read once from the untenanted `GET /api/v1/configs`, which reports `version`
    and `edition` on both 1.x and 2.x, then cached for the life of the process.
    Setting `KESTRA_API_VERSION` skips the probe entirely.
    """
    forced = configured_major()
    if forced is not None:
        return forced

    cached = getattr(client, _CACHE_ATTR, None)
    if cached is not None:
        return cached

    major = _FALLBACK_MAJOR
    try:
        resp = await client.get(_root_api_url("/configs", client))
        resp.raise_for_status()
        detected = _parse_major(resp.json().get("version"))
        if detected:
            major = detected
        else:
            logger.warning(
                "GET /api/v1/configs returned no version; assuming Kestra %s.x. "
                "Set KESTRA_API_VERSION to override.",
                _FALLBACK_MAJOR,
            )
    except Exception as exc:
        logger.warning(
            "Could not read the Kestra version from /api/v1/configs (%s); "
            "assuming Kestra %s.x. Set KESTRA_API_VERSION to override.",
            exc,
            _FALLBACK_MAJOR,
        )

    setattr(client, _CACHE_ATTR, major)
    return major


async def is_v2(client: httpx.AsyncClient) -> bool:
    return await api_major(client) >= 2


# ─── Paths ───────────────────────────────────────────────────────────────────


async def execution_action_path(
    client: httpx.AsyncClient, execution_id: str, action: str
) -> str:
    """Path for a single-execution action such as `pause`, `kill` or `restart`.

    These moved from `/executions/{id}/{action}` to
    `/executions/{id}/actions/{action}` in Kestra 2.0. On a 2.x server the old
    path is matched by `POST /executions/{namespace}/{id}` instead, so the
    action is answered with a 404, a 405 or a 415 rather than being performed.
    """
    if await is_v2(client):
        return f"/executions/{execution_id}/actions/{action}"
    return f"/executions/{execution_id}/{action}"


async def backfill_create_path(client: httpx.AsyncClient) -> str:
    """Path that creates a backfill, moved off `PUT /triggers` in Kestra 2.0."""
    if await is_v2(client):
        return "/triggers/backfill/create"
    return "/triggers"


async def supports_dashboard_crud(client: httpx.AsyncClient) -> bool:
    """Whether dashboards can be created over the API.

    Kestra 2.0 kept dashboard reads in OSS but moved creating, updating and
    deleting them to the Enterprise Edition, so `POST /dashboards` is absent
    from OSS 2.x.
    """
    if not await is_v2(client):
        return True
    try:
        resp = await client.get(_root_api_url("/configs", client))
        resp.raise_for_status()
        return bool(resp.json().get("isCustomDashboardsEnabled", False))
    except Exception:
        return True


# ─── Filters ─────────────────────────────────────────────────────────────────


def _filter_key(field: str, operation: str, sub_key: Optional[str] = None) -> str:
    key = f"filters[{field}][{operation}]"
    return f"{key}[{sub_key}]" if sub_key else key


def _add(params: dict, field: str, operation: str, value: Any) -> None:
    if value is None or value == "" or value == []:
        return
    params[_filter_key(field, operation)] = value


def _add_labels(params: dict, labels: Any) -> None:
    """Add labels, which 2.x nests one level deeper than every other filter."""
    if not labels:
        return
    if isinstance(labels, dict):
        pairs = labels.items()
    else:
        pairs = (
            item.split(":", 1) if isinstance(item, str) else (item["key"], item["value"])
            for item in labels
        )
    for key, value in pairs:
        params[_filter_key("labels", "EQUALS", key)] = value


def _legacy_labels(labels: Any) -> list[str]:
    """Labels as the `key:value` strings the 1.x endpoints expect."""
    if isinstance(labels, dict):
        return [f"{k}:{v}" for k, v in labels.items()]
    out = []
    for item in labels or []:
        if isinstance(item, str):
            out.append(item)
        else:
            out.append(f"{item['key']}:{item['value']}")
    return out


# ─── Per-endpoint query parameters ───────────────────────────────────────────


async def executions_search_params(
    client: httpx.AsyncClient,
    *,
    namespace: Optional[str] = None,
    flow_id: Optional[str] = None,
    state: Optional[str] = None,
    query: Optional[str] = None,
    labels: Any = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_range: Optional[str] = None,
    page: Optional[int] = None,
    size: Optional[int] = None,
    sort: Optional[str] = None,
) -> dict:
    """Query parameters for `GET /executions/search`."""
    params: dict[str, Any] = {}
    if page is not None:
        params["page"] = page
    if size is not None:
        params["size"] = size
    if sort:
        params["sort"] = sort

    if await is_v2(client):
        _add(params, "namespace", "EQUALS", namespace)
        _add(params, "flowId", "EQUALS", flow_id)
        _add(params, "state", "EQUALS", state)
        _add(params, "q", "EQUALS", query)
        _add(params, "startDate", "GREATER_THAN_OR_EQUAL_TO", start_date)
        _add(params, "endDate", "LESS_THAN_OR_EQUAL_TO", end_date)
        _add(params, "timeRange", "EQUALS", time_range)
        _add_labels(params, labels)
        return params

    if namespace:
        params["namespace"] = namespace
    if flow_id:
        params["flowId"] = flow_id
    if state:
        params["state"] = state
    if query:
        params["q"] = query
    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date
    if time_range:
        params["timeRange"] = time_range
    if labels:
        params["labels"] = _legacy_labels(labels)
    return params


def executions_filter_params(
    *,
    states: Optional[list[str]] = None,
    namespace: Optional[str] = None,
    include_child_namespaces: bool = False,
    flow_id: Optional[str] = None,
    labels: Any = None,
    query: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page: Optional[int] = None,
    size: Optional[int] = None,
    sort: Optional[str] = None,
) -> dict:
    """Query parameters for `GET /executions/search` in the `filters` model only.

    Unlike the other builders this one is the same for both majors. Kestra 1.x
    accepts the `filters[field][OPERATION]` model next to its flat parameters
    (checked on 1.0 and 1.3), and it is the only way to express what this
    search needs there: several states at once, and a namespace matched
    exactly, since the flat 1.x `namespace` is always a prefix. The two
    dialects cannot be mixed, because a 1.x server that receives any `filters`
    key ignores the flat ones.

    `states` is sent as one comma-joined `IN` value; repeating the key matches
    nothing on either major.
    """
    params: dict[str, Any] = {}
    if page is not None:
        params["page"] = page
    if size is not None:
        params["size"] = size
    if sort:
        params["sort"] = sort

    if states:
        _add(params, "state", "IN", ",".join(states))
    _add(
        params,
        "namespace",
        "PREFIX" if include_child_namespaces else "EQUALS",
        namespace,
    )
    _add(params, "flowId", "EQUALS", flow_id)
    _add(params, "q", "EQUALS", query)
    _add(params, "startDate", "GREATER_THAN_OR_EQUAL_TO", start_date)
    _add(params, "endDate", "LESS_THAN_OR_EQUAL_TO", end_date)
    _add_labels(params, labels)
    return params


async def flows_search_params(
    client: httpx.AsyncClient,
    *,
    query: Optional[str] = None,
    namespace: Optional[str] = None,
    scope: Optional[str] = None,
    labels: Any = None,
    page: Optional[int] = None,
    size: Optional[int] = None,
    sort: Optional[str] = None,
) -> dict:
    """Query parameters for `GET /flows/search`."""
    params: dict[str, Any] = {}
    if page is not None:
        params["page"] = page
    if size is not None:
        params["size"] = size
    if sort:
        params["sort"] = sort

    if await is_v2(client):
        _add(params, "q", "EQUALS", query)
        _add(params, "namespace", "EQUALS", namespace)
        _add(params, "scope", "EQUALS", scope)
        _add_labels(params, labels)
        return params

    if query:
        params["q"] = query
    if namespace:
        params["namespace"] = namespace
    if scope:
        params["scope"] = scope
    if labels:
        params["labels"] = _legacy_labels(labels)
    return params


async def execution_logs_params(
    client: httpx.AsyncClient,
    *,
    min_level: Optional[str] = None,
    task_id: Optional[str] = None,
    task_run_id: Optional[str] = None,
    attempt: Optional[int] = None,
) -> dict:
    """Query parameters for the per-execution log endpoints.

    Covers `GET /logs/{id}`, `/logs/{id}/download` and `/logs/{id}/follow`,
    which all traded their flat parameters for `filters` in Kestra 2.0.
    `DELETE /logs/{id}` kept the flat ones and is not routed through here.
    """
    params: dict[str, Any] = {}

    if await is_v2(client):
        _add(params, "level", "GREATER_THAN_OR_EQUAL_TO", min_level)
        _add(params, "taskId", "EQUALS", task_id)
        _add(params, "taskRunId", "EQUALS", task_run_id)
        if attempt is not None:
            _add(params, "attemptNumber", "EQUALS", attempt)
        return params

    if min_level:
        params["minLevel"] = min_level
    if task_id:
        params["taskId"] = task_id
    if task_run_id:
        params["taskRunId"] = task_run_id
    if attempt is not None:
        params["attempt"] = attempt
    return params


async def logs_search_params(
    client: httpx.AsyncClient,
    *,
    query: Optional[str] = None,
    namespace: Optional[str] = None,
    flow_id: Optional[str] = None,
    trigger_id: Optional[str] = None,
    min_level: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page: Optional[int] = None,
    size: Optional[int] = None,
) -> dict:
    """Query parameters for `GET /logs/search`."""
    params: dict[str, Any] = {}
    if page is not None:
        params["page"] = page
    if size is not None:
        params["size"] = size

    if await is_v2(client):
        _add(params, "q", "EQUALS", query)
        _add(params, "namespace", "EQUALS", namespace)
        _add(params, "flowId", "EQUALS", flow_id)
        _add(params, "triggerId", "EQUALS", trigger_id)
        _add(params, "level", "GREATER_THAN_OR_EQUAL_TO", min_level)
        _add(params, "startDate", "GREATER_THAN_OR_EQUAL_TO", start_date)
        _add(params, "endDate", "LESS_THAN_OR_EQUAL_TO", end_date)
        return params

    if query:
        params["q"] = query
    if namespace:
        params["namespace"] = namespace
    if flow_id:
        params["flowId"] = flow_id
    if trigger_id:
        params["triggerId"] = trigger_id
    if min_level:
        params["minLevel"] = min_level
    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date
    return params


async def namespaces_search_params(
    client: httpx.AsyncClient,
    *,
    query: Optional[str] = None,
    existing: Optional[bool] = None,
    page: Optional[int] = None,
    size: Optional[int] = None,
) -> dict:
    """Query parameters for `GET /namespaces/search`.

    `existing` stayed a flat parameter in 2.x; only the text query moved.
    """
    params: dict[str, Any] = {}
    if page is not None:
        params["page"] = page
    if size is not None:
        params["size"] = size
    if existing is not None:
        params["existing"] = existing

    if await is_v2(client):
        _add(params, "q", "EQUALS", query)
        return params

    if query:
        params["q"] = query
    return params


async def apps_search_params(
    client: httpx.AsyncClient,
    *,
    query: Optional[str] = None,
    namespace: Optional[str] = None,
    flow_id: Optional[str] = None,
    tags: Any = None,
    page: Optional[int] = None,
    size: Optional[int] = None,
    sort: Any = None,
) -> dict:
    """Query parameters for the Enterprise `GET /apps/search`."""
    params: dict[str, Any] = {}
    if page is not None:
        params["page"] = page
    if size is not None:
        params["size"] = size
    if sort:
        params["sort"] = sort

    if await is_v2(client):
        _add(params, "q", "EQUALS", query)
        _add(params, "namespace", "EQUALS", namespace)
        _add(params, "flowId", "EQUALS", flow_id)
        if tags:
            _add(params, "tags", "IN", ",".join(str(t) for t in tags))
        return params

    if query:
        params["q"] = query
    if namespace:
        params["namespace"] = namespace
    if flow_id:
        params["flowId"] = flow_id
    if tags:
        params["tags"] = tags
    return params


# ─── Endpoints whose request shape, not just its parameters, changed ─────────


async def list_kv_keys(client: httpx.AsyncClient, namespace: str) -> list:
    """Every KV entry in `namespace`.

    Kestra 2.0 dropped `GET /namespaces/{namespace}/kv` and lists keys through
    `GET /kv` with the namespace expressed as a filter. That endpoint answers a
    paginated envelope rather than a bare array, and it reports entries from
    child namespaces too, so the results are narrowed back to the namespace
    asked for.
    """
    if not await is_v2(client):
        resp = await client.get(f"/namespaces/{namespace}/kv")
        resp.raise_for_status()
        return resp.json()

    entries: list = []
    page = 1
    while True:
        params = {"page": page, "size": 100}
        _add(params, "namespace", "EQUALS", namespace)
        resp = await client.get("/kv", params=params)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("results", []) if isinstance(data, dict) else data
        if not batch:
            break
        entries.extend(e for e in batch if e.get("namespace", namespace) == namespace)
        if len(batch) < 100:
            break
        page += 1
    return entries


async def trigger_states(client: httpx.AsyncClient, namespace: str) -> dict:
    """Runtime state of every trigger in `namespace`, keyed by `(flowId, triggerId)`.

    A trigger disabled from the UI or through the set-disabled API keeps
    `disabled` unset in the flow source; only `GET /triggers/search` reports it.
    2.x answers `{trigger, state}` items whose `state` also carries `locked` and
    `nextEvaluationDate`. 1.x answers `{abstractTrigger, triggerContext}` items
    with `disabled` and `nextExecutionDate`, and has no `locked` flag. The 1.x
    `namespace` parameter is a prefix, so the results are narrowed back to the
    namespace asked for, as in `list_kv_keys`.

    Each value is `{"disabled": bool, "locked": bool | None, "next_evaluation":
    str | None}`.
    """
    v2 = await is_v2(client)
    states: dict = {}
    size = 100
    page = 1
    while True:
        params: dict[str, Any] = {"page": page, "size": size}
        if v2:
            _add(params, "namespace", "EQUALS", namespace)
        else:
            params["namespace"] = namespace
        resp = await client.get("/triggers/search", params=params)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            batch = data.get("results") or []
            total = data.get("total")
        else:
            batch, total = data or [], None
        for item in batch:
            state = (item.get("state") if v2 else item.get("triggerContext")) or {}
            if state.get("namespace", namespace) != namespace:
                continue
            states[(state.get("flowId"), state.get("triggerId"))] = {
                "disabled": bool(state.get("disabled")),
                "locked": state.get("locked") if v2 else None,
                "next_evaluation": state.get(
                    "nextEvaluationDate" if v2 else "nextExecutionDate"
                ),
            }
        # `total` counts trigger rows before the server drops the ones whose flow
        # was deleted, so a page can come back short, or even empty, while more
        # pages exist. Page by `total`, and by an empty page only without it.
        if total is None:
            if not batch:
                break
        elif page * size >= total:
            break
        page += 1
    return states


async def attach_execution_outputs(
    client: httpx.AsyncClient, execution: Any
) -> Any:
    """Put the flow-level outputs back on an execution object.

    Kestra 1.x carries `outputs` on the execution itself. Kestra 2.0 removed
    them and serves them from `GET /outputs/executions/{id}` instead, so the
    same tool call would otherwise answer with outputs on one version and
    without them on the other. Executions that have produced none, and servers
    that do not expose the endpoint, are left untouched.
    """
    if not isinstance(execution, dict) or execution.get("outputs"):
        return execution
    execution_id = execution.get("id")
    if not execution_id or not await is_v2(client):
        return execution

    try:
        resp = await client.get(f"/outputs/executions/{execution_id}")
        if resp.status_code != 200 or not resp.content:
            return execution
        outputs = resp.json()
    except Exception:
        return execution

    if outputs:
        return {**execution, "outputs": outputs}
    return execution


# `attach_task_outputs` makes one request per task run that has outputs. These
# bound the requests and their parallelism for a large execution, such as a
# ForEach over thousands of items; task runs past the limit are named in a
# warning instead of being read.
TASK_OUTPUTS_LIMIT = 100
_TASK_OUTPUTS_CONCURRENCY = 8


async def attach_task_outputs(client: httpx.AsyncClient, execution: Any) -> Any:
    """Put the task outputs back on the task runs of an execution object.

    Kestra 1.x carries them in `taskRunList[].outputs`. Kestra 2.0 removed them
    from the execution: `GET /outputs/tasks/{id}` lists the task runs that have
    outputs, and `GET /outputs/tasks/{id}/{taskRunId}` serves each of them.
    Without this, every task of an execution read on 2.x looks like it produced
    nothing. Servers that do not expose the endpoint are left untouched, as in
    `attach_execution_outputs` (a 404 on the listing is silent), but a listing
    that fails otherwise, and a task run whose outputs could not be read, are
    reported in a `warning` key rather than shown as having none.
    """
    if not isinstance(execution, dict):
        return execution
    execution_id = execution.get("id")
    task_runs = execution.get("taskRunList")
    if not execution_id or not task_runs or not await is_v2(client):
        return execution
    task_ids = {
        task_run["id"]: task_run.get("taskId")
        for task_run in task_runs
        if isinstance(task_run, dict) and task_run.get("id") and not task_run.get("outputs")
    }
    if not task_ids:
        return execution

    listing_url = f"/outputs/tasks/{execution_id}"

    def listing_failed(reason: str) -> dict:
        return {
            **execution,
            "warning": (
                f"Task outputs could not be listed from GET {listing_url} ({reason}), "
                "so every task run is shown without outputs."
            ),
        }

    try:
        resp = await client.get(listing_url)
    except httpx.HTTPError as e:
        logger.debug("Task outputs of %s could not be listed: %s", execution_id, e)
        return listing_failed(type(e).__name__)
    if resp.status_code == 404:
        # The execution is gone or the server does not expose the endpoint.
        return execution
    if resp.status_code != 200:
        return listing_failed(f"HTTP {resp.status_code}")
    if not resp.content:
        return execution
    try:
        listed = resp.json()
    except ValueError:
        return listing_failed("the response is not JSON")
    if not isinstance(listed, list):
        return listing_failed("the response is not a list")
    wanted = list(
        dict.fromkeys(
            item.get("taskRunId")
            for item in listed
            if isinstance(item, dict) and item.get("taskRunId") in task_ids
        )
    )
    if not wanted:
        return execution

    semaphore = asyncio.Semaphore(_TASK_OUTPUTS_CONCURRENCY)

    async def read(task_run_id: str) -> tuple[str, Any]:
        async with semaphore:
            try:
                resp = await client.get(f"/outputs/tasks/{execution_id}/{task_run_id}")
                if resp.status_code == 200 and resp.content:
                    return task_run_id, resp.json()
                logger.debug("Task outputs of %s answered %s", task_run_id, resp.status_code)
            except (httpx.HTTPError, ValueError) as e:
                logger.debug("Task outputs of %s could not be read: %s", task_run_id, e)
            return task_run_id, None

    read_outputs = await asyncio.gather(*(read(t) for t in wanted[:TASK_OUTPUTS_LIMIT]))
    outputs = {task_run_id: out for task_run_id, out in read_outputs if out is not None}
    result = {
        **execution,
        "taskRunList": [
            {**task_run, "outputs": outputs[task_run["id"]]}
            if isinstance(task_run, dict) and task_run.get("id") in outputs
            else task_run
            for task_run in task_runs
        ],
    }

    warnings = []
    unread = [
        f"{task_ids[task_run_id]} ({task_run_id})"
        for task_run_id, out in read_outputs
        if out is None
    ]
    if unread:
        warnings.append(
            f"The outputs of {', '.join(unread)} could not be read from "
            f"GET /outputs/tasks/{execution_id}/<taskRunId>, so they are shown as empty."
        )
    if len(wanted) > TASK_OUTPUTS_LIMIT:
        warnings.append(
            f"Only the outputs of the first {TASK_OUTPUTS_LIMIT} task runs were read; "
            f"{len(wanted) - TASK_OUTPUTS_LIMIT} more task runs have outputs that are "
            "shown as empty."
        )
    if warnings:
        result["warning"] = " ".join(warnings)
    return result


def normalize_bulk_response(payload: Any) -> Any:
    """Keep `count` present on bulk responses across both majors.

    1.x answers the bulk endpoints with `{"count": n}`. 2.x answers with
    `{"operationId": ..., "totalItems": n}`. Both fields are passed through and
    `count` is mirrored from `totalItems` so callers written against either
    version keep working.
    """
    if isinstance(payload, dict) and "count" not in payload and "totalItems" in payload:
        return {**payload, "count": payload["totalItems"]}
    return payload
