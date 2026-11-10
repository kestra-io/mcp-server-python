"""Offline tests for `search_executions`.

The tool is registered on an in-memory FastMCP server whose httpx client is
backed by a MockTransport, so the request it sends and the response it builds
can be checked without a running Kestra. `test_execution.py` covers the same
tool against a real server.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from fastmcp import Client, FastMCP

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kestra.constants import _SEARCHABLE_STATES, _VALID_STATES  # noqa: E402
from tools import execution  # noqa: E402
from tools.execution import (  # noqa: E402
    _failed_task_ids,
    _resolve_start_date,
    _search_summary,
    _validate_states,
)


def _execution(id_, state="FAILED", task_runs=None, labels=None, inputs=None):
    e = {
        "id": id_,
        "namespace": "company.team",
        "flowId": "etl",
        "state": {
            "current": state,
            "startDate": "2026-11-10T10:00:00Z",
            "endDate": "2026-11-10T10:00:05Z",
            "duration": "PT5S",
            "histories": [{"state": "CREATED"}],
        },
        "labels": labels or [],
    }
    if task_runs is not None:
        e["taskRunList"] = task_runs
    if inputs is not None:
        e["inputs"] = inputs
    return e


def _task_run(task_id, state):
    return {
        "id": f"tr-{task_id}-{state}",
        "taskId": task_id,
        "state": {"current": state},
    }


class FakeKestra:
    """Answers `/executions/search` with a fixed page and `/executions/{id}`
    from a dict, recording every request."""

    def __init__(self, results, total, details=None, detail_status=200):
        self.results = results
        self.total = total
        self.details = details or {}
        self.detail_status = detail_status
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path.removeprefix("/api/v1/main")
        if path == "/executions/search":
            page = int(request.url.params.get("page", 1))
            size = int(request.url.params.get("size", 10))
            results = self.results
            total = self.total
            if total is not None and (page - 1) * size >= total > 0:
                # What Kestra answers past the last page.
                results, total = [], 0
            elif page == 1 and size < len(results):
                results = results[:size]
            body = {"results": results}
            if total is not None:
                body["total"] = total
            return httpx.Response(200, json=body)
        if path.startswith("/executions/"):
            if self.detail_status != 200:
                return httpx.Response(self.detail_status, json={})
            return httpx.Response(200, json=self.details[path.rsplit("/", 1)[1]])
        return httpx.Response(404, json={})

    def search_params(self) -> list[tuple[str, str]]:
        search = [r for r in self.requests if r.url.path.endswith("/executions/search")]
        assert len(search) == 1, "exactly one search request per call"
        return list(search[0].url.params.multi_items())


async def _call(fake: FakeKestra, **arguments) -> dict:
    mcp = FastMCP("test")
    client = httpx.AsyncClient(
        base_url="http://kestra/api/v1/main",
        transport=httpx.MockTransport(fake.handler),
    )
    execution.register_execution_tools(mcp, client)
    async with Client(mcp) as c:
        result = await c.call_tool("search_executions", arguments)
    await client.aclose()
    return json.loads(result.content[0].text)


# ─── Request ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sends_one_filtered_sorted_page_request():
    fake = FakeKestra(results=[_execution("e1", "SUCCESS")], total=100)
    await _call(
        fake,
        state=["failed", "KILLED"],
        namespace="company",
        include_child_namespaces=True,
        flow_id="etl",
        labels={"env": "prod"},
        start_date="2026-11-01T00:00:00Z",
        limit=10,
        page=3,
    )
    params = dict(fake.search_params())
    assert params == {
        "page": "3",
        "size": "10",
        "sort": "state.startDate:desc",
        "filters[state][IN]": "FAILED,KILLED",
        "filters[namespace][PREFIX]": "company",
        "filters[flowId][EQUALS]": "etl",
        "filters[startDate][GREATER_THAN_OR_EQUAL_TO]": "2026-11-01T00:00:00Z",
        "filters[labels][EQUALS][env]": "prod",
    }


@pytest.mark.asyncio
async def test_defaults_search_every_namespace_over_the_last_90_days():
    fake = FakeKestra(results=[], total=0)
    await _call(fake)
    params = dict(fake.search_params())
    assert not any(k.startswith("filters[namespace]") for k in params)
    assert not any(k.startswith("filters[state]") for k in params)
    assert params["size"] == "25" and params["page"] == "1"
    assert "filters[startDate][GREATER_THAN_OR_EQUAL_TO]" in params


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [{"limit": 0}, {"limit": 101}, {"page": 0}])
async def test_rejects_limit_and_page_outside_the_server_bounds(arguments):
    fake = FakeKestra(results=[], total=0)
    with pytest.raises(Exception):
        await _call(fake, **arguments)
    assert fake.requests == []


@pytest.mark.asyncio
async def test_limit_at_its_bounds_is_accepted():
    for limit in (1, 100):
        fake = FakeKestra(results=[], total=0)
        await _call(fake, limit=limit)
        assert dict(fake.search_params())["size"] == str(limit)


@pytest.mark.asyncio
async def test_rejects_an_unknown_state_before_any_request():
    fake = FakeKestra(results=[], total=0)
    with pytest.raises(Exception, match="BROKEN"):
        await _call(fake, state=["FAILED", "BROKEN"])
    assert fake.requests == []


# ─── has_more at the page boundary ───────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "page,limit,total,expected",
    [
        (1, 2, 1, False),  # fewer matches than a page
        (1, 2, 2, False),  # exactly one full page
        (1, 2, 3, True),  # one past a full page
        (2, 2, 3, False),  # the last, short page
        (2, 2, 4, False),  # the last page, exactly full
        (2, 2, 5, True),
        (5, 2, 3, False),  # past the end: empty and nothing more
    ],
)
async def test_has_more_is_true_only_while_matches_remain_on_later_pages(
    page, limit, total, expected
):
    shown = max(0, min(limit, total - (page - 1) * limit))
    fake = FakeKestra(
        results=[_execution(f"e{i}", "SUCCESS") for i in range(shown)], total=total
    )
    out = await _call(fake, page=page, limit=limit)
    assert (out["total"], out["page"], out["limit"], out["has_more"]) == (
        total,
        page,
        limit,
        expected,
    )
    assert len(out["results"]) == shown


@pytest.mark.asyncio
async def test_a_missing_total_is_derived_from_the_page_so_has_more_stays_false():
    fake = FakeKestra(results=[_execution("e1", "SUCCESS")], total=None)
    out = await _call(fake, page=3, limit=2)
    assert out["total"] == 5 and out["has_more"] is False


@pytest.mark.asyncio
async def test_past_the_last_page_the_real_total_is_reported_not_zero():
    """Kestra answers `total: 0` past the last page; the tool re-reads it."""
    fake = FakeKestra(results=[_execution("e1", "SUCCESS")], total=2)
    out = await _call(fake, page=3, limit=1)
    assert (out["total"], out["results"], out["has_more"]) == (2, [], False)
    second = [r for r in fake.requests if r.url.path.endswith("/search")][1]
    assert (second.url.params["page"], second.url.params["size"]) == ("1", "1")


@pytest.mark.asyncio
async def test_a_page_within_range_is_one_request():
    fake = FakeKestra(results=[_execution("e1", "SUCCESS")], total=2)
    await _call(fake, page=2, limit=1)
    assert len(fake.requests) == 1


@pytest.mark.asyncio
async def test_an_empty_first_page_is_not_reread():
    fake = FakeKestra(results=[], total=0)
    out = await _call(fake)
    assert out["total"] == 0 and len(fake.requests) == 1


# ─── failedTasks ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_failed_tasks_come_from_the_search_result_on_1x():
    """1.x search results carry taskRunList, so no execution is read again."""
    fake = FakeKestra(
        results=[
            _execution(
                "f1",
                task_runs=[
                    _task_run("extract", "SUCCESS"),
                    _task_run("load", "FAILED"),
                ],
            ),
            _execution("s1", "SUCCESS", task_runs=[_task_run("extract", "SUCCESS")]),
        ],
        total=2,
    )
    out = await _call(fake)
    assert out["results"][0]["failedTasks"] == ["load"]
    assert "failedTasks" not in out["results"][1]
    assert len(fake.requests) == 1
    assert "warning" not in out


@pytest.mark.asyncio
async def test_failed_tasks_are_read_per_failed_execution_on_2x():
    """2.x search results have no taskRunList; only FAILED executions are read."""
    fake = FakeKestra(
        results=[_execution("f1"), _execution("s1", "SUCCESS"), _execution("f2")],
        total=3,
        details={
            "f1": _execution("f1", task_runs=[_task_run("load", "FAILED")]),
            "f2": _execution(
                "f2", task_runs=[_task_run("a", "FAILED"), _task_run("b", "FAILED")]
            ),
        },
    )
    out = await _call(fake)
    assert [r.get("failedTasks") for r in out["results"]] == [
        ["load"],
        None,
        ["a", "b"],
    ]
    read = sorted(r.url.path.rsplit("/", 1)[1] for r in fake.requests[1:])
    assert read == ["f1", "f2"]


@pytest.mark.asyncio
async def test_unreadable_failed_tasks_are_reported_not_shown_as_empty():
    fake = FakeKestra(results=[_execution("f1")], total=1, detail_status=500)
    out = await _call(fake)
    assert "failedTasks" not in out["results"][0]
    assert "f1" in out["warning"]


def test_failed_task_ids_keep_run_order_and_drop_repeats():
    # A task retried in a loop fails once per iteration but is one failed task.
    e = _execution(
        "x",
        task_runs=[
            _task_run("each", "FAILED"),
            _task_run("ok", "SUCCESS"),
            _task_run("load", "FAILED"),
            _task_run("each", "FAILED"),
            _task_run("warn", "WARNING"),
        ],
    )
    assert _failed_task_ids(e) == ["each", "load"]
    assert _failed_task_ids(_execution("y")) == []


# ─── Result shape ────────────────────────────────────────────────────────────


def test_summary_flattens_labels_and_drops_system_ones():
    e = _execution(
        "x",
        labels=[
            {"key": "env", "value": "prod"},
            {"key": "system.correlationId", "value": "abc"},
        ],
        inputs={"day": "2026-11-10"},
    )
    summary = _search_summary(e, include_inputs=False)
    assert summary["labels"] == {"env": "prod"}
    assert "inputs" not in summary
    assert "histories" not in summary["state"]
    assert summary["state"]["duration"] == "PT5S"
    assert _search_summary(e, include_inputs=True)["inputs"] == {"day": "2026-11-10"}


def test_summary_omits_labels_when_only_system_ones_exist():
    e = _execution("x", labels=[{"key": "system.from", "value": "api"}])
    assert "labels" not in _search_summary(e, include_inputs=False)


# The `State.Type` enum of Kestra 1.3 and 2.0, read from their class files.
# 1.0 lacks SUBMITTED and RESUBMITTED and answers a search for them with a 422.
KESTRA_STATES = [
    "BREAKPOINT",
    "CANCELLED",
    "CREATED",
    "FAILED",
    "KILLED",
    "KILLING",
    "PAUSED",
    "QUEUED",
    "RESTARTED",
    "RESUBMITTED",
    "RETRIED",
    "RETRYING",
    "RUNNING",
    "SKIPPED",
    "SUBMITTED",
    "SUCCESS",
    "WARNING",
]


def test_searchable_states_are_exactly_kestras_states():
    assert sorted(_SEARCHABLE_STATES) == KESTRA_STATES


def test_change_status_validation_is_not_widened():
    # The server only accepts terminal states (and RESUBMITTED) there, so the
    # set gating `change_status` stays as it was.
    assert {"SUBMITTED", "BREAKPOINT", "RESUBMITTED"}.isdisjoint(_VALID_STATES)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", KESTRA_STATES)
async def test_every_kestra_state_is_searchable(state):
    fake = FakeKestra(results=[], total=0)
    await _call(fake, state=[state.lower()])
    assert dict(fake.search_params())["filters[state][IN]"] == state


@pytest.mark.asyncio
async def test_a_rejected_search_reports_kestras_reason():
    class Rejecting(FakeKestra):
        def handler(self, request):
            self.requests.append(request)
            return httpx.Response(
                422,
                json={
                    "message": "Illegal argument: No enum constant State.Type.SUBMITTED"
                },
            )

    with pytest.raises(Exception, match="No enum constant.*SUBMITTED"):
        await _call(Rejecting(results=[], total=0), state=["SUBMITTED"])


@pytest.mark.asyncio
async def test_all_states_at_once_are_sent_as_one_in_value():
    fake = FakeKestra(results=[], total=0)
    await _call(fake, state=KESTRA_STATES)
    sent = dict(fake.search_params())["filters[state][IN]"]
    assert sent.split(",") == KESTRA_STATES


def test_validate_states_normalizes_and_deduplicates():
    assert _validate_states([" failed", "FAILED", "killed"]) == ["FAILED", "KILLED"]
    assert _validate_states(None) == []
    assert _validate_states([""]) == []
    with pytest.raises(ValueError, match="DONE"):
        _validate_states(["DONE"])


# ─── Date window ─────────────────────────────────────────────────────────────

NOW = datetime(2026, 11, 10, 18, 30, 15, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "time_range,expected",
    [
        ("P1D", "2026-11-09T00:00:00Z"),  # day-only: snaps to midnight
        ("P10D", "2026-10-31T00:00:00Z"),
        ("PT1H", "2026-11-10T17:30:15Z"),  # sub-day: exact
        ("PT5M", "2026-11-10T18:25:15Z"),
        ("P1DT1H", "2026-11-09T17:30:15Z"),  # mixed: exact, no snap
        (
            "PT24H",
            "2026-11-09T00:00:00Z",
        ),  # whole days snap, as list_executions always did
        ("PT48H", "2026-11-08T00:00:00Z"),
        ("", "2026-08-12T00:00:00Z"),  # default 90 days
    ],
)
def test_start_date_window(time_range, expected):
    assert _resolve_start_date("", time_range, now=NOW) == expected


def test_explicit_start_date_wins_over_time_range():
    assert (
        _resolve_start_date("2026-01-01T00:00:00Z", "PT5M", now=NOW)
        == "2026-01-01T00:00:00Z"
    )


def test_unsupported_duration_is_rejected():
    with pytest.raises(ValueError, match="P1W"):
        _resolve_start_date("", "P1W", now=NOW)
