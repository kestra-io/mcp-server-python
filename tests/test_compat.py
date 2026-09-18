"""Unit tests for the Kestra 1.x / 2.x request translation.

These need no running Kestra. They pin the request shape the server sends to
each major so that a change to either dialect shows up as a failing assertion
rather than as a 404 or, worse, a silently unfiltered result set.
"""

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kestra import compat  # noqa: E402


class FakeClient:
    """Stands in for httpx.AsyncClient, recording the requests it is given."""

    def __init__(self, version="2.0.1", configs=None, base_url="http://kestra/api/v1/main"):
        self.base_url = httpx.URL(base_url)
        self._version = version
        self._configs = configs
        self.requests: list[tuple[str, dict]] = []

    async def get(self, url, params=None):
        self.requests.append((str(url), params or {}))
        if str(url).endswith("/configs"):
            body = self._configs
            if body is None:
                body = {"version": self._version, "edition": "OSS"}
            return httpx.Response(200, json=body, request=httpx.Request("GET", url))
        return httpx.Response(200, json=[], request=httpx.Request("GET", url))


@pytest.fixture(autouse=True)
def no_forced_version(monkeypatch):
    monkeypatch.delenv("KESTRA_API_VERSION", raising=False)


def v1():
    return FakeClient(version="1.3.3")


def v2():
    return FakeClient(version="2.0.1")


# ─── Version detection ───────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "version,expected",
    [("1.3.3", 1), ("1.2.24", 1), ("2.0.0", 2), ("2.1.0-SNAPSHOT", 2), ("v2.0.1", 2)],
)
async def test_api_major_reads_the_reported_version(version, expected):
    assert await compat.api_major(FakeClient(version=version)) == expected


@pytest.mark.asyncio
async def test_api_major_probes_the_untenanted_configs_endpoint():
    client = v2()
    await compat.api_major(client)
    assert client.requests[0][0] == "http://kestra/api/v1/configs"


@pytest.mark.asyncio
async def test_api_major_is_read_once_per_client():
    client = v2()
    await compat.api_major(client)
    await compat.api_major(client)
    assert len(client.requests) == 1


@pytest.mark.asyncio
async def test_env_override_wins_and_skips_the_probe(monkeypatch):
    monkeypatch.setenv("KESTRA_API_VERSION", "1.3")
    client = v2()
    assert await compat.api_major(client) == 1
    assert client.requests == []


@pytest.mark.asyncio
async def test_a_nonsense_override_is_rejected(monkeypatch):
    monkeypatch.setenv("KESTRA_API_VERSION", "latest")
    with pytest.raises(ValueError, match="KESTRA_API_VERSION"):
        await compat.api_major(v2())


@pytest.mark.asyncio
async def test_an_unreadable_version_falls_back_to_the_newer_major():
    class Broken(FakeClient):
        async def get(self, url, params=None):
            raise httpx.ConnectError("no route to host")

    assert await compat.api_major(Broken()) == 2


# ─── Paths ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action",
    ["pause", "kill", "change-status", "labels", "restart", "resume", "force-run", "state"],
)
async def test_execution_actions_move_under_actions_on_v2(action):
    assert await compat.execution_action_path(v1(), "abc", action) == f"/executions/abc/{action}"
    assert (
        await compat.execution_action_path(v2(), "abc", action)
        == f"/executions/abc/actions/{action}"
    )


@pytest.mark.asyncio
async def test_backfill_path():
    assert await compat.backfill_create_path(v1()) == "/triggers"
    assert await compat.backfill_create_path(v2()) == "/triggers/backfill/create"


@pytest.mark.asyncio
async def test_dashboard_crud_follows_the_instance_config():
    assert await compat.supports_dashboard_crud(v1()) is True
    enabled = FakeClient(configs={"version": "2.0.1", "isCustomDashboardsEnabled": True})
    assert await compat.supports_dashboard_crud(enabled) is True
    disabled = FakeClient(configs={"version": "2.0.1", "isCustomDashboardsEnabled": False})
    assert await compat.supports_dashboard_crud(disabled) is False


# ─── Search parameters ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_executions_search_uses_flat_params_on_v1():
    params = await compat.executions_search_params(
        v1(), namespace="company.team", flow_id="get_data", state="FAILED", size=10
    )
    assert params == {
        "size": 10,
        "namespace": "company.team",
        "flowId": "get_data",
        "state": "FAILED",
    }


@pytest.mark.asyncio
async def test_executions_search_uses_filters_on_v2():
    params = await compat.executions_search_params(
        v2(),
        namespace="company.team",
        flow_id="get_data",
        state="FAILED",
        start_date="2026-01-01T00:00:00Z",
        end_date="2026-02-01T00:00:00Z",
        size=10,
    )
    assert params == {
        "size": 10,
        "filters[namespace][EQUALS]": "company.team",
        "filters[flowId][EQUALS]": "get_data",
        "filters[state][EQUALS]": "FAILED",
        "filters[startDate][GREATER_THAN_OR_EQUAL_TO]": "2026-01-01T00:00:00Z",
        "filters[endDate][LESS_THAN_OR_EQUAL_TO]": "2026-02-01T00:00:00Z",
    }


@pytest.mark.asyncio
async def test_no_legacy_filter_parameter_survives_into_a_v2_request():
    """The 2.x endpoints ignore these rather than rejecting them, which is the
    failure the whole module exists to prevent."""
    params = await compat.executions_search_params(
        v2(),
        namespace="n",
        flow_id="f",
        state="FAILED",
        query="q",
        labels={"env": "prod"},
        start_date="2026-01-01T00:00:00Z",
        end_date="2026-02-01T00:00:00Z",
        time_range="P7D",
    )
    leaked = {k for k in params if not k.startswith("filters[")}
    assert leaked == set()


@pytest.mark.asyncio
async def test_labels_nest_one_level_deeper_on_v2():
    params = await compat.executions_search_params(v2(), labels={"env": "prod"})
    assert params == {"filters[labels][EQUALS][env]": "prod"}
    params = await compat.executions_search_params(v1(), labels={"env": "prod"})
    assert params == {"labels": ["env:prod"]}


@pytest.mark.asyncio
async def test_labels_accept_every_shape_the_tools_pass():
    for labels in ({"env": "prod"}, ["env:prod"], [{"key": "env", "value": "prod"}]):
        assert await compat.executions_search_params(v2(), labels=labels) == {
            "filters[labels][EQUALS][env]": "prod"
        }


@pytest.mark.asyncio
async def test_flows_search_full_text_query():
    assert await compat.flows_search_params(v1(), query="hello") == {"q": "hello"}
    assert await compat.flows_search_params(v2(), query="hello") == {
        "filters[q][EQUALS]": "hello"
    }


@pytest.mark.asyncio
async def test_execution_log_filters():
    assert await compat.execution_logs_params(v1(), min_level="WARN", attempt=0) == {
        "minLevel": "WARN",
        "attempt": 0,
    }
    assert await compat.execution_logs_params(v2(), min_level="WARN", attempt=0) == {
        "filters[level][GREATER_THAN_OR_EQUAL_TO]": "WARN",
        "filters[attemptNumber][EQUALS]": 0,
    }


@pytest.mark.asyncio
async def test_logs_search_filters():
    assert await compat.logs_search_params(
        v2(), namespace="company.team", min_level="ERROR", page=1, size=25
    ) == {
        "page": 1,
        "size": 25,
        "filters[namespace][EQUALS]": "company.team",
        "filters[level][GREATER_THAN_OR_EQUAL_TO]": "ERROR",
    }


@pytest.mark.asyncio
async def test_namespaces_search_keeps_existing_flat_on_both():
    assert await compat.namespaces_search_params(v1(), query="dev", existing=True) == {
        "existing": True,
        "q": "dev",
    }
    assert await compat.namespaces_search_params(v2(), query="dev", existing=True) == {
        "existing": True,
        "filters[q][EQUALS]": "dev",
    }


@pytest.mark.asyncio
async def test_apps_search_filters():
    assert await compat.apps_search_params(v1(), query="a", tags=["x", "y"]) == {
        "q": "a",
        "tags": ["x", "y"],
    }
    assert await compat.apps_search_params(v2(), query="a", tags=["x", "y"]) == {
        "filters[q][EQUALS]": "a",
        "filters[tags][IN]": "x,y",
    }


@pytest.mark.asyncio
async def test_empty_values_are_dropped_rather_than_sent_as_blanks():
    assert await compat.executions_search_params(v2(), namespace="", flow_id=None) == {}
    assert await compat.executions_search_params(v1(), namespace="", flow_id=None) == {}


# ─── Response shapes ─────────────────────────────────────────────────────────


def test_bulk_response_keeps_count_across_both_majors():
    assert compat.normalize_bulk_response({"count": 3}) == {"count": 3}
    assert compat.normalize_bulk_response({"operationId": "x", "totalItems": 3}) == {
        "operationId": "x",
        "totalItems": 3,
        "count": 3,
    }
    assert compat.normalize_bulk_response(None) is None


@pytest.mark.asyncio
async def test_execution_outputs_are_left_alone_on_v1():
    execution = {"id": "abc"}
    assert await compat.attach_execution_outputs(v1(), execution) == execution


@pytest.mark.asyncio
async def test_execution_outputs_are_fetched_on_v2():
    class WithOutputs(FakeClient):
        async def get(self, url, params=None):
            if str(url).endswith("/outputs/executions/abc"):
                self.requests.append((str(url), params or {}))
                return httpx.Response(
                    200, json={"data": "kestra:///x.ion"}, request=httpx.Request("GET", url)
                )
            return await super().get(url, params)

    result = await compat.attach_execution_outputs(WithOutputs(), {"id": "abc"})
    assert result == {"id": "abc", "outputs": {"data": "kestra:///x.ion"}}


@pytest.mark.asyncio
async def test_execution_without_outputs_is_returned_unchanged():
    class NoOutputs(FakeClient):
        async def get(self, url, params=None):
            if str(url).endswith("/outputs/executions/abc"):
                return httpx.Response(404, request=httpx.Request("GET", url))
            return await super().get(url, params)

    assert await compat.attach_execution_outputs(NoOutputs(), {"id": "abc"}) == {"id": "abc"}


@pytest.mark.asyncio
async def test_kv_listing_endpoint_differs_by_major():
    client = v1()
    await compat.list_kv_keys(client, "company.team")
    assert client.requests[-1][0] == "/namespaces/company.team/kv"

    class KvClient(FakeClient):
        async def get(self, url, params=None):
            if str(url) == "/kv":
                self.requests.append((str(url), params or {}))
                return httpx.Response(
                    200,
                    json={"results": [{"namespace": "company.team", "key": "a"}]},
                    request=httpx.Request("GET", url),
                )
            return await super().get(url, params)

    client = KvClient()
    entries = await compat.list_kv_keys(client, "company.team")
    url, params = client.requests[-1]
    assert url == "/kv"
    assert params["filters[namespace][EQUALS]"] == "company.team"
    assert entries == [{"namespace": "company.team", "key": "a"}]


@pytest.mark.asyncio
async def test_kv_listing_excludes_child_namespaces_on_v2():
    class KvClient(FakeClient):
        async def get(self, url, params=None):
            if str(url) == "/kv":
                return httpx.Response(
                    200,
                    json={
                        "results": [
                            {"namespace": "company.team", "key": "a"},
                            {"namespace": "company.team.sub", "key": "b"},
                        ]
                    },
                    request=httpx.Request("GET", url),
                )
            return await super().get(url, params)

    entries = await compat.list_kv_keys(KvClient(), "company.team")
    assert [e["key"] for e in entries] == ["a"]
