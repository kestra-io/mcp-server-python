"""get_latest_execution must select the newest match, not the newest of the oldest page.

These need no running Kestra. `/executions/search` returns executions oldest-first
unless `sort=state.startDate:desc` is sent, and only one page is read.
"""

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kestra.utils import get_latest_execution  # noqa: E402


def _execution(index: int, state: str = "SUCCESS") -> dict:
    return {
        "id": f"exec-{index:02d}",
        "namespace": "mcpaudit",
        "flowId": "hello",
        "state": {
            "current": state,
            "startDate": f"2026-01-{(index // 60) + 1:02d}T00:{index % 60:02d}:00.000Z",
        },
    }


class _SearchClient:
    """Serves `/executions/search` the way Kestra does: oldest-first, one page."""

    def __init__(self, executions, version="2.0.6"):
        self.base_url = httpx.URL("http://kestra/api/v1/main")
        self._version = version
        self._executions = executions
        self.search_params = None

    async def get(self, url, params=None):
        url = str(url)
        request = httpx.Request("GET", url)
        if url.endswith("/configs"):
            return httpx.Response(
                200, json={"version": self._version, "edition": "OSS"}, request=request
            )
        if not url.endswith("/executions/search"):
            raise AssertionError(url)

        self.search_params = dict(params or {})
        matched = list(self._executions)
        state = self.search_params.get("state") or self.search_params.get(
            "filters[state][EQUALS]"
        )
        if state:
            matched = [e for e in matched if e["state"]["current"] == state]
        # Unrecognised sort values, including the `field,direction` form, are
        # ignored and the default oldest-first order is kept.
        reverse = self.search_params.get("sort") == "state.startDate:desc"
        matched.sort(key=lambda e: e["state"]["startDate"], reverse=reverse)
        size = int(self.search_params.get("size") or 25)
        page = matched[:size]
        key = "results" if self._version.startswith("2") else "content"
        return httpx.Response(200, json={key: page}, request=request)


@pytest.fixture(autouse=True)
def no_forced_version(monkeypatch):
    monkeypatch.delenv("KESTRA_API_VERSION", raising=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["1.3.3", "2.0.6"])
async def test_latest_execution_is_not_the_newest_of_the_oldest_page(version):
    executions = [_execution(i) for i in range(30)]
    client = _SearchClient(executions, version=version)

    latest = await get_latest_execution(client, "mcpaudit", "hello")

    assert latest["id"] == "exec-29"
    assert client.search_params["sort"] == "state.startDate:desc"
    assert client.search_params["size"] == 25


@pytest.mark.asyncio
async def test_latest_execution_respects_state_beyond_the_first_page():
    # Thirty FAILED runs, then a newer SUCCESS. Restart/resume ask for one state.
    executions = [_execution(i, "FAILED") for i in range(30)]
    executions.append(_execution(30, "SUCCESS"))
    client = _SearchClient(executions)

    latest = await get_latest_execution(client, "mcpaudit", "hello", "FAILED")

    assert latest["id"] == "exec-29"
    assert latest["state"]["current"] == "FAILED"


@pytest.mark.asyncio
async def test_latest_execution_is_empty_when_nothing_matches():
    client = _SearchClient([_execution(i) for i in range(30)])

    assert await get_latest_execution(client, "mcpaudit", "hello", "PAUSED") == {}
