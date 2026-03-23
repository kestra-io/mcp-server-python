import pytest
import pytest_asyncio
import json
import asyncio
from fastmcp import Client
from test_utils import mcp_server_config


class ResourceTracker:
    """Tracks resources created during a test for automatic cleanup."""

    def __init__(self):
        self.flows: list[tuple[str, str]] = []  # (namespace, flow_id)
        self.execution_ids: list[str] = []

    def track_flow(self, namespace: str, flow_id: str):
        if (namespace, flow_id) not in self.flows:
            self.flows.append((namespace, flow_id))

    def track_execution(self, execution_id: str):
        if execution_id not in self.execution_ids:
            self.execution_ids.append(execution_id)

    async def cleanup(self, client: Client):
        """Delete all tracked resources. Order: kill executions → delete executions → delete flows."""
        # 1. Kill any running/paused executions
        for exec_id in self.execution_ids:
            try:
                result = await client.call_tool(
                    "manage_executions", {"action": "get", "execution_id": exec_id}
                )
                state = json.loads(result.content[0].text).get("state", {}).get("current", "")
                if state in ("RUNNING", "PAUSED", "CREATED", "QUEUED"):
                    await client.call_tool(
                        "manage_executions", {"action": "kill", "execution_id": exec_id}
                    )
            except Exception:
                pass

        # 2. Brief wait for executions to reach terminal state
        if self.execution_ids:
            await asyncio.sleep(1)

        # 3. Delete executions
        for exec_id in self.execution_ids:
            try:
                await client.call_tool(
                    "manage_executions", {"action": "delete", "execution_id": exec_id}
                )
            except Exception:
                pass

        # 4. Delete flows
        for namespace, flow_id in self.flows:
            try:
                await client.call_tool(
                    "manage_flow",
                    {"action": "delete", "namespace": namespace, "flow_id": flow_id},
                )
            except Exception:
                pass


@pytest_asyncio.fixture
async def kestra_client():
    """Function-scoped MCP client."""
    async with Client(mcp_server_config) as client:
        yield client


@pytest_asyncio.fixture
async def cleanup(kestra_client):
    """Function-scoped resource tracker that cleans up after each test."""
    tracker = ResourceTracker()
    yield tracker
    await tracker.cleanup(kestra_client)
