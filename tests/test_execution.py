import pytest
from fastmcp import Client
from dotenv import load_dotenv
import json
from pathlib import Path
from test_utils import mcp_server_config, create_flow, poll_for_execution
import time
from datetime import datetime, timezone
import asyncio

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env", override=True)


@pytest.mark.asyncio
async def test_execute_flow():
    """Test executing a flow with inputs and outputs."""
    async with Client(mcp_server_config) as client:
        response_json = await create_flow("app_get_data_flow.yaml", client)
        print("Created base flow:", response_json)
        flow_id = response_json.get("id", "")
        namespace = response_json.get("namespace", "")
        base_revision = response_json.get("revision", 1)

        # Test case 1: Execute flow with default inputs
        result = await client.call_tool(
            "execute_flow",
            {"namespace": namespace, "flow_id": flow_id},
        )
        response_json = json.loads(result[0].text) if result and result[0].text else {}
        print(response_json)
        assert response_json["url"] is not None
        assert response_json["inputs"]["data"] == "customers"
        assert response_json["inputs"]["startDate"] == "2025-12-03"

        # Test case 2: Execute flow with custom inputs
        result = await client.call_tool(
            "execute_flow",
            {
                "namespace": namespace,
                "flow_id": flow_id,
                "inputs": {"data": "products", "startDate": "2026-01-01"},
            },
        )
        response_json = json.loads(result[0].text) if result and result[0].text else {}
        print("Execute flow with custom inputs response:", response_json)
        assert response_json["url"] is not None
        assert response_json["inputs"]["data"] == "products"
        assert response_json["inputs"]["startDate"] == "2026-01-01"

        # Test case 3: Execute flow with custom labels
        result = await client.call_tool(
            "execute_flow",
            {
                "namespace": namespace,
                "flow_id": flow_id,
                "labels": [
                    "environment:test",
                    "purpose:integration_test",
                    "priority:high",
                ],
            },
        )
        response_json = json.loads(result[0].text) if result and result[0].text else {}
        print("Execute flow with custom labels response:", response_json)
        assert response_json["url"] is not None
        assert "labels" in response_json
        assert isinstance(response_json["labels"], list)

        # Convert response labels to a dict for easier checking
        response_labels = {
            label["key"]: label["value"] for label in response_json["labels"]
        }
        assert response_labels["environment"] == "test"
        assert response_labels["purpose"] == "integration_test"
        assert response_labels["priority"] == "high"

        # Test case 4: Execute flow with scheduled date
        # Schedule for 1 hour from now
        future_time = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 3600)
        )
        result = await client.call_tool(
            "execute_flow",
            {"namespace": namespace, "flow_id": flow_id, "schedule_date": future_time},
        )
        response_json = json.loads(result[0].text) if result and result[0].text else {}
        print("Execute flow with scheduled date response:", response_json)
        assert response_json["url"] is not None
        assert "state" in response_json
        assert response_json["state"]["current"] == "CREATED"
        assert "scheduleDate" in response_json
        assert response_json["scheduleDate"] == future_time
        assert "metadata" in response_json
        assert "originalCreatedDate" in response_json["metadata"]

        # Test case 5: Execute flow with wait=True
        # Update the flow to add the log task
        response_json = await create_flow("get_data_updated.yaml", client)
        print("Updated flow:", response_json)
        updated_revision = response_json.get("revision", 2)

        # Execute the updated version
        result = await client.call_tool(
            "execute_flow",
            {"namespace": namespace, "flow_id": flow_id, "wait": True},
        )
        response_json = json.loads(result[0].text) if result and result[0].text else {}
        print("Execute flow with wait=True response:", response_json)
        assert response_json["url"] is not None
        assert "state" in response_json
        assert response_json["state"]["current"] in ["SUCCESS", "FAILED", "WARNING"]

        # Verify that the taskRunList has two tasks: extract and log
        assert "taskRunList" in response_json
        assert len(response_json["taskRunList"]) == 2
        assert "outputs" in response_json
        assert "data" in response_json["outputs"]
        task_run = response_json["taskRunList"][0]
        assert "attempts" in task_run
        assert len(task_run["attempts"]) > 0
        assert "outputs" in task_run
        assert len(response_json["state"]["histories"]) > 1
        assert any(h["state"] == "RUNNING" for h in response_json["state"]["histories"])
        assert any(h["state"] == "SUCCESS" for h in response_json["state"]["histories"])

        # Test case 6: Execute flow with specific revision (the base version)
        result = await client.call_tool(
            "execute_flow",
            {
                "namespace": namespace,
                "flow_id": flow_id,
                "revision": base_revision,
                "wait": True,
            },
        )
        response_json = json.loads(result[0].text) if result and result[0].text else {}
        print("Execute flow with base revision response:", response_json)
        assert response_json["url"] is not None
        assert "taskRunList" in response_json
        # Verify that only the extract task exists (no log task)
        assert len(response_json["taskRunList"]) == 1
        assert response_json["taskRunList"][0]["taskId"] == "extract"

        # Test case 7: Add labels to execution
        execution_id = response_json["id"]
        result = await client.call_tool(
            "add_execution_labels",
            {
                "execution_id": execution_id,
                "labels": [
                    {"key": "revision", "value": "1"},
                    {"key": "reason", "value": "test"},
                ],
            },
        )
        response_json = json.loads(result[0].text) if result and result[0].text else {}
        print("Add execution labels response:", response_json)
        assert "labels" in response_json
        # Convert response labels to a dict for easier checking
        response_labels = {
            label["key"]: label["value"] for label in response_json["labels"]
        }
        assert response_labels["revision"] == "1"
        assert response_labels["reason"] == "test"

        # Test case 8: Execute flow with invalid input
        with pytest.raises(Exception):
            await client.call_tool(
                "execute_flow",
                {
                    "namespace": namespace,
                    "flow_id": flow_id,
                    "inputs": {"data": "invalid_data_type"},
                },
            )

        # Test case 9: Delete the flow
        result = await client.call_tool(
            "manage_flow",
            {"action": "delete", "namespace": namespace, "flow_id": flow_id},
        )
        response_json = json.loads(result[0].text) if result and result[0].text else {}
        print("Delete flow response:", response_json)

        # Verify flow was deleted by attempting to get it
        with pytest.raises(Exception):
            await client.call_tool(
                "manage_flow",
                {"action": "get_yaml", "namespace": namespace, "flow_id": flow_id},
            )


@pytest.mark.asyncio
async def test_list_executions():
    """Test listing executions."""
    async with Client(mcp_server_config) as client:
        # Test case 1: List all executions for a specific flow
        result = await client.call_tool(
            "list_executions", {"namespace": "company.team", "flow_id": "get_data"}
        )
        response_json = json.loads(result[0].text) if result and result[0].text else {}
        print("List executions for flow response:", response_json)
        # Handle both single execution and list of executions
        executions = (
            response_json.get("results", [response_json])
            if isinstance(response_json, dict) and response_json
            else response_json
        )
        assert isinstance(executions, list)
        for execution in executions:
            assert execution["namespace"] == "company.team"
            assert execution["flowId"] == "get_data"

        # Test case 2: List two recent executions within last 15 minutes
        result = await client.call_tool(
            "list_executions",
            {
                "namespace": "company.team",
                "flow_id": "get_data",
                "count": 2,
                "minutes": 15,
            },
        )
        response_json = json.loads(result[0].text) if result and result[0].text else {}
        print("List recent executions response:", response_json)
        # Handle empty response case
        if not response_json:
            executions = []
        else:
            executions = (
                response_json.get("results", [response_json])
                if isinstance(response_json, dict)
                else response_json
            )
        assert isinstance(executions, list)
        assert len(executions) <= 2  # Should return at most 2 executions

        # Only check execution details if we have results
        if executions:
            for execution in executions:
                assert execution["namespace"] == "company.team"
                assert execution["flowId"] == "get_data"
                # Verify execution is within last 15 minutes
                start_date = datetime.fromisoformat(
                    execution["state"]["startDate"].replace("Z", "+00:00")
                )
                time_diff = datetime.now(timezone.utc) - start_date
                assert time_diff.total_seconds() <= 15 * 60  # 15 minutes in seconds

        # Test case 3: List latest execution
        result = await client.call_tool(
            "list_executions",
            {"namespace": "company.team", "flow_id": "get_data", "count": 1},
        )
        response_json = json.loads(result[0].text) if result and result[0].text else {}
        print("List latest execution response:", response_json)
        # Handle empty response case
        if not response_json:
            executions = []
        else:
            executions = (
                response_json.get("results", [response_json])
                if isinstance(response_json, dict)
                else response_json
            )
        assert isinstance(executions, list)
        assert len(executions) == 1  # Should return exactly 1 execution
        execution = executions[0]
        assert execution["namespace"] == "company.team"
        assert execution["flowId"] == "get_data"
        # Verify this is the latest execution by comparing with previous results
        if len(executions) > 0:  # If we have previous executions to compare with
            latest_start = datetime.fromisoformat(
                execution["state"]["startDate"].replace("Z", "+00:00")
            )
            for prev_exec in executions[1:]:
                prev_start = datetime.fromisoformat(
                    prev_exec["state"]["startDate"].replace("Z", "+00:00")
                )
                assert (
                    latest_start >= prev_start
                )  # Latest execution should be newer than or equal to others


@pytest.mark.asyncio
async def test_manage_executions():
    """Test managing an execution."""
    async with Client(mcp_server_config) as client:
        subflow_response_json = await create_flow("hello_mcp.yaml", client)
        print(subflow_response_json)
        response_json = await create_flow("sleeper.yaml", client)
        print(response_json)
        flow_id = response_json.get("id", "")
        namespace = response_json.get("namespace", "")

        # Test case 1: Execute flow with default inputs
        result = await client.call_tool(
            "execute_flow",
            {"namespace": namespace, "flow_id": flow_id},
        )
        response_json = json.loads(result[0].text) if result and result[0].text else {}
        print("Execute flow response:", response_json)
        execution_id = response_json["id"]

        # Test case 2: Get execution details and wait for RUNNING state
        response_json = await poll_for_execution(
            client,
            execution_id,
            desired_state="RUNNING",
            max_retries=5,
            retry_interval=1,
        )

        # Test case 3: Pause execution (only if it's RUNNING)
        if response_json["state"]["current"] == "RUNNING":
            result = await client.call_tool(
                "manage_executions", {"action": "pause", "execution_id": execution_id}
            )
            response_json = (
                json.loads(result[0].text) if result and result[0].text else {}
            )
            print("Pause execution response:", response_json)
            assert response_json["status"] == "paused"

        # Test case 4: Kill execution with cascade=True
        result = await client.call_tool(
            "manage_executions",
            {"action": "kill", "execution_id": execution_id, "cascade": True},
        )
        response_json = json.loads(result[0].text) if result and result[0].text else {}
        print("Kill execution response:", response_json)
        assert response_json["executionId"] == execution_id
        assert response_json["status"] in ["kill_requested", "already_finished"]

        # Test case 5: Execute get_data flow and change its status to WARNING
        result = await client.call_tool(
            "execute_flow",
            {"namespace": "company.team", "flow_id": "get_data", "wait": True},
        )
        response_json = json.loads(result[0].text) if result and result[0].text else {}
        print("Execute get_data flow response:", response_json)
        assert response_json["url"] is not None
        assert response_json["state"]["current"] in ["SUCCESS", "FAILED", "WARNING"]

        # Change status to WARNING
        result = await client.call_tool(
            "manage_executions",
            {
                "action": "change_status",
                "execution_id": response_json["id"],
                "status": "WARNING",
            },
        )
        response_json = json.loads(result[0].text) if result and result[0].text else {}
        print("Change status to WARNING response:", response_json)
        assert response_json["id"] == response_json["id"]
        assert response_json["state"]["current"] == "WARNING"

        # Try to change status to INVALID state
        with pytest.raises(Exception):
            await client.call_tool(
                "manage_executions",
                {
                    "action": "change_status",
                    "execution_id": response_json["id"],
                    "status": "INVALID",
                },
            )


@pytest.mark.asyncio
async def test_execute_flow_with_subflow():  # TODO
    """Test executing a flow with a subflow and killing the parent execution with cascade=False."""
    async with Client(mcp_server_config) as client:
        subflow_response_json = await create_flow("hello_mcp.yaml", client)
        print(f"Subflow hello_mcp response: {subflow_response_json}")
        response_json = await create_flow("sleeper.yaml", client)
        print(f"Parent sleeper response: {response_json}")
        result = await client.call_tool(
            "execute_flow",
            {"namespace": "company.team", "flow_id": "sleeper"},
        )
        response_json = json.loads(result[0].text) if result and result[0].text else {}
        print("Execute flow with subflow response:", response_json)
        parent_execution_id = response_json["id"]
        correlation_id = None
        for label in response_json["labels"]:
            if label["key"] == "system.correlationId":
                correlation_id = label["value"]
                break

        assert (
            correlation_id is not None
        ), "Could not find correlation ID in parent execution"

        # Find the subflow execution using the correlation ID
        max_retries = 5
        retry_count = 0
        subflow_execution_id = None

        while retry_count < max_retries and subflow_execution_id is None:
            result = await client.call_tool(
                "list_executions",
                {"namespace": "company.team", "flow_id": "hello_mcp", "minutes": 2},
            )
            response_json = (
                json.loads(result[0].text) if result and result[0].text else {}
            )
            print(
                f"List executions response (attempt {retry_count + 1}):", response_json
            )

            # Handle both single execution and list of executions
            executions = (
                [response_json] if isinstance(response_json, dict) else response_json
            )

            # Find the execution with matching correlation ID
            for execution in executions:
                if "labels" in execution:
                    for label in execution["labels"]:
                        if (
                            label["key"] == "system.correlationId"
                            and label["value"] == correlation_id
                        ):
                            subflow_execution_id = execution["id"]
                            break
                    if subflow_execution_id:
                        break

            if subflow_execution_id is None:
                retry_count += 1
                if retry_count < max_retries:
                    print(
                        f"Subflow execution not found, retrying in 2 seconds... (attempt {retry_count + 1}/{max_retries})"
                    )
                    await asyncio.sleep(2)

        assert (
            subflow_execution_id is not None
        ), "Could not find subflow execution with matching correlation ID"

        # Kill parent execution with cascade=False
        result = await client.call_tool(
            "manage_executions",
            {"action": "kill", "execution_id": parent_execution_id, "cascade": False},
        )
        response_json = json.loads(result[0].text) if result and result[0].text else {}
        print("Kill parent execution response:", response_json)
        assert response_json["executionId"] == parent_execution_id
        assert response_json["status"] in ["kill_requested", "already_finished"]

        # Wait for parent execution to be KILLED
        response_json = await poll_for_execution(
            client,
            parent_execution_id,
            desired_state="KILLED",
            max_retries=10,
            retry_interval=1,
        )

        assert (
            response_json["state"]["current"] == "KILLED"
        ), "Parent execution was not killed"

        # Verify subflow execution is still running
        result = await client.call_tool(
            "manage_executions", {"action": "get", "execution_id": subflow_execution_id}
        )
        response_json = json.loads(result[0].text) if result and result[0].text else {}
        print("Get subflow execution response:", response_json)
        # The subflow execution should be in a non-terminal state
        assert response_json["state"]["current"] not in [
            "KILLED",
            "FAILED",
            "SUCCESS",
            "WARNING",
            "CANCELLED",
            "SKIPPED",
        ]


@pytest.mark.asyncio
async def test_delete_execution():
    """Test deleting an execution."""
    async with Client(mcp_server_config) as client:
        response_json = await create_flow("healthcheck.yaml", client)
        print("Created healthcheck flow:", response_json)
        flow_id = response_json.get("id", "")
        namespace = response_json.get("namespace", "")

        result = await client.call_tool(
            "execute_flow",
            {"namespace": namespace, "flow_id": flow_id},
        )
        response_json = json.loads(result[0].text) if result and result[0].text else {}
        print("Execute flow response:", response_json)
        execution_id = response_json["id"]

        # wait for the execution to be in a terminal state
        response_json = await poll_for_execution(
            client, execution_id, max_retries=10, retry_interval=1
        )

        # delete the execution
        result = await client.call_tool(
            "manage_executions", {"action": "delete", "execution_id": execution_id}
        )
        response_json = json.loads(result[0].text) if result and result[0].text else {}
        print("Delete execution response:", response_json)

        # Verify execution was deleted by attempting to get it
        with pytest.raises(Exception):
            await client.call_tool(
                "manage_executions", {"action": "get", "execution_id": execution_id}
            )


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_execute_flow())
    asyncio.run(test_execute_flow_with_subflow())
    asyncio.run(test_list_executions())
    asyncio.run(test_manage_executions())
    asyncio.run(test_delete_execution())
