from mcp.server.fastmcp import FastMCP
import httpx
from typing import Annotated
from pydantic import Field
from kestra.utils import get_latest_execution


def register_restart_tools(mcp: FastMCP, client: httpx.AsyncClient) -> None:
    @mcp.tool()
    async def restart_execution(
        namespace: Annotated[str, Field(description="The namespace of the flow")],
        flow_id: Annotated[str, Field(description="The ID of the flow")],
        execution_id: Annotated[
            str,
            Field(
                description="The ID of the execution to restart. If not provided, the latest failed execution for the given namespace and flow_id is restarted."
            ),
        ] = None,
        revision: Annotated[
            int,
            Field(
                description="The revision of the flow to restart. If not provided, the same revision as the original execution is used."
            ),
        ] = 0,
    ) -> dict:
        """
        Restart a failed execution. If execution_id is provided, restart that execution. If not, retrieve the latest failed execution for the given namespace and flow_id and restart it.
        Returns the restarted execution's JSON metadata.
        """
        if not execution_id:
            latest = await get_latest_execution(client, namespace, flow_id, "FAILED")
            execution_id = latest.get("id")
            if not execution_id:
                raise ValueError(
                    f"No failed executions found for {namespace}/{flow_id}."
                )
            current_state = latest.get("state", {}).get("current")
        else:
            get_resp = await client.get(f"/executions/{execution_id}")
            get_resp.raise_for_status()
            exec_data = get_resp.json()
            current_state = exec_data.get("state", {}).get("current")

        if current_state != "FAILED":
            raise ValueError(
                f"Execution `{execution_id}` is in state `{current_state}`, which cannot be restarted. "
                "Only executions in the FAILED state can be restarted."
            )

        params: dict[str, int] = {}
        if revision:
            params["revision"] = revision

        resp = await client.post(f"/executions/{execution_id}/restart", params=params)
        resp.raise_for_status()
        return resp.json()

    @mcp.tool()
    async def change_taskrun_state(
        execution_id: Annotated[
            str, Field(description="The ID of the execution to operate on")
        ],
        state: Annotated[
            str,
            Field(description="The new state for the task run. Default is 'SUCCESS'."),
        ] = "SUCCESS",
        task_run_id: Annotated[
            str,
            Field(
                description="The ID of the task run to operate on. If not provided, the first failed task in the execution is restarted."
            ),
        ] = "",
    ) -> dict:
        """Restart the execution from a failed task or change the state of a task run.
        If the task_run_id is omitted, find the first failed task in that execution
        and mark it SUCCESS to restart downstream tasks. Returns the updated execution object.
        """
        get_resp = await client.get(f"/executions/{execution_id}")
        get_resp.raise_for_status()
        exec_data = get_resp.json()

        task_runs = (
            exec_data.get("taskRunList")
            or exec_data.get("taskRuns")
            or exec_data.get("tasks")
            or []
        )

        if not task_run_id:
            failed = []
            for tr in task_runs:
                st = tr.get("state")
                current = st.get("current") if isinstance(st, dict) else st
                if current == "FAILED":
                    failed.append(tr)
            if not failed:
                raise ValueError(
                    f"No failed task runs found in execution `{execution_id}`"
                )
            task_run_id = failed[0]["id"]

        body = {"taskRunId": task_run_id, "state": state}
        resp = await client.post(f"/executions/{execution_id}/state", json=body)
        resp.raise_for_status()
        return resp.json()
