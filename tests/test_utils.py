from pathlib import Path
from fastmcp import Client
import json
import asyncio

mcp_server_config = {
    "mcpServers": {
        "kestraPython": {
            "command": "uv",
            "args": [
                "--directory",
                "src",
                "run",
                "server.py",
            ],
        }
    }
}


async def create_flow(file: str, client: Client) -> dict:
    yaml_path = Path(__file__).parent / "code" / file
    with open(yaml_path, "r") as f:
        yaml_flow = f.read()
    result = await client.call_tool(
        "create_flow_from_yaml", {"yaml_definition": yaml_flow}
    )
    response_json = json.loads(result[0].text)
    return response_json


async def create_test(file: str, client: Client) -> dict:
    yaml_path = Path(__file__).parent / "code" / file
    with open(yaml_path, "r") as f:
        yaml_source = f.read()
    result = await client.call_tool(
        "manage_tests", {"action": "create", "yaml_source": yaml_source}
    )
    response_json = json.loads(result[0].text)
    return response_json


async def create_app(file: str, client: Client) -> dict:
    yaml_path = Path(__file__).parent / "code" / file
    with open(yaml_path, "r") as f:
        yaml_source = f.read()
    result = await client.call_tool(
        "manage_apps", {"action": "create", "yaml_source": yaml_source}
    )
    response_json = json.loads(result[0].text)
    return response_json


async def poll_for_execution(
    client, execution_id, desired_state=None, max_retries=5, retry_interval=1
):
    """
    Polls for an execution to reach a desired state or any terminal state if desired_state is None.
    Args:
        client: The FastMCP client instance.
        execution_id: The execution ID to poll.
        desired_state: The state to wait for (e.g., 'RUNNING', 'SUCCESS'). If None, waits for any terminal state.
        max_retries: Maximum number of polling attempts.
        retry_interval: Delay in seconds between retries.
    Returns:
        The execution response JSON when the desired state or a terminal state is reached or last polled.
    Raises:
        AssertionError if execution_id does not match.
    """
    terminal_states = {"SUCCESS", "WARNING", "FAILED", "KILLED", "CANCELLED", "SKIPPED"}

    retry_count = 0
    response_json = None
    while retry_count < max_retries:
        result = await client.call_tool(
            "manage_executions", {"action": "get", "execution_id": execution_id}
        )
        response_json = json.loads(result[0].text) if result and result[0].text else {}
        assert response_json["id"] == execution_id
        current_state = response_json["state"]["current"]
        if desired_state:
            if current_state == desired_state:
                break
        else:
            if current_state in terminal_states:
                break
        retry_count += 1
        if retry_count < max_retries:
            await asyncio.sleep(retry_interval)
    return response_json
