from mcp.server.fastmcp import FastMCP
import httpx
from typing import Annotated, List, Literal, Union, Any
from pydantic import Field
import json
import re


def register_kv_tools(mcp: FastMCP, client: httpx.AsyncClient) -> None:
    @mcp.tool()
    async def manage_kv_store(
        namespace: Annotated[str, Field(description="The namespace to operate in")],
        action: Annotated[
            Literal["get", "set", "list", "delete"],
            Field(description="The operation to perform: get, set, list, or delete"),
        ],
        key: Annotated[
            str, Field(description="Required for 'get', 'set', and 'delete' actions")
        ] = None,
        value: Annotated[Any, Field(description="Required for 'set' action")] = None,
    ) -> Union[dict, List[str], str]:
        """Perform a KV operation in the given namespace. Returns:
        - for "get": the value stored at the key (as JSON);
        - for "list": a list of all keys in the namespace;
        - for "set": {"status": "ok"} on success;
        - for "delete": {"deleted": <original response>} or an error message.
        """
        if action == "get":
            if not key:
                raise ValueError("`key` is required for 'get' action")
            resp = await client.get(f"/namespaces/{namespace}/kv/{key}")
            resp.raise_for_status()
            return resp.json()

        elif action == "set":
            if not key or value is None:
                raise ValueError("`key` and `value` are required for 'set' action")
            date_regex = re.compile(r"^\d{4}-\d{2}-\d{2}$")
            datetime_regex = re.compile(
                r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:Z)?$"
            )
            if not isinstance(value, str):
                content = json.dumps(value)
            elif date_regex.match(value) or datetime_regex.match(value):
                content = value  # send as raw string, not quoted
            else:
                try:
                    json_value = json.loads(value)
                    content = value  # already valid JSON
                except Exception:
                    content = json.dumps(value)
            resp = await client.put(
                f"/namespaces/{namespace}/kv/{key}",
                content=content,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return {"status": "ok"}

        elif action == "list":
            resp = await client.get(f"/namespaces/{namespace}/kv")
            resp.raise_for_status()
            return resp.json()

        elif action == "delete":
            if not key:
                raise ValueError("`key` is required for 'delete' action")
            resp = await client.delete(f"/namespaces/{namespace}/kv/{key}")
            try:
                resp.raise_for_status()
                deleted = resp.json() if resp.content else None
                return {"deleted": deleted}
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return f"Error: Key '{key}' not found in namespace '{namespace}'."
                raise

        else:
            raise ValueError(f"Unknown action: {action}")
