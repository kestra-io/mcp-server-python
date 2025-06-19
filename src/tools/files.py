from mcp.server.fastmcp import FastMCP
import httpx
from typing import Annotated, List, Literal, Union
from pydantic import Field
import os


def register_files_tools(mcp: FastMCP, client: httpx.AsyncClient) -> None:
    @mcp.tool()
    async def namespace_file_action(
        namespace: Annotated[str, Field(description="The namespace to operate in")],
        path: Annotated[
            str, Field(description="The file path. Not required for 'search' action.")
        ] = None,
        action: Literal["get", "create", "move", "delete", "search"] = "get",
        file_content: Annotated[
            bytes,
            Field(
                description="Required for 'create' action, the content of the file as bytes."
            ),
        ] = None,
        to_path: Annotated[
            str, Field(description="Required for 'move' action, the destination path.")
        ] = None,
        q: Annotated[
            str,
            Field(
                description="Required for 'search' action, the query string for full-text search in file paths."
            ),
        ] = None,
    ) -> Union[dict, List[str], str]:
        """Perform a file operation in the given namespace. Returns: for 'get', 'create', 'move', 'delete', and 'search' actions, see their respective return types."""
        if action == "get":
            resp = await client.get(
                f"/namespaces/{namespace}/files", params={"path": path}
            )
            resp.raise_for_status()
            return resp.content

        elif action == "create":
            if file_content is None:
                raise ValueError("file_content is required for 'create' action")
            files = {"fileContent": (path, file_content, "application/octet-stream")}

            resp = await client.post(
                f"/namespaces/{namespace}/files", params={"path": path}, files=files
            )
            resp.raise_for_status()
            return resp.json() if resp.content else {"status": "created"}

        elif action == "move":
            if to_path is None:
                raise ValueError("to_path is required for 'move' action")
            try:
                resp = await client.put(
                    f"/namespaces/{namespace}/files",
                    params={"from": path, "to": to_path},
                )
                resp.raise_for_status()
                return resp.json() if resp.content else {"status": "moved"}
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    # Check if the source file exists
                    check_file = await client.get(
                        f"/namespaces/{namespace}/files", params={"path": path}
                    )
                    if check_file.status_code == 404:
                        return f"Error: The source file '{path}' does not exist in namespace '{namespace}'."
                    # Check if the destination directory exists
                    dest_dir = os.path.dirname(to_path)
                    if dest_dir:
                        check_dir = await client.get(
                            f"/namespaces/{namespace}/files/directory",
                            params={"path": dest_dir},
                        )
                        if check_dir.status_code == 404:
                            return f"Error: The destination directory '{dest_dir}' does not exist in namespace '{namespace}'. Please create it first."
                    # Generic 404
                    return f"Error: Move failed with 404 Not Found. Please check your paths and try again."
                else:
                    raise

        elif action == "delete":
            resp = await client.delete(
                f"/namespaces/{namespace}/files", params={"path": path}
            )
            resp.raise_for_status()
            return resp.json() if resp.content else {"status": "deleted"}
        elif action == "search":
            if q is None:
                raise ValueError("q is required for 'search' action")
            resp = await client.get(
                f"/namespaces/{namespace}/files/search", params={"q": q}
            )
            resp.raise_for_status()
            return resp.json()
        else:
            raise ValueError(f"Unknown action: {action}")

    @mcp.tool()
    async def namespace_directory_action(
        namespace: Annotated[str, Field(description="The namespace to operate in")],
        path: Annotated[str, Field(description="The directory path")],
        action: Annotated[
            Literal["list", "create", "delete", "move"],
            Field(description="The action to perform: list, create, move, or delete"),
        ],
        to_path: Annotated[
            str, Field(description="Required for 'move' action, the destination path.")
        ] = None,
    ) -> Union[List[dict], dict, str]:
        """Perform a directory operation in the given namespace."""
        if action == "list":
            resp = await client.get(
                f"/namespaces/{namespace}/files/directory", params={"path": path}
            )
            resp.raise_for_status()
            return resp.json()
        elif action == "create":
            resp = await client.post(
                f"/namespaces/{namespace}/files/directory", params={"path": path}
            )
            resp.raise_for_status()
            return resp.json() if resp.content else {"status": "directory_created"}
        elif action == "delete":
            resp = await client.delete(
                f"/namespaces/{namespace}/files", params={"path": path}
            )
            if resp.status_code == 404:
                return f"Error: The directory '{path}' does not exist in namespace '{namespace}'."
            resp.raise_for_status()
            return resp.json() if resp.content else {"status": "directory_deleted"}
        elif action == "move":
            if to_path is None:
                return "Error: 'to_path' must be provided for move action."
            try:
                resp = await client.put(
                    f"/namespaces/{namespace}/files",
                    params={"from": path, "to": to_path},
                )
                resp.raise_for_status()
                return resp.json() if resp.content else {"status": "directory_moved"}
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    # Check if the source directory exists
                    check_dir = await client.get(
                        f"/namespaces/{namespace}/files/directory",
                        params={"path": path},
                    )
                    if check_dir.status_code == 404:
                        return f"Error: The source directory '{path}' does not exist in namespace '{namespace}'."
                    # Check if the destination parent directory exists
                    dest_parent = os.path.dirname(to_path)
                    if dest_parent:
                        check_dest = await client.get(
                            f"/namespaces/{namespace}/files/directory",
                            params={"path": dest_parent},
                        )
                        if check_dest.status_code == 404:
                            return f"Error: The destination parent directory '{dest_parent}' does not exist in namespace '{namespace}'. Please create it first."
                    return f"Error: Move failed with 404 Not Found. Please check your paths and try again."
                else:
                    raise
        else:
            raise ValueError(f"Unknown action: {action}")
