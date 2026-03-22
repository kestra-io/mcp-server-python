from fastmcp import FastMCP
import httpx
from typing import Annotated, Any, List, Literal, Optional
from pydantic import Field


def register_logs_tools(mcp: FastMCP, client: httpx.AsyncClient) -> None:
    @mcp.tool()
    async def get_execution_logs(
        execution_id: Annotated[
            str, Field(description="The execution ID to get logs for")
        ],
        min_level: Annotated[
            Optional[str],
            Field(
                description="Minimum log level filter. Must be one of: ERROR, WARN, INFO, DEBUG, TRACE. Default is None (no filter)."
            ),
        ] = None,
        task_run_id: Annotated[
            Optional[str],
            Field(description="Filter logs by specific task run ID. Default is None.")
        ] = None,
        task_id: Annotated[
            Optional[str],
            Field(description="Filter logs by specific task ID. Default is None.")
        ] = None,
        attempt: Annotated[
            Optional[int],
            Field(description="Filter logs by specific attempt number. Default is None.")
        ] = None,
    ) -> dict:
        """Get logs for a specific execution with optional filtering.
        
        Returns a list of log entries in JSON format. Each log entry contains:
        - namespace: The namespace of the flow
        - flowId: The flow identifier
        - taskId: The task identifier (if applicable)
        - executionId: The execution identifier
        - taskRunId: The task run identifier (if applicable)
        - attemptNumber: The attempt number (if applicable)
        - triggerId: The trigger identifier (if applicable)
        - timestamp: When the log was created
        - level: The log level (ERROR, WARN, INFO, DEBUG, TRACE)
        - thread: The thread name
        - message: The log message
        - deleted: Whether the log entry is deleted
        - executionKind: The kind of execution
        
        Use this tool when the user asks to:
        - Get logs for execution [ID]
        - Show logs from execution [ID]
        - View execution logs with [filters]
        """
        valid_levels = {"ERROR", "WARN", "INFO", "DEBUG", "TRACE"}
        if min_level and min_level not in valid_levels:
            raise ValueError(
                f"Invalid min_level '{min_level}'. Must be one of: {', '.join(sorted(valid_levels))}"
            )

        params = {}
        if min_level:
            params["minLevel"] = min_level
        if task_run_id:
            params["taskRunId"] = task_run_id
        if task_id:
            params["taskId"] = task_id
        if attempt is not None:
            params["attempt"] = attempt

        resp = await client.get(f"/logs/{execution_id}", params=params)
        resp.raise_for_status()
        data = resp.json()
        return {"results": data} if isinstance(data, list) else data

    @mcp.tool()
    async def download_execution_logs(
        execution_id: Annotated[
            str, Field(description="The execution ID to download logs for")
        ],
        min_level: Annotated[
            Optional[str],
            Field(
                description="Minimum log level filter. Must be one of: ERROR, WARN, INFO, DEBUG, TRACE. Default is None (no filter)."
            ),
        ] = None,
        task_run_id: Annotated[
            Optional[str],
            Field(description="Filter logs by specific task run ID. Default is None.")
        ] = None,
        task_id: Annotated[
            Optional[str],
            Field(description="Filter logs by specific task ID. Default is None.")
        ] = None,
        attempt: Annotated[
            Optional[int],
            Field(description="Filter logs by specific attempt number. Default is None.")
        ] = None,
    ) -> dict:
        """Download logs for a specific execution as plain text.
        
        Returns the logs as a plain text string, formatted for easy reading.
        This is useful when you need to save logs to a file or process them as text.
        
        Use this tool when the user asks to:
        - Download logs for execution [ID]
        - Get logs as text for execution [ID]
        - Export logs from execution [ID]
        """
        valid_levels = {"ERROR", "WARN", "INFO", "DEBUG", "TRACE"}
        if min_level and min_level not in valid_levels:
            raise ValueError(
                f"Invalid min_level '{min_level}'. Must be one of: {', '.join(sorted(valid_levels))}"
            )

        params = {}
        if min_level:
            params["minLevel"] = min_level
        if task_run_id:
            params["taskRunId"] = task_run_id
        if task_id:
            params["taskId"] = task_id
        if attempt is not None:
            params["attempt"] = attempt

        resp = await client.get(f"/logs/{execution_id}/download", params=params)
        resp.raise_for_status()
        return {"result": resp.text}

    @mcp.tool()
    async def search_logs(
        query: Annotated[
            Optional[str],
            Field(description="Search term to look for in log messages. Default is None.")
        ] = None,
        namespace: Annotated[
            Optional[str],
            Field(description="Filter logs by namespace. Default is None.")
        ] = None,
        flow_id: Annotated[
            Optional[str],
            Field(description="Filter logs by flow ID. Default is None.")
        ] = None,
        min_level: Annotated[
            Optional[str],
            Field(
                description="Minimum log level filter. Must be one of: ERROR, WARN, INFO, DEBUG, TRACE. Default is None (no filter)."
            ),
        ] = None,
        start_date: Annotated[
            Optional[str],
            Field(
                description="Start date for log search in ISO 8601 format (e.g., '2024-01-01T00:00:00Z'). Default is None."
            ),
        ] = None,
        end_date: Annotated[
            Optional[str],
            Field(
                description="End date for log search in ISO 8601 format (e.g., '2024-01-31T23:59:59Z'). Default is None."
            ),
        ] = None,
        page: Annotated[
            int,
            Field(description="Page number for pagination. Default is 1.")
        ] = 1,
        size: Annotated[
            int,
            Field(description="Number of results per page. Default is 25.")
        ] = 25,
    ) -> dict:
        """Search logs across all executions with flexible filtering options.
        
        Returns a paginated result containing log entries that match the search criteria.
        The result includes:
        - results: List of log entries matching the search criteria
        - total: Total number of matching log entries
        - page: Current page number
        - size: Number of results per page
        
        Use this tool when the user asks to:
        - Search for logs containing [text]
        - Find logs with errors in [namespace/flow]
        - Show logs from [date range]
        - Search logs by level [level]
        """
        valid_levels = {"ERROR", "WARN", "INFO", "DEBUG", "TRACE"}
        if min_level and min_level not in valid_levels:
            raise ValueError(
                f"Invalid min_level '{min_level}'. Must be one of: {', '.join(sorted(valid_levels))}"
            )

        params = {
            "page": page,
            "size": size,
        }
        
        if query:
            params["q"] = query
        if namespace:
            params["namespace"] = namespace
        if flow_id:
            params["flowId"] = flow_id
        if min_level:
            params["minLevel"] = min_level
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date

        resp = await client.get("/logs/search", params=params)
        resp.raise_for_status()
        return resp.json()

    @mcp.tool()
    async def delete_execution_logs(
        execution_id: Annotated[
            str, Field(description="The execution ID to delete logs for")
        ],
        min_level: Annotated[
            Optional[str],
            Field(
                description="Minimum log level filter for deletion. Must be one of: ERROR, WARN, INFO, DEBUG, TRACE. Default is None (delete all logs)."
            ),
        ] = None,
        task_run_id: Annotated[
            Optional[str],
            Field(description="Delete logs for specific task run ID only. Default is None (delete all).")
        ] = None,
        task_id: Annotated[
            Optional[str],
            Field(description="Delete logs for specific task ID only. Default is None (delete all).")
        ] = None,
        attempt: Annotated[
            Optional[int],
            Field(description="Delete logs for specific attempt number only. Default is None (delete all).")
        ] = None,
    ) -> dict:
        """Delete logs for a specific execution with optional filtering.
        
        Returns a confirmation of the deletion operation.
        Use with caution as deleted logs cannot be recovered.
        
        Use this tool when the user asks to:
        - Delete logs for execution [ID]
        - Remove logs from execution [ID]
        - Clean up logs for execution [ID]
        """
        valid_levels = {"ERROR", "WARN", "INFO", "DEBUG", "TRACE"}
        if min_level and min_level not in valid_levels:
            raise ValueError(
                f"Invalid min_level '{min_level}'. Must be one of: {', '.join(sorted(valid_levels))}"
            )

        params = {}
        if min_level:
            params["minLevel"] = min_level
        if task_run_id:
            params["taskRunId"] = task_run_id
        if task_id:
            params["taskId"] = task_id
        if attempt is not None:
            params["attempt"] = attempt

        resp = await client.delete(f"/logs/{execution_id}", params=params)
        resp.raise_for_status()
        return resp.json() if resp.content else {"status": "deleted"}

    @mcp.tool()
    async def delete_flow_logs(
        namespace: Annotated[
            str, Field(description="The namespace of the flow")
        ],
        flow_id: Annotated[
            str, Field(description="The flow ID to delete logs for")
        ],
        trigger_id: Annotated[
            Optional[str],
            Field(description="Delete logs for specific trigger ID only. Default is None (delete all).")
        ] = None,
    ) -> dict:
        """Delete logs for all executions of a specific flow.
        
        Returns a confirmation of the deletion operation.
        Use with caution as deleted logs cannot be recovered.
        
        Use this tool when the user asks to:
        - Delete logs for flow [flow_id] in namespace [namespace]
        - Remove logs from flow [flow_id]
        - Clean up logs for flow [flow_id]
        """
        params = {}
        if trigger_id:
            params["triggerId"] = trigger_id

        resp = await client.delete(f"/logs/{namespace}/{flow_id}", params=params)
        resp.raise_for_status()
        return resp.json() if resp.content else {"status": "deleted"}

    @mcp.tool()
    async def follow_execution_logs(
        execution_id: Annotated[
            str, Field(description="The execution ID to follow logs for")
        ],
        min_level: Annotated[
            Optional[str],
            Field(
                description="Minimum log level filter. Must be one of: ERROR, WARN, INFO, DEBUG, TRACE. Default is None (no filter)."
            ),
        ] = None,
    ) -> dict:
        """Follow logs for a specific execution in real-time.
        
        Returns logs as they are generated using Server-Sent Events (SSE).
        This is useful for monitoring running executions.
        
        Note: This endpoint returns a stream of events, but the MCP tool will return
        the initial response. For true real-time following, consider using the API directly.
        
        Use this tool when the user asks to:
        - Follow logs for execution [ID]
        - Monitor logs for execution [ID]
        - Stream logs from execution [ID]
        """
        valid_levels = {"ERROR", "WARN", "INFO", "DEBUG", "TRACE"}
        if min_level and min_level not in valid_levels:
            raise ValueError(
                f"Invalid min_level '{min_level}'. Must be one of: {', '.join(sorted(valid_levels))}"
            )

        params = {}
        if min_level:
            params["minLevel"] = min_level

        resp = await client.get(f"/logs/{execution_id}/follow", params=params)
        resp.raise_for_status()
        return {"result": resp.text}
