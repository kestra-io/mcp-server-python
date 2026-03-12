# Logs Tools

The `logs` tool group provides comprehensive log management capabilities for Kestra executions.

## Available Tools

- **get_execution_logs**: Retrieve logs for a specific execution with optional filtering by level, task, or attempt
- **download_execution_logs**: Download logs as plain text for a specific execution
- **search_logs**: Search logs across all executions with flexible filtering options
- **delete_execution_logs**: Delete logs for a specific execution (use with caution)
- **delete_flow_logs**: Delete logs for all executions of a specific flow
- **follow_execution_logs**: Follow logs in real-time for a running execution

## Filtering Options

All log tools support filtering by:
- **Log level**: ERROR, WARN, INFO, DEBUG, TRACE
- **Task ID**: Filter by specific task identifier
- **Task Run ID**: Filter by specific task run identifier
- **Attempt number**: Filter by specific attempt number
- **Date ranges**: For search operations (startDate, endDate)

## Usage Examples

### Get logs for a specific execution
```
Get logs for execution abc123 with INFO level or higher
```

### Search for logs across executions
```
Search for error logs in namespace production.etl
Search for logs containing "database" in the last 24 hours
```

### Download logs as text
```
Download all logs from execution xyz789
Download ERROR level logs from execution abc123
```

### Delete logs (use with caution)
```
Delete logs for execution abc123
Delete logs for flow data-pipeline in namespace production
```

### Follow logs in real-time
```
Follow logs for execution abc123
Follow ERROR level logs for execution xyz789
```

## API Endpoints Used

The logs tools interact with the following Kestra API endpoints:

- `GET /api/v1/{tenant}/logs/{executionId}` - Get logs for execution
- `GET /api/v1/{tenant}/logs/{executionId}/download` - Download logs as text
- `GET /api/v1/{tenant}/logs/search` - Search logs globally
- `DELETE /api/v1/{tenant}/logs/{executionId}` - Delete execution logs
- `DELETE /api/v1/{tenant}/logs/{namespace}/{flowId}` - Delete flow logs
- `GET /api/v1/{tenant}/logs/{executionId}/follow` - Follow logs (SSE)

## Log Entry Structure

Each log entry contains the following fields:
- `namespace`: The namespace of the flow
- `flowId`: The flow identifier
- `taskId`: The task identifier (if applicable)
- `executionId`: The execution identifier
- `taskRunId`: The task run identifier (if applicable)
- `attemptNumber`: The attempt number (if applicable)
- `triggerId`: The trigger identifier (if applicable)
- `timestamp`: When the log was created (ISO 8601 format)
- `level`: The log level (ERROR, WARN, INFO, DEBUG, TRACE)
- `thread`: The thread name
- `message`: The log message content
- `deleted`: Whether the log entry is deleted
- `executionKind`: The kind of execution

## Notes

- Logs are stored in the database in Community Edition
- Enterprise Edition can use Elasticsearch as additional backend
- Deleted logs cannot be recovered
- Real-time following uses Server-Sent Events (SSE)
- All tools work with both Community and Enterprise editions
