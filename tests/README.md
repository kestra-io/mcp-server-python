# Testing

## Running Tests

### For OSS (Open Source) Edition
Run all tests except Enterprise Edition tests:
```bash
uv run pytest tests/ --ignore=tests/test_ee.py
```

**Note:** You can also disable EE-specific tools in the MCP server by adding this environment variable to your `.env` file:
```bash
KESTRA_MCP_DISABLED_TOOLS=ee
```

This prevents the MCP server from attempting to load any Enterprise Edition specific tools.

### For Enterprise Edition (EE/Cloud)
Run all tests including Enterprise Edition tests:
```bash
uv run pytest tests/
```

### Run Specific Test Files
```bash
# Run only flow tests
uv run pytest tests/test_flow.py

# Run only execution tests
uv run pytest tests/test_execution.py

# Run multiple specific test files
uv run pytest tests/test_flow.py tests/test_execution.py
```

### Run with Verbose Output
```bash
uv run pytest tests/ --ignore=tests/test_ee.py -v
```

## Test Structure

- `test_ee.py` - Enterprise Edition specific tests (requires EE/Cloud environment)
- `test_*.py` - All other tests work with OSS, EE, and Cloud editions
