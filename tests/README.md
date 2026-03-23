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

## Running Against Multiple Kestra Versions

To validate cross-version compatibility, run tests against all configured instances:

```bash
./tests/run_all_versions.sh        # default: --tb=short
./tests/run_all_versions.sh -v -x  # verbose, stop on first failure
```

The script loops through 4 Kestra instances (EE develop, OSS develop, EE latest, OSS latest), skips any that aren't reachable, and prints a summary at the end. Edit the `INSTANCES` array in the script to add/remove targets. Results are saved to `.test-results/`.

## Test Structure

- `test_ee.py` - Enterprise Edition specific tests (requires EE/Cloud environment)
- `test_*.py` - All other tests work with OSS, EE, and Cloud editions

## Test Coverage Notes

All MCP tools have integration test coverage **except** the 4 AI generation tools:

- `generate_flow`
- `generate_dashboard`
- `generate_app`
- `generate_test`

These tools delegate to Kestra's AI endpoints which depend on an external AI provider configuration. Their outputs are non-deterministic (LLM-generated YAML), making them unsuitable for automated assertions. They are best validated manually via the prompt files in `docs/prompts/`.
