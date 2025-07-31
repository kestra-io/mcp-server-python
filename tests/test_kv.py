import pytest
from fastmcp import Client
from dotenv import load_dotenv
import json
from pathlib import Path
from test_utils import mcp_server_config
import asyncio
import fastmcp.exceptions

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env", override=True)


@pytest.mark.asyncio
async def test_kv_actions():
    """Test KV store operations."""
    async with Client(mcp_server_config) as client:
        # Test setting different types of values
        test_cases = [
            ("string_key", "test_value"),
            ("number_key", 42),
            ("boolean_key", True),
            ("date_key", "2024-03-15"),
            ("datetime_key", "2024-03-15T14:30:00Z"),
            ("json_key", {"nested": {"key": "value"}}),
            ("duration_key", "PT4H"),
        ]

        for key, value in test_cases:
            result = await client.call_tool(
                "manage_kv_store",
                {
                    "namespace": "company.team",
                    "action": "set",
                    "key": key,
                    "value": value,
                },
            )
            print(f"Set {key} result: {result}")
            assert json.loads(result[0].text)["status"] == "ok"

        # Test getting values
        for key, expected_value in test_cases:
            result = await client.call_tool(
                "manage_kv_store",
                {"namespace": "company.team", "action": "get", "key": key},
            )
            print(f"Get {key} result: {result}")
            response = json.loads(result[0].text)
            assert "type" in response
            assert "value" in response
            # For JSON values, compare directly if already a dict
            if response["type"] == "JSON":
                assert response["value"] == expected_value
            elif isinstance(expected_value, (dict, list)):
                assert json.loads(response["value"]) == expected_value
            else:
                assert response["value"] == expected_value

        # Test listing all keys
        result = await client.call_tool(
            "manage_kv_store", {"namespace": "company.team", "action": "list"}
        )
        print(f"List keys result: {json.loads(result[0].text)}")
        # Parse each key info from the result list
        keys = [json.loads(item.text) for item in result]
        assert isinstance(keys, list)
        # Verify all our test keys are in the list
        for key, _ in test_cases:
            assert any(k["key"] == key for k in keys)
            # Verify each key has required fields
            key_info = next(k for k in keys if k["key"] == key)
            assert "creationDate" in key_info
            assert "updateDate" in key_info
            # expirationDate may not always be present, so check if it exists before asserting
            if "expirationDate" in key_info:
                assert key_info["expirationDate"] is None or isinstance(
                    key_info["expirationDate"], str
                )

        # Test deleting keys
        for key, _ in test_cases:
            result = await client.call_tool(
                "manage_kv_store",
                {"namespace": "company.team", "action": "delete", "key": key},
            )
            deleted_response = json.loads(result[0].text)
            print(f"Delete {key} result: {deleted_response}")
            assert deleted_response.get("deleted") is True

        # Verify keys are deleted by trying to get them
        for key, _ in test_cases:
            with pytest.raises(fastmcp.exceptions.ToolError):
                await client.call_tool(
                    "manage_kv_store",
                    {"namespace": "company.team", "action": "get", "key": key},
                )

        # Test error cases
        # Test setting without value (should raise ToolError)
        with pytest.raises(fastmcp.exceptions.ToolError):
            await client.call_tool(
                "manage_kv_store",
                {"namespace": "company.team", "action": "set", "key": "error_key"},
            )

        # Test getting non-existent key
        with pytest.raises(fastmcp.exceptions.ToolError):
            await client.call_tool(
                "manage_kv_store",
                {
                    "namespace": "company.team",
                    "action": "get",
                    "key": "non_existent_key",
                },
            )

        # Test deleting non-existent key
        result = await client.call_tool(
            "manage_kv_store",
            {
                "namespace": "company.team",
                "action": "delete",
                "key": "non_existent_key",
            },
        )
        print(f"Delete non-existent key result: {result}")
        deleted_response = json.loads(result[0].text)
        assert deleted_response.get("deleted") is False


if __name__ == "__main__":
    asyncio.run(test_kv_actions())
