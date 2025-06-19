import pytest
from fastmcp import Client
from dotenv import load_dotenv
import json
from pathlib import Path
from test_utils import mcp_server_config
import asyncio

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env", override=True)


@pytest.mark.asyncio
async def test_namespace_file_actions():
    """Test namespace file actions."""
    async with Client(mcp_server_config) as client:
        test_file_path = "etl.py"
        original_path = Path(__file__).parent / "code" / "files" / test_file_path
        with open(original_path, "rb") as f:
            original_content = f.read()
        assert isinstance(
            original_content, bytes
        ), f"original_content is {type(original_content)}"

        # Test file creation
        result = await client.call_tool(
            "namespace_file_action",
            {
                "namespace": "company.team",
                "path": test_file_path,
                "action": "create",
                "file_content": original_content,
            },
        )
        assert json.loads(result[0].text)["status"] == "created"

        # Test file retrieval
        content = await client.call_tool(
            "namespace_file_action",
            {"namespace": "company.team", "path": test_file_path, "action": "get"},
        )
        print(f"Get etl.py result: {content[0].text}")
        retrieved = json.loads(content[0].text)
        assert "import pandas as pd" in retrieved

        # Test file search
        search_results = await client.call_tool(
            "namespace_file_action",
            {"namespace": "company.team", "action": "search", "q": "etl.py"},
        )
        print(f"First in a list of search results for etl.py: {search_results[0].text}")
        paths = [item.text for item in search_results]
        assert any("etl.py" in path for path in paths)

        # Test file move
        move_result = await client.call_tool(
            "namespace_file_action",
            {
                "namespace": "company.team",
                "path": test_file_path,
                "action": "move",
                "to_path": "moved_etl.py",
            },
        )
        assert json.loads(move_result[0].text)["status"] == "moved"

        # Verify the file was moved
        moved_content = await client.call_tool(
            "namespace_file_action",
            {"namespace": "company.team", "path": "moved_etl.py", "action": "get"},
        )
        print(f"Get moved_etl.py result: {moved_content[0].text}")
        retrieved = json.loads(moved_content[0].text)
        assert "import pandas as pd" in retrieved

        # Test file deletion
        delete_result = await client.call_tool(
            "namespace_file_action",
            {"namespace": "company.team", "path": "moved_etl.py", "action": "delete"},
        )
        assert json.loads(delete_result[0].text)["status"] == "deleted"


@pytest.mark.asyncio
async def test_namespace_directory_actions():
    """Test namespace directory actions."""
    async with Client(mcp_server_config) as client:
        # Test directory creation
        result = await client.call_tool(
            "namespace_directory_action",
            {"namespace": "company.team", "path": "test_dir", "action": "create"},
        )
        print(f"Create test_dir result: {result}")
        assert json.loads(result[0].text)["status"] == "directory_created"

        # Create a subdirectory
        result = await client.call_tool(
            "namespace_directory_action",
            {
                "namespace": "company.team",
                "path": "test_dir/subdir",
                "action": "create",
            },
        )
        assert json.loads(result[0].text)["status"] == "directory_created"

        # Test directory listing
        listing = await client.call_tool(
            "namespace_directory_action",
            {"namespace": "company.team", "path": "test_dir", "action": "list"},
        )
        print(f"List test_dir result: {listing}")
        listing_data = json.loads(listing[0].text)
        # Verify the response schema for a single directory item
        assert "type" in listing_data
        assert "size" in listing_data
        assert "fileName" in listing_data
        assert "lastModifiedTime" in listing_data
        assert "creationTime" in listing_data
        assert listing_data["type"] == "Directory"
        assert listing_data["fileName"] == "subdir"

        # Test directory move
        move_result = await client.call_tool(
            "namespace_directory_action",
            {
                "namespace": "company.team",
                "path": "test_dir/subdir",
                "action": "move",
                "to_path": "moved_subdir",
            },
        )
        assert json.loads(move_result[0].text)["status"] == "directory_moved"

        # Verify the directory was moved by checking its properties
        moved_listing = await client.call_tool(
            "namespace_directory_action",
            {"namespace": "company.team", "path": "moved_subdir", "action": "list"},
        )
        print(f"List moved_subdir result: {moved_listing}")
        if moved_listing:  # Check if we got a response
            moved_data = json.loads(moved_listing[0].text)
            assert moved_data["type"] == "Directory"
            assert moved_data["fileName"] == "moved_subdir"

        # Test directory deletion
        delete_result = await client.call_tool(
            "namespace_directory_action",
            {"namespace": "company.team", "path": "moved_subdir", "action": "delete"},
        )
        print(f"Delete test_dir result: {delete_result}")
        assert json.loads(delete_result[0].text)["status"] == "directory_deleted"


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_namespace_file_actions())
    asyncio.run(test_namespace_directory_actions())
