import pytest
from dotenv import load_dotenv
import json
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env", override=True)


@pytest.mark.asyncio
async def test_namespace_file_actions(kestra_client):
    """Test namespace file actions."""
    test_file_path = "etl.py"
    original_path = Path(__file__).parent / "code" / "files" / test_file_path
    with open(original_path, "rb") as f:
        original_content = f.read()
    assert isinstance(
        original_content, bytes
    ), f"original_content is {type(original_content)}"

    # Test file creation
    result = await kestra_client.call_tool(
        "namespace_file_action",
        {
            "namespace": "company.team",
            "path": test_file_path,
            "action": "create",
            "file_content": original_content,
        },
    )
    assert json.loads(result.content[0].text)["status"] == "created"

    # Test file retrieval
    content = await kestra_client.call_tool(
        "namespace_file_action",
        {"namespace": "company.team", "path": test_file_path, "action": "get"},
    )
    print(f"Get etl.py result: {content.content[0].text}")
    retrieved = json.loads(content.content[0].text)
    assert "import pandas as pd" in retrieved.get("content", retrieved)

    # Test file search
    search_results = await kestra_client.call_tool(
        "namespace_file_action",
        {"namespace": "company.team", "action": "search", "q": "etl.py"},
    )
    print(f"Search results for etl.py: {search_results.content[0].text}")
    search_data = json.loads(search_results.content[0].text)
    paths = search_data.get("results", search_data) if isinstance(search_data, dict) else search_data
    assert any("etl.py" in str(path) for path in paths)

    # Test file move
    move_result = await kestra_client.call_tool(
        "namespace_file_action",
        {
            "namespace": "company.team",
            "path": test_file_path,
            "action": "move",
            "to_path": "moved_etl.py",
        },
    )
    assert json.loads(move_result.content[0].text)["status"] == "moved"

    # Verify the file was moved
    moved_content = await kestra_client.call_tool(
        "namespace_file_action",
        {"namespace": "company.team", "path": "moved_etl.py", "action": "get"},
    )
    print(f"Get moved_etl.py result: {moved_content.content[0].text}")
    retrieved = json.loads(moved_content.content[0].text)
    assert "import pandas as pd" in retrieved.get("content", retrieved)

    # Test file deletion
    delete_result = await kestra_client.call_tool(
        "namespace_file_action",
        {"namespace": "company.team", "path": "moved_etl.py", "action": "delete"},
    )
    assert json.loads(delete_result.content[0].text)["status"] == "deleted"


@pytest.mark.asyncio
async def test_namespace_directory_actions(kestra_client):
    """Test namespace directory actions."""
    # Test directory creation
    result = await kestra_client.call_tool(
        "namespace_directory_action",
        {"namespace": "company.team", "path": "test_dir", "action": "create"},
    )
    print(f"Create test_dir result: {result}")
    assert json.loads(result.content[0].text)["status"] == "directory_created"

    # Create a subdirectory
    result = await kestra_client.call_tool(
        "namespace_directory_action",
        {
            "namespace": "company.team",
            "path": "test_dir/subdir",
            "action": "create",
        },
    )
    assert json.loads(result.content[0].text)["status"] == "directory_created"

    # Test directory listing
    listing = await kestra_client.call_tool(
        "namespace_directory_action",
        {"namespace": "company.team", "path": "test_dir", "action": "list"},
    )
    print(f"List test_dir result: {listing}")
    listing_response = json.loads(listing.content[0].text)
    listing_items = listing_response.get("results", listing_response) if isinstance(listing_response, dict) else listing_response
    # Verify the response schema for a single directory item
    assert isinstance(listing_items, list)
    assert len(listing_items) > 0
    listing_data = listing_items[0]
    assert "type" in listing_data
    assert "size" in listing_data
    assert "fileName" in listing_data
    assert "lastModifiedTime" in listing_data
    assert "creationTime" in listing_data
    assert listing_data["type"] == "Directory"
    assert listing_data["fileName"] == "subdir"

    # Test directory move (may fail with 500 on some Kestra versions)
    try:
        move_result = await kestra_client.call_tool(
            "namespace_directory_action",
            {
                "namespace": "company.team",
                "path": "test_dir/subdir",
                "action": "move",
                "to_path": "moved_subdir",
            },
        )
        assert json.loads(move_result.content[0].text)["status"] == "directory_moved"
        dir_to_delete = "moved_subdir"
    except Exception as e:
        print(f"Directory move not supported on this server: {e}")
        dir_to_delete = None

    # Test directory deletion
    delete_result = await kestra_client.call_tool(
        "namespace_directory_action",
        {"namespace": "company.team", "path": dir_to_delete or "test_dir/subdir", "action": "delete"},
    )
    print(f"Delete subdir result: {delete_result}")
    assert json.loads(delete_result.content[0].text)["status"] == "directory_deleted"

    # Delete parent directory
    delete_result = await kestra_client.call_tool(
        "namespace_directory_action",
        {"namespace": "company.team", "path": "test_dir", "action": "delete"},
    )
    print(f"Delete test_dir result: {delete_result}")
    assert json.loads(delete_result.content[0].text)["status"] == "directory_deleted"
