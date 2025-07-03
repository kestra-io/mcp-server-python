import pytest
from fastmcp import Client
from dotenv import load_dotenv
import json
from pathlib import Path
from test_utils import mcp_server_config, create_flow, create_test, create_app
import time


load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env", override=True)


@pytest.mark.asyncio
async def test_manage_maintenance_mode():
    """Test managing maintenance mode with different actions."""
    async with Client(mcp_server_config) as client:
        # Test case 1: Enter maintenance mode
        result = await client.call_tool("manage_maintenance_mode", {"action": "enter"})
        response_json = json.loads(result[0].text)
        print("Enter maintenance mode response:", response_json)

        # Verify enter response is an empty dictionary
        assert response_json == {}

        # Test case 2: Exit maintenance mode
        result = await client.call_tool("manage_maintenance_mode", {"action": "exit"})
        response_json = json.loads(result[0].text)
        print("Exit maintenance mode response:", response_json)

        # Verify exit response is an empty dictionary
        assert response_json == {}

        # Test case 3: Error - invalid action
        with pytest.raises(Exception):
            await client.call_tool(
                "manage_maintenance_mode", {"action": "invalid_action"}
            )

        # Test case 4: Error - missing action
        with pytest.raises(Exception):
            await client.call_tool("manage_maintenance_mode", {})


@pytest.mark.skip(reason="API changed in 0.24 cycle - needs to be updated") # TODO: update this test
@pytest.mark.asyncio
async def test_search_users():
    """Test searching users with different filters."""
    async with Client(mcp_server_config) as client:
        # Test case 1: Search users by email
        result = await client.call_tool(
            "search_users", {"q": "test@kestra.io", "page": 1, "size": 10}
        )
        response_json = json.loads(result[0].text)
        print("Search users response:", response_json)

        # Verify response structure
        assert "results" in response_json
        assert "total" in response_json
        assert isinstance(response_json["results"], list)

        # Test case 2: Search users by type
        result = await client.call_tool(
            "search_users", {"type": "STANDARD", "page": 1, "size": 10}
        )
        response_json = json.loads(result[0].text)
        print("Search users by type response:", response_json)

        # Verify response structure and type filter
        assert "results" in response_json
        assert "total" in response_json
        assert isinstance(response_json["results"], list)
        for user in response_json["results"]:
            assert user["type"] == "STANDARD"

        # Test case 3: Search with pagination
        result = await client.call_tool("search_users", {"page": 1, "size": 2})
        response_json = json.loads(result[0].text)
        print("Search users with pagination response:", response_json)

        # Verify pagination
        assert "results" in response_json
        assert "total" in response_json
        assert len(response_json["results"]) <= 2  # Should not exceed page size

        # Test case 4: Search with sort
        result = await client.call_tool(
            "search_users", {"sort": ["username:asc"], "page": 1, "size": 10}
        )
        response_json = json.loads(result[0].text)
        print("Search users with sort response:", response_json)

        # Verify response structure
        assert "results" in response_json
        assert "total" in response_json
        assert isinstance(response_json["results"], list)

        # Test case 5: Error - invalid type
        with pytest.raises(Exception):
            await client.call_tool(
                "search_users", {"type": "INVALID_TYPE", "page": 1, "size": 10}
            )


@pytest.mark.asyncio
async def test_manage_worker_groups():
    """Test managing worker groups with different actions."""
    async with Client(mcp_server_config) as client:
        # First check if test worker groups exist and delete them
        result = await client.call_tool("manage_worker_groups", {"action": "list"})
        response_json = json.loads(result[0].text)
        print("List worker groups response:", response_json)

        # Delete any existing test worker groups
        for worker_group in response_json["workerGroups"]:
            if worker_group["key"].startswith("test_worker"):
                await client.call_tool(
                    "manage_worker_groups",
                    {"action": "delete", "id_": worker_group["id"]},
                )

        # Verify list response structure
        assert "workerGroups" in response_json
        assert isinstance(response_json["workerGroups"], list)

        # If there are any worker groups, verify their structure
        if response_json["workerGroups"]:
            worker_group = response_json["workerGroups"][0]
            assert "id" in worker_group
            assert "key" in worker_group
            assert "description" in worker_group
            assert "activeWorkers" in worker_group
            assert isinstance(worker_group["activeWorkers"], int)

        # Test case 2: Create a new worker group with unique key
        unique_key = (
            f"test_worker_{int(time.time())}"  # Use timestamp to ensure uniqueness
        )
        result = await client.call_tool(
            "manage_worker_groups",
            {
                "action": "create",
                "key": unique_key,
                "description": "Test worker group for unit tests",
            },
        )
        response_json = json.loads(result[0].text)
        print("Create worker group response:", response_json)

        # Verify create response structure
        assert "id" in response_json
        assert "key" in response_json
        assert response_json["key"] == unique_key
        assert "description" in response_json
        assert response_json["description"] == "Test worker group for unit tests"
        assert "allowedTenants" in response_json
        assert isinstance(response_json["allowedTenants"], list)

        worker_group_id = response_json["id"]

        # Test case 3: Get worker group details
        result = await client.call_tool(
            "manage_worker_groups", {"action": "get", "id_": worker_group_id}
        )
        response_json = json.loads(result[0].text)
        print("Get worker group response:", response_json)

        # Verify get response structure
        assert "id" in response_json
        assert response_json["id"] == worker_group_id
        assert "key" in response_json
        assert response_json["key"] == unique_key
        assert "description" in response_json
        assert response_json["description"] == "Test worker group for unit tests"
        assert "allowedTenants" in response_json
        assert isinstance(response_json["allowedTenants"], list)
        assert "workers" in response_json
        assert isinstance(response_json["workers"], list)

        # Test case 4: Update worker group - only update description and allowedTenants
        result = await client.call_tool(
            "manage_worker_groups",
            {
                "action": "update",
                "id_": worker_group_id,
                "key": unique_key,  # Keep the same key
                "description": "Updated test worker group",
                "allowedTenants": ["main"],
            },
        )
        response_json = json.loads(result[0].text)
        print("Update worker group response:", response_json)

        # Verify update response structure
        assert "id" in response_json
        assert response_json["id"] == worker_group_id
        assert "key" in response_json
        assert response_json["key"] == unique_key  # Key should remain unchanged
        assert "description" in response_json
        assert response_json["description"] == "Updated test worker group"
        assert "allowedTenants" in response_json
        assert isinstance(response_json["allowedTenants"], list)
        assert "main" in response_json["allowedTenants"]

        # Test case 5: Delete worker group
        result = await client.call_tool(
            "manage_worker_groups", {"action": "delete", "id_": worker_group_id}
        )
        response_json = json.loads(result[0].text)
        print("Delete worker group response:", response_json)
        assert response_json == {}  # Verify empty dictionary response

        # Test case 6: Error - missing required parameters for create
        with pytest.raises(Exception):
            await client.call_tool("manage_worker_groups", {"action": "create"})

        # Test case 7: Error - missing required parameters for update
        with pytest.raises(Exception):
            await client.call_tool("manage_worker_groups", {"action": "update"})

        # Test case 8: Error - missing required parameters for get
        with pytest.raises(Exception):
            await client.call_tool("manage_worker_groups", {"action": "get"})

        # Test case 9: Error - missing required parameters for delete
        with pytest.raises(Exception):
            await client.call_tool("manage_worker_groups", {"action": "delete"})

        # Test case 10: Error - invalid action
        with pytest.raises(Exception):
            await client.call_tool("manage_worker_groups", {"action": "invalid_action"})


@pytest.mark.asyncio
async def test_manage_maintenance_mode():
    """Test managing maintenance mode with different actions."""
    async with Client(mcp_server_config) as client:
        # Test case 1: Enter maintenance mode
        result = await client.call_tool("manage_maintenance_mode", {"action": "enter"})
        response_json = json.loads(result[0].text)
        print("Enter maintenance mode response:", response_json)

        # Verify enter response structure
        # The response might be empty or contain status information
        # We just verify it's a valid JSON response

        # Test case 2: Exit maintenance mode
        result = await client.call_tool("manage_maintenance_mode", {"action": "exit"})
        response_json = json.loads(result[0].text)
        print("Exit maintenance mode response:", response_json)

        # Verify exit response structure
        # The response might be empty or contain status information
        # We just verify it's a valid JSON response

        # Test case 3: Error - invalid action
        with pytest.raises(Exception):
            await client.call_tool(
                "manage_maintenance_mode", {"action": "invalid_action"}
            )

        # Test case 4: Error - missing action
        with pytest.raises(Exception):
            await client.call_tool("manage_maintenance_mode", {})


@pytest.mark.asyncio
async def test_manage_tests():
    """Test managing tests with different actions."""
    async with Client(mcp_server_config) as client:
        # Test case 1: Create a new test
        await create_flow("healthcheck.yaml", client)
        response_json = await create_test("healthcheck_test.yaml", client)
        print("Create test response:", response_json)

        # Verify create response structure
        assert "id" in response_json
        assert response_json["id"] == "test_healthcheck"
        assert "namespace" in response_json
        assert response_json["namespace"] == "tutorial"
        assert "flowId" in response_json
        assert response_json["flowId"] == "healthcheck"
        assert "testCases" in response_json
        assert len(response_json["testCases"]) == 2
        test_case_1 = response_json["testCases"][0]
        assert test_case_1["id"] == "server_should_be_reachable"
        assert test_case_1["type"] == "io.kestra.core.tests.flow.UnitTest"
        assert "assertions" in test_case_1
        assert len(test_case_1["assertions"]) == 1

        # Test case 2: Run the test
        result = await client.call_tool(
            "manage_tests",
            {"action": "run", "namespace": "tutorial", "id_": "test_healthcheck"},
        )
        response_json = json.loads(result[0].text)
        print("Run test response:", response_json)

        # Verify run response structure
        assert "id" in response_json
        assert "testSuiteId" in response_json
        assert response_json["testSuiteId"] == "test_healthcheck"
        assert "namespace" in response_json
        assert response_json["namespace"] == "tutorial"
        assert "flowId" in response_json
        assert response_json["flowId"] == "healthcheck"
        assert "state" in response_json
        assert response_json["state"] in ["ERROR", "SUCCESS", "FAILED", "SKIPPED"]
        assert "results" in response_json
        assert isinstance(response_json["results"], list)
        assert len(response_json["results"]) == 2

        # Verify first test result
        result1 = response_json["results"][0]
        assert "testId" in result1
        assert "testType" in result1
        assert "executionId" in result1
        assert "url" in result1
        assert "state" in result1
        assert "assertionResults" in result1
        assert isinstance(result1["assertionResults"], list)
        assert len(result1["assertionResults"]) > 0
        assert "errors" in result1
        assert isinstance(result1["errors"], list)
        assert "fixtures" in result1

        # Test case 3: Delete the test
        result = await client.call_tool(
            "manage_tests",
            {"action": "delete", "namespace": "tutorial", "id_": "test_healthcheck"},
        )
        response_json = json.loads(result[0].text)
        assert response_json == {}  # Verify empty dictionary response

        # Test case 4: Error - missing required parameters for create
        with pytest.raises(Exception):
            await client.call_tool("manage_tests", {"action": "create"})

        # Test case 5: Error - missing required parameters for run
        with pytest.raises(Exception):
            await client.call_tool("manage_tests", {"action": "run"})

        # Test case 6: Error - missing required parameters for delete
        with pytest.raises(Exception):
            await client.call_tool("manage_tests", {"action": "delete"})

        # Test case 7: Error - invalid action
        with pytest.raises(Exception):
            await client.call_tool("manage_tests", {"action": "invalid_action"})


@pytest.mark.asyncio
async def test_manage_groups():
    """Test managing groups with different actions."""
    async with Client(mcp_server_config) as client:
        # 1. Create a group
        unique_name = f"test-mcp-group-{int(time.time())}"
        create_result = await client.call_tool(
            "manage_group",
            {
                "action": "create",
                "name": unique_name,
                "description": "Initial description",
                "role": "admin",
            },
        )
        group = json.loads(create_result[0].text)
        print("Create group response:", group)
        group_id = group["id"]
        assert group_id is not None

        # 2. Get the group by id
        get_result = await client.call_tool(
            "manage_group", {"action": "get", "id_": group_id}
        )
        group_got = json.loads(get_result[0].text)
        print("Get group response:", group_got)
        assert group_got["id"] == group_id
        assert group_got["name"] == unique_name
        assert group_got["description"] == "Initial description"

        # 3. Update the group name and description
        updated_name = unique_name + "-updated"
        updated_description = "Updated description"
        update_result = await client.call_tool(
            "manage_group",
            {
                "action": "update",
                "id_": group_id,
                "name": updated_name,
                "description": updated_description,
            },
        )
        group_updated = json.loads(update_result[0].text)
        print("Update group response:", group_updated)
        assert group_updated["id"] == group_id
        assert group_updated["name"] == updated_name
        assert group_updated["description"] == updated_description

        # 4. Get the group again and verify update
        get_result2 = await client.call_tool(
            "manage_group", {"action": "get", "id_": group_id}
        )
        group_got2 = json.loads(get_result2[0].text)
        print("Get group after update response:", group_got2)
        assert group_got2["name"] == updated_name
        assert group_got2["description"] == updated_description

        # 5. Delete the group
        delete_result = await client.call_tool(
            "manage_group", {"action": "delete", "id_": group_id}
        )
        try:
            delete_response = json.loads(delete_result[0].text)
        except Exception:
            delete_response = {}
        print("Delete group response:", delete_response)
        assert delete_response == {} or "id" not in delete_response

        # 6. Attempt to get the group again and expect an error
        with pytest.raises(Exception):
            await client.call_tool("manage_group", {"action": "get", "id_": group_id})


@pytest.mark.asyncio
async def test_manage_apps():
    """Test managing apps with different actions."""
    async with Client(mcp_server_config) as client:
        # Test case 1: Create a new app
        await create_flow("app_get_data_flow.yaml", client)
        response_json = await create_app("app_get_data.yaml", client)
        print("Create app response:", response_json)

        # Verify create response structure
        assert "uid" in response_json
        assert "name" in response_json
        assert response_json["name"] == "Form to request and download data"
        assert "namespace" in response_json
        assert response_json["namespace"] == "company.team"
        assert "tags" in response_json
        assert isinstance(response_json["tags"], list)
        assert "Reporting" in response_json["tags"]
        assert "Analytics" in response_json["tags"]

        # Handle both response structures (disabled for new creation, enabled for updates)
        if "disabled" in response_json:
            assert response_json["disabled"] is False
            assert "source" in response_json
        elif "enabled" in response_json:
            assert response_json["enabled"] is True
            assert "type" in response_json
            assert response_json["type"] == "io.kestra.plugin.ee.apps.Execution"
        else:
            assert False, "Response must contain either 'disabled' or 'enabled' field"

        assert "created" in response_json
        assert "updated" in response_json

        uid = response_json["uid"]

        # Test case 2: Disable the app
        result = await client.call_tool(
            "manage_apps", {"action": "disable", "uid": uid}
        )
        response_json = json.loads(result[0].text)
        print("Disable app response:", response_json)

        # Verify disable response structure
        assert "uid" in response_json
        assert response_json["uid"] == uid
        assert "id" in response_json
        assert response_json["id"] == "app_get_data"
        assert "name" in response_json
        assert response_json["name"] == "Form to request and download data"
        assert "type" in response_json
        assert response_json["type"] == "io.kestra.plugin.ee.apps.Execution"
        assert "namespace" in response_json
        assert response_json["namespace"] == "company.team"
        assert "tags" in response_json
        assert isinstance(response_json["tags"], list)
        assert "enabled" in response_json
        assert response_json["enabled"] is False
        assert "created" in response_json
        assert "updated" in response_json

        # Test case 3: Enable the app
        result = await client.call_tool("manage_apps", {"action": "enable", "uid": uid})
        response_json = json.loads(result[0].text)
        print("Enable app response:", response_json)

        # Verify enable response structure
        assert "uid" in response_json
        assert response_json["uid"] == uid
        assert "id" in response_json
        assert response_json["id"] == "app_get_data"
        assert "name" in response_json
        assert response_json["name"] == "Form to request and download data"
        assert "type" in response_json
        assert response_json["type"] == "io.kestra.plugin.ee.apps.Execution"
        assert "namespace" in response_json
        assert response_json["namespace"] == "company.team"
        assert "tags" in response_json
        assert isinstance(response_json["tags"], list)
        assert "enabled" in response_json
        assert response_json["enabled"] is True
        assert "created" in response_json
        assert "updated" in response_json

        # Test case 4: Delete the app
        result = await client.call_tool("manage_apps", {"action": "delete", "uid": uid})
        response_json = json.loads(result[0].text)
        print("Delete app response:", response_json)
        assert response_json == {}  # Verify empty dictionary response

        # Test case 5: Error - missing required parameters for create
        with pytest.raises(Exception):
            await client.call_tool("manage_apps", {"action": "create"})

        # Test case 6: Error - missing required parameters for enable
        with pytest.raises(Exception):
            await client.call_tool("manage_apps", {"action": "enable"})

        # Test case 7: Error - missing required parameters for disable
        with pytest.raises(Exception):
            await client.call_tool("manage_apps", {"action": "disable"})

        # Test case 8: Error - missing required parameters for delete
        with pytest.raises(Exception):
            await client.call_tool("manage_apps", {"action": "delete"})

        # Test case 9: Error - invalid action
        with pytest.raises(Exception):
            await client.call_tool("manage_apps", {"action": "invalid_action"})


@pytest.mark.asyncio
async def test_search_apps():
    """Test searching for apps with different parameters."""
    async with Client(mcp_server_config) as client:
        # First check if app exists and delete it if it does
        result = await client.call_tool(
            "search_apps", {"q": "app_newsletter", "namespace": "company.team"}
        )
        response_json = json.loads(result[0].text)
        if len(response_json["results"]) > 0:
            for app in response_json["results"]:
                await client.call_tool(
                    "manage_apps", {"action": "delete", "uid": app["uid"]}
                )

        # Create a new app to search for
        await create_flow("app_newsletter_flow.yaml", client)
        await create_app("app_newsletter.yaml", client)

        # Test case 1: Basic search with default parameters
        result = await client.call_tool("search_apps", {})
        response_json = json.loads(result[0].text)
        print("Basic search response:", response_json)

        # Verify response structure
        assert "results" in response_json
        assert "total" in response_json
        assert isinstance(response_json["results"], list)
        assert len(response_json["results"]) > 0

        # Verify first result structure
        first_result = response_json["results"][0]
        assert "uid" in first_result
        assert "id" in first_result
        assert "name" in first_result
        assert "type" in first_result
        assert "namespace" in first_result
        # Tags are optional, so we don't assert their presence
        assert "enabled" in first_result
        assert "created" in first_result
        assert "updated" in first_result

        # Test case 2: Search with specific query
        result = await client.call_tool("search_apps", {"q": "app_newsletter"})
        response_json = json.loads(result[0].text)
        print("Query search response:", response_json)
        assert len(response_json["results"]) > 0
        assert response_json["results"][0]["id"] == "app_newsletter"

        # Test case 3: Search with namespace filter
        result = await client.call_tool("search_apps", {"namespace": "company.team"})
        response_json = json.loads(result[0].text)
        print("Namespace search response:", response_json)
        assert len(response_json["results"]) > 0
        assert response_json["results"][0]["namespace"] == "company.team"

        # Test case 4: Search with tags filter
        result = await client.call_tool("search_apps", {"tags": ["Newsletter"]})
        response_json = json.loads(result[0].text)
        print("Tags search response:", response_json)
        assert len(response_json["results"]) > 0
        # Find the app with tags
        app_with_tags = next(
            (app for app in response_json["results"] if "tags" in app), None
        )
        assert app_with_tags is not None, "No app with tags found"
        assert "Newsletter" in app_with_tags["tags"]

        # Test case 5: Search with flowId filter
        result = await client.call_tool("search_apps", {"flowId": "newsletter"})
        response_json = json.loads(result[0].text)
        print("FlowId search response:", response_json)
        assert len(response_json["results"]) > 0
        # Note: flowId might not be directly in the response, but we can verify the app exists

        # Test case 6: Search with pagination
        result = await client.call_tool("search_apps", {"page": 1, "size": 5})
        response_json = json.loads(result[0].text)
        print("Pagination search response:", response_json)
        assert len(response_json["results"]) <= 5  # Should not exceed page size

        # Test case 7: Search with multiple filters
        result = await client.call_tool(
            "search_apps",
            {
                "q": "app_newsletter",
                "namespace": "company.team",
                "tags": ["Newsletter", "Marketing"],
                "page": 1,
                "size": 10,
            },
        )
        response_json = json.loads(result[0].text)
        print("Multiple filters search response:", response_json)
        assert len(response_json["results"]) > 0
        first_result = response_json["results"][0]
        assert first_result["id"] == "app_newsletter"
        assert first_result["namespace"] == "company.team"
        assert "tags" in first_result
        assert "Newsletter" in first_result["tags"]
        assert "Marketing" in first_result["tags"]

        # Clean up - delete the app we created
        result = await client.call_tool(
            "search_apps", {"q": "app_newsletter", "namespace": "company.team"}
        )
        response_json = json.loads(result[0].text)
        if len(response_json["results"]) > 0:
            for app in response_json["results"]:
                await client.call_tool(
                    "manage_apps", {"action": "delete", "uid": app["uid"]}
                )


@pytest.mark.asyncio
async def test_manage_announcements():
    """Test managing announcements with different actions."""
    async with Client(mcp_server_config) as client:
        # Test case 1a: Create a new announcement with only message
        result = await client.call_tool(
            "manage_announcements",
            {"action": "create", "message": "Test announcement minimal"},
        )
        response_json = json.loads(result[0].text)
        print("Create announcement (minimal) response:", response_json)
        assert "id" in response_json
        assert "message" in response_json
        assert response_json["message"] == "Test announcement minimal"
        first_announcement_id = response_json["id"]

        # Test case 1b: Create a new announcement with all fields
        current_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        future_time = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 3600)
        )  # 1 hour from now
        result = await client.call_tool(
            "manage_announcements",
            {
                "action": "create",
                "message": "Test announcement",
                "type": "INFO",
                "startDate": current_time,
                "endDate": future_time,
                "active": True,
            },
        )
        response_json = json.loads(result[0].text)
        print("Create announcement response:", response_json)
        # Verify create response structure
        assert "id" in response_json
        assert "message" in response_json
        assert response_json["message"] == "Test announcement"
        assert "type" in response_json
        assert response_json["type"] == "INFO"
        assert "startDate" in response_json
        assert "endDate" in response_json
        assert "active" in response_json
        assert response_json["active"] is True

        second_announcement_id = response_json["id"]

        # Test case 2: List announcements
        result = await client.call_tool("manage_announcements", {"action": "list"})
        # Parse all announcements using list comprehension
        announcements = [
            json.loads(item.text) for item in result if hasattr(item, "text")
        ]
        print("List announcements response:", announcements)

        # Verify list response structure
        assert isinstance(announcements, list)
        assert len(announcements) >= 2
        # Find the announcement with id == second_announcement_id
        found = None
        for ann in announcements:
            assert "id" in ann
            assert "message" in ann
            if ann.get("id") == second_announcement_id:
                found = ann
        assert (
            found is not None
        ), f"Announcement with id {second_announcement_id} not found in list."
        # Only the announcement created with all fields should have these fields
        assert "type" in found
        assert "startDate" in found
        assert "endDate" in found
        assert "active" in found

        # Test case 3: Update announcement
        result = await client.call_tool(
            "manage_announcements",
            {
                "action": "update",
                "id_": second_announcement_id,
                "message": "Test announcement updated",
                "type": "WARNING",
                "startDate": current_time,
                "endDate": future_time,
                "active": False,
            },
        )
        response_json = json.loads(result[0].text)
        print("Update announcement response:", response_json)

        # Verify update response structure
        assert "id" in response_json
        assert response_json["id"] == second_announcement_id
        assert "message" in response_json
        assert response_json["message"] == "Test announcement updated"
        assert "type" in response_json
        assert response_json["type"] == "WARNING"
        assert "active" in response_json
        assert response_json["active"] is False

        # Test case 4: Delete announcement
        # Delete the announcement created in 1a
        delete_result = await client.call_tool(
            "manage_announcements",
            {"action": "delete", "id_": first_announcement_id},
        )
        delete_response_json = json.loads(delete_result[0].text)
        print("Delete announcement (minimal) response:", delete_response_json)
        assert delete_response_json == {}

        # Delete the announcement created in 1b
        delete_result = await client.call_tool(
            "manage_announcements", {"action": "delete", "id_": second_announcement_id}
        )
        delete_response_json = json.loads(delete_result[0].text)
        print("Delete announcement response:", delete_response_json)
        assert delete_response_json == {}  # Verify empty dictionary response

        # Test case 5: Error - missing required parameters for create
        with pytest.raises(Exception):
            await client.call_tool("manage_announcements", {"action": "create"})

        # Test case 6: Error - missing required parameters for update
        with pytest.raises(Exception):
            await client.call_tool("manage_announcements", {"action": "update"})

        # Test case 7: Error - missing required parameters for delete
        with pytest.raises(Exception):
            await client.call_tool("manage_announcements", {"action": "delete"})

        # Test case 8: Error - invalid action
        with pytest.raises(Exception):
            await client.call_tool("manage_announcements", {"action": "invalid_action"})


@pytest.mark.asyncio
async def test_invite_users():
    """Test inviting a user with different scenarios."""
    async with Client(mcp_server_config) as client:
        # Collect invitation ids from responses
        invite_ids = []

        # Test case 1: Basic invitation with no groups or role
        print("\n=== Test case 1: Basic invitation ===")
        result = await client.call_tool("invite_user", {"email": "test@kestra.io"})
        response_json = json.loads(result[0].text)
        print(f"First invitation response: {json.dumps(response_json, indent=2)}")
        assert "id" in response_json
        assert "email" in response_json
        assert response_json["email"] == "test@kestra.io"
        assert "status" in response_json
        assert response_json["status"] == "PENDING"
        assert "userType" in response_json
        assert response_json["userType"] == "STANDARD"
        assert "link" in response_json
        assert "sentAt" in response_json
        assert "expiredAt" in response_json
        assert "deleted" in response_json
        assert "isExpired" in response_json
        assert "superAdmin" in response_json
        assert "tenantId" in response_json
        assert "groupIds" in response_json
        assert isinstance(response_json["groupIds"], list)
        invite_ids.append(response_json["id"])

        # Try to invite the same user again - should return existing invitation
        print("\n=== Test case 2: Duplicate invitation ===")
        result = await client.call_tool("invite_user", {"email": "test@kestra.io"})
        response_json = json.loads(result[0].text)
        print(f"Second invitation response: {json.dumps(response_json, indent=2)}")
        assert response_json["email"] == "test@kestra.io"
        assert response_json["status"] == "PENDING"
        invite_ids.append(response_json["id"])

        # Test case 2: Invitation with IAM role
        print("\n=== Test case 4: Invitation with IAM role ===")
        result = await client.call_tool(
            "invite_user", {"email": "test2@kestra.io", "role": "admin"}
        )
        response_json = json.loads(result[0].text)
        print(f"Role invitation response: {json.dumps(response_json, indent=2)}")
        assert "bindings" in response_json
        assert len(response_json["bindings"]) == 1
        binding = response_json["bindings"][0]
        assert binding["type"] == "USER"
        assert binding["roleId"].startswith("admin_")
        assert binding["deleted"] is False
        invite_ids.append(response_json["id"])

        # === Create a group ===
        group_name = f"test-invite-group-{int(time.time())}"
        create_result = await client.call_tool(
            "manage_group",
            {
                "action": "create",
                "name": group_name,
                "description": "Invite group",
                "role": "admin",
            },
        )
        group = json.loads(create_result[0].text)
        group_id = group["id"]
        print(f"Group ID: {group_id}")

        try:
            # === Test case 3: Invitation with groups ===
            print("\n=== Test case 3: Invitation with groups ===")
            result = await client.call_tool(
                "invite_user", {"email": "test3@kestra.io", "group_names": [group_name]}
            )
            response_json = json.loads(result[0].text)
            print(f"Third invitation response: {json.dumps(response_json, indent=2)}")
            assert response_json["email"] == "test3@kestra.io"
            assert response_json["status"] == "PENDING"
            assert "groupIds" in response_json
            assert len(response_json["groupIds"]) > 0  # Should be added to the group
            assert (
                group_id in response_json["groupIds"]
            )  # Verify the correct group ID is assigned
            invite_ids.append(response_json["id"])

            # Test case 4: Invitation with both groups and role
            print("\n=== Test case 5: Invitation with groups and role ===")
            result = await client.call_tool(
                "invite_user",
                {
                    "email": "test4@kestra.io",
                    "group_names": [group_name],
                    "role": "developer",
                },
            )
            response_json = json.loads(result[0].text)
            print(
                f"Combined invitation response: {json.dumps(response_json, indent=2)}"
            )
            assert "groupIds" in response_json
            assert len(response_json["groupIds"]) > 0
            assert "bindings" in response_json
            assert len(response_json["bindings"]) == 1
            binding = response_json["bindings"][0]
            assert binding["type"] == "USER"
            assert binding["roleId"].startswith("developer_")
            assert binding["deleted"] is False
            invite_ids.append(response_json["id"])

        # Cleanup: delete the group
        finally:
            delete_result = await client.call_tool(
                "manage_group", {"action": "delete", "id_": group_id}
            )
            try:
                delete_response = json.loads(delete_result[0].text)
            except Exception:
                delete_response = {}
            print("Delete group response (cleanup):", delete_response)

            # Cleanup: delete all invitations created in this test
            for invite_id in set(invite_ids):
                del_result = await client.call_tool(
                    "manage_invitations", {"action": "delete", "id_": invite_id}
                )
                try:
                    del_response = json.loads(del_result[0].text)
                except Exception:
                    del_response = {}
                print(f"Delete invitation {invite_id} response:", del_response)

        # Test case 5: Error case - non-existent group
        print("\n=== Test case 6: Error case - non-existent group ===")
        with pytest.raises(Exception):
            await client.call_tool(
                "invite_user",
                {"email": "test5@kestra.io", "group_names": ["NonExistentGroup"]},
            )

        # Test case 6: Error case - invalid role
        with pytest.raises(Exception):
            await client.call_tool(
                "invite_user", {"email": "test6@kestra.io", "role": "INVALID_ROLE"}
            )
            # If the API still returns a response, try to capture the id
            try:
                response_json = json.loads(result[0].text)
                if "id" in response_json:
                    invite_ids.append(response_json["id"])
            except Exception:
                pass


@pytest.mark.asyncio
async def test_license_info():
    """Response example:
    {'type': 'CUSTOMER', 'expiry': '2030-05-02T00:00:00.000+00:00', 'expired': False}
    """
    async with Client(mcp_server_config) as client:
        result = await client.call_tool("get_instance_info", {"info": "license_info"})
        response_json = json.loads(result[0].text)
        assert "type" in response_json and isinstance(response_json["type"], str)
        assert "expiry" in response_json and isinstance(response_json["expiry"], str)
        assert "expired" in response_json and isinstance(response_json["expired"], bool)


@pytest.mark.asyncio
async def test_active_services():
    async with Client(mcp_server_config) as client:
        """Response example:
        {'total': 5, 'services': [{'type': 'WEBSERVER', 'total': 1}, {'type': 'SCHEDULER', 'total': 1}, {'type': 'WORKER', 'total': 1}, {'type': 'EXECUTOR', 'total': 1}, {'type': 'INDEXER', 'total': 1}]}
        """
        result = await client.call_tool(
            "get_instance_info", {"info": "active_services"}
        )
        response_json = json.loads(result[0].text)
        assert "total" in response_json and isinstance(response_json["total"], int)
        assert "services" in response_json and isinstance(
            response_json["services"], list
        )
        for service in response_json["services"]:
            assert "type" in service and isinstance(service["type"], str)
            assert service["type"] in [
                "WEBSERVER",
                "SCHEDULER",
                "WORKER",
                "EXECUTOR",
                "INDEXER",
            ]
            assert "total" in service and isinstance(service["total"], int)


@pytest.mark.asyncio
async def test_configuration():
    async with Client(mcp_server_config) as client:
        result = await client.call_tool("get_instance_info", {"info": "configuration"})
        response_json = json.loads(result[0].text)
        # Top-level required fields
        assert "uuid" in response_json and isinstance(response_json["uuid"], str)
        assert "version" in response_json and isinstance(response_json["version"], str)
        assert "commitId" in response_json and isinstance(
            response_json["commitId"], str
        )
        assert "commitDate" in response_json and isinstance(
            response_json["commitDate"], str
        )
        assert "isCustomDashboardsEnabled" in response_json and isinstance(
            response_json["isCustomDashboardsEnabled"], bool
        )
        assert "isTaskRunEnabled" in response_json and isinstance(
            response_json["isTaskRunEnabled"], bool
        )
        assert "isAnonymousUsageEnabled" in response_json and isinstance(
            response_json["isAnonymousUsageEnabled"], bool
        )
        assert "isTemplateEnabled" in response_json and isinstance(
            response_json["isTemplateEnabled"], bool
        )
        assert "environment" in response_json and isinstance(
            response_json["environment"], dict
        )
        assert "name" in response_json["environment"] and isinstance(
            response_json["environment"]["name"], str
        )
        assert "url" in response_json and isinstance(response_json["url"], str)
        assert "isBasicAuthEnabled" in response_json and isinstance(
            response_json["isBasicAuthEnabled"], bool
        )
        assert "systemNamespace" in response_json and isinstance(
            response_json["systemNamespace"], str
        )
        assert "hiddenLabelsPrefixes" in response_json and isinstance(
            response_json["hiddenLabelsPrefixes"], list
        )
        assert "tenants" in response_json and isinstance(response_json["tenants"], dict)
        assert "storageByTenant" in response_json["tenants"] and isinstance(
            response_json["tenants"]["storageByTenant"], bool
        )
        assert "secretByTenant" in response_json["tenants"] and isinstance(
            response_json["tenants"]["secretByTenant"], bool
        )
        assert "secretsEnabled" in response_json and isinstance(
            response_json["secretsEnabled"], bool
        )
        assert "supportedStorages" in response_json and isinstance(
            response_json["supportedStorages"], list
        )
        assert "supportedSecrets" in response_json and isinstance(
            response_json["supportedSecrets"], list
        )
        assert "pluginManagementEnabled" in response_json and isinstance(
            response_json["pluginManagementEnabled"], bool
        )
        assert "pluginCustomEnabled" in response_json and isinstance(
            response_json["pluginCustomEnabled"], bool
        )
        assert "mailServiceEnabled" in response_json and isinstance(
            response_json["mailServiceEnabled"], bool
        )
        assert "outputsInInternalStorageEnabled" in response_json and isinstance(
            response_json["outputsInInternalStorageEnabled"], bool
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_manage_maintenance_mode())
    asyncio.run(test_search_users())
    asyncio.run(test_manage_worker_groups())
    asyncio.run(test_manage_maintenance_mode())
    asyncio.run(test_manage_tests())
    asyncio.run(test_manage_groups())
    asyncio.run(test_manage_apps())
    asyncio.run(test_search_apps())
    asyncio.run(test_manage_announcements())
    asyncio.run(test_invite_users())
    asyncio.run(test_license_info())
    asyncio.run(test_active_services())
    asyncio.run(test_configuration())
