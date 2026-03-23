import pytest
from fastmcp import Client
from dotenv import load_dotenv
import json
from pathlib import Path
from test_utils import create_flow, create_test, create_app
import time
import os
import httpx


load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

# Check if EE tools are disabled
DISABLED_TOOLS = os.getenv("KESTRA_MCP_DISABLED_TOOLS", "").split(",")
DISABLED_TOOLS = [tool.strip() for tool in DISABLED_TOOLS if tool.strip()]
EE_TOOLS_DISABLED = "ee" in DISABLED_TOOLS


def _make_http_clients() -> tuple[httpx.AsyncClient, httpx.AsyncClient]:
    """Create httpx clients for tenant-scoped and root API calls."""
    base = os.getenv("KESTRA_BASE_URL", "http://localhost:8080/api/v1")
    tenant = os.getenv("KESTRA_TENANT_ID")
    headers: dict[str, str] = {}
    auth = None
    if (user := os.getenv("KESTRA_USERNAME")) and (pwd := os.getenv("KESTRA_PASSWORD")):
        auth = httpx.BasicAuth(user, pwd)
    elif token := os.getenv("KESTRA_API_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    if tenant:
        headers["X-Kestra-Tenant"] = tenant
    root_client = httpx.AsyncClient(base_url=base, auth=auth, headers=headers)
    tenant_base = f"{base.rstrip('/')}/{tenant}" if tenant else base
    tenant_client = httpx.AsyncClient(base_url=tenant_base, auth=auth, headers=headers)
    return root_client, tenant_client


async def _cleanup_test_user_access(emails: list[str]):
    """Remove global users, tenant access, and invitations for test emails so invite tests start clean."""
    root_http, tenant_http = _make_http_clients()
    async with root_http, tenant_http:
        for email in emails:
            # Delete any existing invitations for this email
            try:
                resp = await tenant_http.get(f"/invitations/email/{email}")
                if resp.status_code == 200:
                    invites = resp.json()
                    # API may return nested list [[...]]
                    if isinstance(invites, list):
                        for item in invites:
                            if isinstance(item, list):
                                for inv in item:
                                    if isinstance(inv, dict) and "id" in inv:
                                        await tenant_http.delete(f"/invitations/{inv['id']}")
                            elif isinstance(item, dict) and "id" in item:
                                await tenant_http.delete(f"/invitations/{item['id']}")
            except Exception:
                pass
            # Remove tenant access if user already has it
            try:
                resp = await tenant_http.get("/tenant-access", params={"page": 1, "size": 100})
                if resp.status_code == 200:
                    data = resp.json()
                    users = data.get("results", [])
                    for user in users:
                        if user.get("username") == email and "id" in user:
                            await tenant_http.delete(f"/tenant-access/{user['id']}")
            except Exception:
                pass
            # Delete the global user so invitation flow works (not direct tenant access)
            try:
                resp = await root_http.get("/users", params={"q": email, "page": 1, "size": 10})
                if resp.status_code == 200:
                    data = resp.json()
                    for user in data.get("results", []):
                        if user.get("username") == email and "id" in user:
                            await root_http.delete(f"/users/{user['id']}")
            except Exception:
                pass




@pytest.mark.asyncio
async def test_manage_tests(kestra_client, cleanup):
    """Test managing tests with different actions."""
    if EE_TOOLS_DISABLED:
        pytest.skip("EE tools are disabled")
    client = kestra_client
    # Test case 1: Create a new test
    await create_flow("healthcheck.yaml", client, cleanup)
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
    response_json = json.loads(result.content[0].text)
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
    response_json = json.loads(result.content[0].text)
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
async def test_manage_groups(kestra_client):
    if EE_TOOLS_DISABLED:
        pytest.skip("EE tools are disabled")
    """Test managing groups with different actions."""
    client = kestra_client
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
    group = json.loads(create_result.content[0].text)
    print("Create group response:", group)
    group_id = group["id"]
    assert group_id is not None

    # 2. Get the group by id
    get_result = await client.call_tool(
        "manage_group", {"action": "get", "id_": group_id}
    )
    group_got = json.loads(get_result.content[0].text)
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
    group_updated = json.loads(update_result.content[0].text)
    print("Update group response:", group_updated)
    assert group_updated["id"] == group_id
    assert group_updated["name"] == updated_name
    assert group_updated["description"] == updated_description

    # 4. Get the group again and verify update
    get_result2 = await client.call_tool(
        "manage_group", {"action": "get", "id_": group_id}
    )
    group_got2 = json.loads(get_result2.content[0].text)
    print("Get group after update response:", group_got2)
    assert group_got2["name"] == updated_name
    assert group_got2["description"] == updated_description

    # 5. Delete the group
    delete_result = await client.call_tool(
        "manage_group", {"action": "delete", "id_": group_id}
    )
    try:
        delete_response = json.loads(delete_result.content[0].text)
    except Exception:
        delete_response = {}
    print("Delete group response:", delete_response)
    assert delete_response == {} or "id" not in delete_response

    # 6. Attempt to get the group again and expect an error
    with pytest.raises(Exception):
        await client.call_tool("manage_group", {"action": "get", "id_": group_id})


@pytest.mark.asyncio
async def test_manage_apps(kestra_client, cleanup):
    if EE_TOOLS_DISABLED:
        pytest.skip("EE tools are disabled")
    """Test managing apps with different actions."""
    client = kestra_client
    # Test case 1: Create a new app
    await create_flow("app_get_data_flow.yaml", client, cleanup)
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
        # The enabled field might be False for newly created apps
        assert isinstance(response_json["enabled"], bool)
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
    response_json = json.loads(result.content[0].text)
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
    response_json = json.loads(result.content[0].text)
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
    response_json = json.loads(result.content[0].text)
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
async def test_search_apps(kestra_client, cleanup):
    if EE_TOOLS_DISABLED:
        pytest.skip("EE tools are disabled")
    """Test searching for apps with different parameters."""
    client = kestra_client
    # First check if app exists and delete it if it does
    result = await client.call_tool(
        "search_apps", {"q": "app_newsletter", "namespace": "company.team"}
    )
    response_json = json.loads(result.content[0].text)
    if len(response_json["results"]) > 0:
        for app in response_json["results"]:
            await client.call_tool(
                "manage_apps", {"action": "delete", "uid": app["uid"]}
            )

    # Create a new app to search for
    await create_flow("app_newsletter_flow.yaml", client, cleanup)
    await create_app("app_newsletter.yaml", client)

    # Test case 1: Basic search with default parameters
    result = await client.call_tool("search_apps", {})
    response_json = json.loads(result.content[0].text)
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
    response_json = json.loads(result.content[0].text)
    print("Query search response:", response_json)
    assert len(response_json["results"]) > 0
    assert response_json["results"][0]["id"] == "app_newsletter"

    # Test case 3: Search with namespace filter
    result = await client.call_tool("search_apps", {"namespace": "company.team"})
    response_json = json.loads(result.content[0].text)
    print("Namespace search response:", response_json)
    assert len(response_json["results"]) > 0
    assert response_json["results"][0]["namespace"] == "company.team"

    # Test case 4: Search with tags filter
    result = await client.call_tool("search_apps", {"tags": ["Newsletter"]})
    response_json = json.loads(result.content[0].text)
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
    response_json = json.loads(result.content[0].text)
    print("FlowId search response:", response_json)
    assert len(response_json["results"]) > 0
    # Note: flowId might not be directly in the response, but we can verify the app exists

    # Test case 6: Search with pagination
    result = await client.call_tool("search_apps", {"page": 1, "size": 5})
    response_json = json.loads(result.content[0].text)
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
    response_json = json.loads(result.content[0].text)
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
    response_json = json.loads(result.content[0].text)
    if len(response_json["results"]) > 0:
        for app in response_json["results"]:
            await client.call_tool(
                "manage_apps", {"action": "delete", "uid": app["uid"]}
            )


@pytest.mark.asyncio
async def test_manage_announcements(kestra_client):
    if EE_TOOLS_DISABLED:
        pytest.skip("EE tools are disabled")
    """Test managing announcements with different actions."""
    client = kestra_client
    # Test case 1a: Create a new announcement with only message
    result = await client.call_tool(
        "manage_announcements",
        {"action": "create", "message": "Test announcement minimal"},
    )
    response_json = json.loads(result.content[0].text)
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
    response_json = json.loads(result.content[0].text)
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
    raw = json.loads(result.content[0].text)
    # Handle both flat list and nested list (single content item containing array)
    announcements = raw if isinstance(raw, list) else [raw]
    # Flatten if nested: [[ann1, ann2]] → [ann1, ann2]
    if announcements and isinstance(announcements[0], list):
        announcements = announcements[0]
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
    response_json = json.loads(result.content[0].text)
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
    delete_response_json = json.loads(delete_result.content[0].text)
    print("Delete announcement (minimal) response:", delete_response_json)
    assert delete_response_json == {}

    # Delete the announcement created in 1b
    delete_result = await client.call_tool(
        "manage_announcements", {"action": "delete", "id_": second_announcement_id}
    )
    delete_response_json = json.loads(delete_result.content[0].text)
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
async def test_invite_users(kestra_client):
    if EE_TOOLS_DISABLED:
        pytest.skip("EE tools are disabled")
    """Test inviting a user with different scenarios."""
    # Clean up leftover state from previous test runs
    test_emails = ["test@kestra.io", "test2@kestra.io", "test3@kestra.io",
                   "test4@kestra.io", "test5@kestra.io", "test6@kestra.io"]
    await _cleanup_test_user_access(test_emails)

    client = kestra_client
    # Collect invitation ids from responses
    invite_ids = []

    # Test case 1: Basic invitation with no groups or role
    print("\n=== Test case 1: Basic invitation ===")
    result = await client.call_tool("invite_user", {"email": "test@kestra.io"})
    response_json = json.loads(result.content[0].text)
    print(f"First invitation response: {json.dumps(response_json, indent=2)}")
    assert "id" in response_json
    assert "email" in response_json
    assert response_json["email"] == "test@kestra.io"
    assert "status" in response_json
    assert response_json["status"] == "PENDING"
    # Handle both old and new API response structures
    if "userType" in response_json:
        assert response_json["userType"] == "STANDARD"
    if "link" in response_json:
        assert isinstance(response_json["link"], str)
    if "sentAt" in response_json:
        assert isinstance(response_json["sentAt"], str)
    if "expiredAt" in response_json:
        assert isinstance(response_json["expiredAt"], str)
    if "deleted" in response_json:
        assert isinstance(response_json["deleted"], bool)
    if "isExpired" in response_json:
        assert isinstance(response_json["isExpired"], bool)
    if "superAdmin" in response_json:
        assert isinstance(response_json["superAdmin"], bool)
    if "tenantId" in response_json:
        assert isinstance(response_json["tenantId"], str)
    if "groupIds" in response_json:
        assert isinstance(response_json["groupIds"], list)
    invite_ids.append(response_json["id"])

    # Try to invite the same user again - should return existing invitation
    print("\n=== Test case 2: Duplicate invitation ===")
    result = await client.call_tool("invite_user", {"email": "test@kestra.io"})
    response_json = json.loads(result.content[0].text)
    print(f"Second invitation response: {json.dumps(response_json, indent=2)}")
    assert response_json["email"] == "test@kestra.io"
    assert response_json["status"] == "PENDING"
    invite_ids.append(response_json["id"])

    # Test case 2: Invitation with IAM role
    print("\n=== Test case 4: Invitation with IAM role ===")
    result = await client.call_tool(
        "invite_user", {"email": "test2@kestra.io", "role": "admin"}
    )
    response_json = json.loads(result.content[0].text)
    print(f"Role invitation response: {json.dumps(response_json, indent=2)}")
    # Handle both old and new API response structures
    # Note: In Kestra 0.24+, bindings are handled differently and may not be present
    if "bindings" in response_json:
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
    group = json.loads(create_result.content[0].text)
    group_id = group["id"]
    print(f"Group ID: {group_id}")

    try:
        # === Test case 3: Invitation with groups ===
        print("\n=== Test case 3: Invitation with groups ===")
        result = await client.call_tool(
            "invite_user", {"email": "test3@kestra.io", "group_names": [group_name]}
        )
        response_json = json.loads(result.content[0].text)
        print(f"Third invitation response: {json.dumps(response_json, indent=2)}")
        assert response_json["email"] == "test3@kestra.io"
        assert response_json["status"] == "PENDING"
        # Handle both old and new API response structures
        if "groupIds" in response_json:
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
        response_json = json.loads(result.content[0].text)
        print(
            f"Combined invitation response: {json.dumps(response_json, indent=2)}"
        )
        # Handle both old and new API response structures
        # Note: In Kestra 0.24+, bindings are handled differently and may not be present
        if "groupIds" in response_json:
            assert len(response_json["groupIds"]) > 0
        if "bindings" in response_json:
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
            delete_response = json.loads(delete_result.content[0].text)
        except Exception:
            delete_response = {}
        print("Delete group response (cleanup):", delete_response)

        # Cleanup: delete all invitations created in this test
        for invite_id in set(invite_ids):
            del_result = await client.call_tool(
                "manage_invitations", {"action": "delete", "id_": invite_id}
            )
            try:
                del_response = json.loads(del_result.content[0].text)
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
            response_json = json.loads(result.content[0].text)
            if "id" in response_json:
                invite_ids.append(response_json["id"])
        except Exception:
            pass


@pytest.mark.asyncio
async def test_manage_invitations(kestra_client):
    if EE_TOOLS_DISABLED:
        pytest.skip("EE tools are disabled")
    """Test manage_invitations get and delete actions."""
    client = kestra_client
    test_emails = ["test-inv-mgmt@kestra.io"]
    await _cleanup_test_user_access(test_emails)

    # Create an invitation to work with
    result = await client.call_tool(
        "invite_user", {"email": "test-inv-mgmt@kestra.io"}
    )
    response = json.loads(result.content[0].text)
    invite_id = response.get("id")
    assert invite_id is not None, f"Expected invitation id, got: {response}"

    # Test get action
    get_result = await client.call_tool(
        "manage_invitations", {"action": "get", "id_": invite_id}
    )
    get_response = json.loads(get_result.content[0].text)
    assert get_response["id"] == invite_id
    assert get_response["email"] == "test-inv-mgmt@kestra.io"

    # Test delete action
    del_result = await client.call_tool(
        "manage_invitations", {"action": "delete", "id_": invite_id}
    )
    # Delete returns empty dict on success
    del_response = json.loads(del_result.content[0].text)
    assert isinstance(del_response, dict)


@pytest.mark.asyncio
async def test_license_info(kestra_client):
    if EE_TOOLS_DISABLED:
        pytest.skip("EE tools are disabled")
    """Response example:
    {'type': 'CUSTOMER', 'expiry': '2030-05-02T00:00:00.000+00:00', 'expired': False}
    """
    client = kestra_client
    result = await client.call_tool("get_instance_info", {"info": "license_info"})
    response_json = json.loads(result.content[0].text)
    assert "type" in response_json and isinstance(response_json["type"], str)
    assert "expiry" in response_json and isinstance(response_json["expiry"], str)
    assert "expired" in response_json and isinstance(response_json["expired"], bool)


@pytest.mark.asyncio
async def test_configuration(kestra_client):
    if EE_TOOLS_DISABLED:
        pytest.skip("EE tools are disabled")
    client = kestra_client
    result = await client.call_tool("get_instance_info", {"info": "configuration"})
    response_json = json.loads(result.content[0].text)
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
    # isTaskRunEnabled is not present in all API versions
    if "isTaskRunEnabled" in response_json:
        assert isinstance(response_json["isTaskRunEnabled"], bool)
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
    # isBasicAuthEnabled is not present in all API versions
    if "isBasicAuthEnabled" in response_json:
        assert isinstance(response_json["isBasicAuthEnabled"], bool)
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

    asyncio.run(test_manage_tests())
    asyncio.run(test_manage_groups())
    asyncio.run(test_manage_apps())
    asyncio.run(test_search_apps())
    asyncio.run(test_manage_announcements())
    asyncio.run(test_invite_users())
    asyncio.run(test_license_info())
    asyncio.run(test_configuration())
