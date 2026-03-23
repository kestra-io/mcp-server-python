from fastmcp import FastMCP
import httpx
import re
import uuid
import yaml
from typing import Annotated, List, Literal, Optional
from pydantic import Field
import os
from kestra.utils import _root_api_url


def register_ee_tools(mcp: FastMCP, client: httpx.AsyncClient) -> None:
    # Version detection with caching
    _cached_api_version = None

    async def _detect_api_version():
        """Detect if we're running against Kestra 0.24+ or older version. Result is cached."""
        nonlocal _cached_api_version
        if _cached_api_version is not None:
            return _cached_api_version

        try:
            # Try 0.24+ endpoint first
            resp = await client.get("/me")
            if resp.status_code == 200:
                _cached_api_version = "0.24+"
                return _cached_api_version
        except Exception:
            pass

        try:
            # Try the old endpoint
            tenant = os.getenv("KESTRA_TENANT_ID")
            if tenant:
                resp = await client.get(f"/{tenant}/me")
                if resp.status_code == 200:
                    _cached_api_version = "0.23-"
                    return _cached_api_version
        except Exception:
            pass

        # Default to 0.24+
        _cached_api_version = "0.24+"
        return _cached_api_version

    def _extract_invitation(data):
        """Extract a single invitation dict from API response (which may be nested lists)."""
        if isinstance(data, dict) and "id" in data:
            return data
        if isinstance(data, list):
            for item in data:
                result = _extract_invitation(item)
                if result:
                    return result
        return None

    async def _fetch_invitation_by_email(email: str):
        """Fetch the most recent invitation for an email address."""
        try:
            resp = await client.get(f"/invitations/email/{email}")
            resp.raise_for_status()
            return _extract_invitation(resp.json())
        except Exception:
            return None

    @mcp.tool()
    async def invite_user(
        email: Annotated[
            str, Field(description="The email address of the user to invite")
        ],
        group_names: Annotated[
            Optional[List[str]],
            Field(description="Optional list of group names to add the user to"),
        ] = None,
        role: Annotated[
            Optional[Literal["admin", "developer", "editor", "launcher", "viewer"]],
            Field(
                description="Optional IAM role to assign: admin, developer, editor, launcher, viewer"
            ),
        ] = None,
    ):
        """Invite a user to a tenant and and assign either a group or an IAM role. If an invitation already exists for this email, the tool returns the existing invitation details."""
        api_version = await _detect_api_version()
        
        # First check if an invitation already exists for this email
        try:
            existing_invites = await client.get(f"/invitations/email/{email}")
            existing_invites.raise_for_status()
            invites = existing_invites.json()
            if invites and len(invites) > 0:
                # Return the most recent invitation
                return invites[0]
        except httpx.HTTPStatusError as e:
            if (
                e.response.status_code != 404
            ):  # 404 means no invites found, which is fine
                raise

        # If no existing invitation, proceed with creating a new one
        groupIds = []
        if group_names:
            for group_name in group_names:
                resp = await client.get(f"/groups/search", params={"q": group_name, "page": 1, "size": 10})
                resp.raise_for_status()
                results = resp.json()
                items = results.get("results", [])
                matched = [g for g in items if g.get("name") == group_name]
                if not matched:
                    raise ValueError(f"Group '{group_name}' not found.")
                groupIds.append(matched[0]["id"])

        # Prepare payload based on API version
        tenant = os.getenv("KESTRA_TENANT_ID")
        
        if api_version == "0.24+":
            # In 0.24+, the API structure has changed
            # For now, we'll create the invitation without bindings
            # as the new API structure needs to be implemented
            payload = {
                "email": email,
                "userType": "STANDARD",  # Only STANDARD is supported
                "groupIds": groupIds,
            }
        else:
            # Prepare bindings if role is specified (for older API)
            bindings = []
            if role:
                bindings.append(
                    {"type": "USER", "roleId": f"{role}_{tenant}", "deleted": False}
                )

            payload = {
                "email": email,
                "userType": "STANDARD",  # Only STANDARD is supported
                "groupIds": groupIds,
                "bindings": bindings,
            }

        resp = await client.post(f"/invitations", json=payload)
        if resp.status_code == 409:
            # 409 Conflict means an invitation already exists or user already has access
            try:
                existing = await client.get(f"/invitations/email/{email}")
                existing.raise_for_status()
                invites = existing.json()
                invite = _extract_invitation(invites)
                if invite:
                    return invite
            except Exception:
                pass
            try:
                return resp.json()
            except Exception:
                resp.raise_for_status()

        # For 500 errors (e.g. SMTP failure), the invitation may still have been created
        if resp.status_code >= 400:
            created_invite = await _fetch_invitation_by_email(email)
            if created_invite:
                return created_invite
            resp.raise_for_status()

        # 201/204 responses may have no body - fetch the created invitation
        try:
            result = resp.json()
            if result:
                return result
        except Exception:
            pass
        created_invite = await _fetch_invitation_by_email(email)
        if created_invite:
            return created_invite
        return {"email": email, "status": "CREATED"}

    @mcp.tool()
    async def manage_tests(
        action: Annotated[
            Literal["create", "run", "delete"],
            Field(description="The action to perform: create, run, delete"),
        ],
        yaml_source: Annotated[
            Optional[str], Field(description="The YAML source for the test")
        ] = None,
        namespace: Annotated[
            Optional[str], Field(description="The namespace of the test")
        ] = None,
        id_: Annotated[Optional[str], Field(description="The id of the test")] = None,
    ):
        """Manage a test (unit test) by action. For 'create', yaml_source is required (YAML string for the test definition). If the test already exists (POST 409 or 422), update it. For 'run' or 'delete', namespace and id_ are required."""
        headers = {"Content-Type": "application/x-yaml"}
        if action == "create":
            if not yaml_source:
                raise ValueError("'yaml_source' is required for create action.")
            try:
                resp = await client.post("/tests", content=yaml_source, headers=headers)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (409, 422):
                    doc = yaml.safe_load(yaml_source)
                    ns = doc.get("namespace")
                    tid = doc.get("id")
                    if not ns or not tid:
                        raise ValueError(
                            "YAML must include 'namespace' and 'id' fields for update."
                        )
                    put_url = f"/tests/{ns}/{tid}"
                    put_resp = await client.put(
                        put_url, content=yaml_source, headers=headers
                    )
                    put_resp.raise_for_status()
                    return put_resp.json()
                raise
        elif action == "delete":
            if not (namespace and id_):
                raise ValueError(
                    "'namespace' and 'id_' are required for delete action."
                )
            url = f"/tests/{namespace}/{id_}"
            resp = await client.delete(url)
            if resp.status_code in (200, 204):
                try:
                    return resp.json()
                except Exception:
                    return {}
            resp.raise_for_status()
            return resp.json()
        elif action == "run":
            if not (namespace and id_):
                raise ValueError("'namespace' and 'id_' are required for run action.")
            url = f"/tests/{namespace}/{id_}/run"
            resp = await client.post(url)
            resp.raise_for_status()
            result = resp.json()
            # Add UI URL for the test suite page
            base = str(client.base_url).rstrip("/")
            ui_base = re.sub(r"/api/v1(/[^/]+)?$", "", base)
            tenant = os.getenv("KESTRA_TENANT_ID")
            tenant_segment = f"/{tenant}" if tenant else ""
            result["url"] = f"{ui_base}/ui{tenant_segment}/tests/{namespace}/{result.get('testSuiteId', id_)}"
            return result
        else:
            raise ValueError("Action must be one of: create, run, delete")

    @mcp.tool()
    async def manage_apps(
        action: Annotated[
            Literal["create", "enable", "disable", "delete"],
            Field(description="The action to perform: create, enable, disable, delete"),
        ],
        uid: Annotated[
            Optional[str],
            Field(
                description="The UID of the app. Required for 'enable', 'disable', or 'delete' action."
            ),
        ] = None,
        yaml_source: Annotated[
            Optional[str],
            Field(
                description="The YAML string for the app definition. Required for 'create' action. If the app already exists, it will be updated."
            ),
        ] = None,
    ):
        """Manage an app by an action (create, enable, disable, delete). For 'create', yaml_source is required (YAML string for the app definition). For 'enable', 'disable', or 'delete', uid is required."""
        headers = {"Content-Type": "application/x-yaml"}
        if action == "create":
            if not yaml_source:
                raise ValueError("'yaml_source' is required for create action.")
            try:
                resp = await client.post("/apps", content=yaml_source, headers=headers)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (409, 422):
                    doc = yaml.safe_load(yaml_source)
                    app_id = doc.get("id")
                    app_namespace = doc.get("namespace")
                    app_flow_id = doc.get("flowId")

                    if not app_id or not app_namespace:
                        raise ValueError(
                            "YAML must include 'id' and 'namespace' fields for update."
                        )

                    # Search for existing app by id and namespace
                    search_resp = await client.get(
                        "/apps/search",
                        params={
                            "q": app_id,
                            "namespace": app_namespace,
                            "flowId": app_flow_id,
                        },
                    )
                    search_resp.raise_for_status()
                    search_results = search_resp.json()

                    if not search_results["results"]:
                        raise ValueError(
                            f"Could not find existing app with id '{app_id}' in namespace '{app_namespace}'"
                        )

                    uid_val = search_results["results"][0]["uid"]
                    if not uid_val:
                        raise ValueError("YAML must include 'uid' field for update.")
                    put_url = f"/apps/{uid_val}"
                    put_resp = await client.put(
                        put_url, content=yaml_source, headers=headers
                    )
                    # Handle 304 Not Modified response
                    if put_resp.status_code == 304:
                        # Return the existing app details from search results
                        return search_results["results"][0]
                    put_resp.raise_for_status()
                    return put_resp.json()
                raise
        elif action == "enable":
            if not uid:
                raise ValueError("'uid' is required for enable action.")
            resp = await client.post(f"/apps/{uid}/enable")
        elif action == "disable":
            if not uid:
                raise ValueError("'uid' is required for disable action.")
            resp = await client.post(f"/apps/{uid}/disable")
        elif action == "delete":
            if not uid:
                raise ValueError("'uid' is required for delete action.")
            resp = await client.delete(f"/apps/{uid}")
        else:
            raise ValueError("Action must be one of: create, enable, disable, delete")
        if resp.status_code in (200, 204):
            try:
                return resp.json()
            except Exception:
                return {}
        resp.raise_for_status()
        return resp.json()

    @mcp.tool()
    async def search_apps(
        page: Annotated[
            int, Field(description="The page number to return. Default is 1.")
        ] = 1,
        size: Annotated[
            int,
            Field(description="The number of items to return per page. Default is 10."),
        ] = 10,
        sort: Annotated[Optional[list], Field(description="The list of sort fields.")] = None,
        tags: Annotated[
            Optional[list], Field(description="The list of tags to filter by.")
        ] = None,
        q: Annotated[
            Optional[str], Field(description="A string in a full-text search query.")
        ] = None,
        namespace: Annotated[
            Optional[str], Field(description="The namespace to filter by.")
        ] = None,
        flowId: Annotated[Optional[str], Field(description="The flowId to filter by.")] = None,
    ):
        """List existing apps, optionally filtered by namespace, flowId, tags, or full-text search string."""
        params = {"page": page, "size": size}
        if sort:
            params["sort"] = sort
        if tags:
            params["tags"] = tags
        if q:
            params["q"] = q
        if namespace:
            params["namespace"] = namespace
        if flowId:
            params["flowId"] = flowId
        resp = await client.get("/apps/search", params=params)
        resp.raise_for_status()
        return resp.json()

    @mcp.tool()
    async def manage_announcements(
        action: Annotated[
            Literal["list", "create", "update", "delete"],
            Field(description="The action to perform: list, create, update, delete"),
        ],
        id_: Annotated[
            Optional[str],
            Field(description="The banner ID. Required for update and delete actions."),
        ] = None,
        message: Annotated[
            Optional[str],
            Field(
                description="The banner message. Required for create and update actions."
            ),
        ] = None,
        type: Annotated[
            Optional[Literal["INFO", "WARNING", "ERROR"]],
            Field(
                description="The banner type. Required for create and update actions."
            ),
        ] = None,
        startDate: Annotated[
            Optional[str],
            Field(
                description="The start date in ISO format. Required for create and update actions."
            ),
        ] = None,
        endDate: Annotated[
            Optional[str],
            Field(
                description="The end date in ISO format. Required for create and update actions."
            ),
        ] = None,
        active: Annotated[
            bool, Field(description="Whether the banner is active. Default is True.")
        ] = True,
        tenantId: Annotated[Optional[str], Field(description="The tenant ID. Optional.")] = None,
    ):
        """Manage announcements (banners): list, create, update, or delete."""
        if action == "list":
            url = _root_api_url("/banners/search", client)
            resp = await client.get(url)
        elif action == "create":
            if not message:
                raise ValueError("'message' is required for create action.")
            data = {
                "message": message,
            }
            if type is not None:
                data["type"] = type
            if startDate is not None:
                data["startDate"] = startDate
            if endDate is not None:
                data["endDate"] = endDate
            if active is not None:
                data["active"] = active
            if tenantId:
                data["tenantId"] = tenantId
            url = _root_api_url("/banners", client)
            resp = await client.post(url, json=data)
        elif action == "update":
            if not (id_ and message and type and startDate and endDate):
                raise ValueError(
                    "'id_', 'message', 'type', 'startDate', and 'endDate' are required for update action."
                )
            data = {
                "id": id_,
                "message": message,
                "type": type,
                "startDate": startDate,
                "endDate": endDate,
                "active": active,
            }
            if tenantId:
                data["tenantId"] = tenantId
            url = _root_api_url(f"/banners/{id_}", client)
            resp = await client.put(url, json=data)
        elif action == "delete":
            if not id_:
                raise ValueError("'id_' is required for delete action.")
            url = _root_api_url(f"/banners/{id_}", client)
            resp = await client.delete(url)
        else:
            raise ValueError("Action must be one of: list, create, update, delete")

        if resp.status_code in (200, 204):
            try:
                return resp.json()
            except Exception:
                return {}
        resp.raise_for_status()
        return resp.json()

    @mcp.tool()
    async def manage_group(
        action: Annotated[
            Literal["create", "get", "update", "delete"],
            Field(description="The action to perform: create, get, update, delete"),
        ],
        id_: Annotated[
            Optional[str], Field(description="The group ID. Required for get and update actions.")
        ] = None,
        name: Annotated[
            Optional[str],
            Field(
                description="The group name. Required for create and update actions."
            ),
        ] = None,
        description: Annotated[
            Optional[str],
            Field(
                description="The group description. Required for create and update actions."
            ),
        ] = None,
        role: Annotated[
            Optional[Literal["admin", "developer", "editor", "launcher", "viewer"]],
            Field(description="The role to assign to the group. Optional."),
        ] = None,
    ):
        """
        Manage a group by action. The action must be one of 'create', 'get', 'update', or 'delete'.
        - 'create': name (required), description (optional), role (optional)
        - 'get': id_ (required)
        - 'update': id_, name, and description (all required)
        - 'delete': id_ (required)
        The tenant is managed by the Kestra client and base URL.
        """
        api_version = await _detect_api_version()
        base_url = "/groups"

        if action == "create":
            if not name:
                raise ValueError("'name' is required for create action.")
            data = {
                "name": name,
                "description": description or "",
                "deleted": False,
                "provider": {},
                "attributes": {},
            }
            try:
                response = await client.post(f"{base_url}", json=data)
                response.raise_for_status()
                group_id = response.json().get("id")
                group_data = response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 422:  # Group already exists
                    # Try to find the existing group
                    search_resp = await client.get(
                        f"{base_url}/search", params={"q": name, "page": 1, "size": 10}
                    )
                    search_resp.raise_for_status()
                    results = search_resp.json()
                    items = results.get("results", [])
                    matched = [g for g in items if g.get("name") == name]
                    if not matched:
                        raise ValueError(f"Group '{name}' not found after 422 error")
                    group_data = matched[0]  # Use the matched group data directly
                    group_id = group_data["id"]
                    # Update the group's description if provided
                    if description:
                        update_data = {
                            "id": group_id,
                            "name": name,
                            "description": description,
                            "deleted": False,
                            "provider": group_data.get("provider", {}),
                            "attributes": group_data.get("attributes", {}),
                            "externalId": group_data.get("externalId", ""),
                            "securityIntegrationId": group_data.get(
                                "securityIntegrationId", ""
                            ),
                            "securityIntegrationName": group_data.get(
                                "securityIntegrationName", ""
                            ),
                        }
                        update_resp = await client.put(
                            f"{base_url}/{group_id}", json=update_data
                        )
                        update_resp.raise_for_status()
                        group_data = update_resp.json()
                else:
                    raise
            if role:
                tenant = os.getenv("KESTRA_TENANT_ID")
                if not tenant:
                    raise ValueError("KESTRA_TENANT_ID environment variable is not set")
                
                # In 0.24+, bindings are handled differently - we need to use the new API
                if api_version == "0.24+":
                    # For 0.24+, we need to use the new tenant-access API or roles API
                    # For now, we'll skip the binding creation as the API structure has changed
                    # and we need more information about the new API structure
                    pass
                else:
                    # First check if binding already exists
                    try:
                        binding_resp = await client.get(
                            "/bindings/search",
                            params={"type": "GROUP", "id": group_id, "page": 1, "size": 10},
                        )
                        binding_resp.raise_for_status()
                        bindings = binding_resp.json()
                        existing_binding = None
                        for binding in bindings.get("results", []):
                            if binding.get("roleId") == f"{role}_{tenant}":
                                existing_binding = binding
                                break
                        if not existing_binding:
                            binding = {
                                "type": "GROUP",
                                "externalId": group_id,
                                "roleId": f"{role}_{tenant}",
                                "deleted": False,
                            }
                            try:
                                binding_response = await client.post(
                                    f"/bindings", json=binding
                                )
                                binding_response.raise_for_status()
                            except httpx.HTTPStatusError as e:
                                if e.response.status_code not in (409, 422):
                                    raise
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code != 404:
                            raise
            return group_data
        elif action == "get":
            if not id_:
                raise ValueError("'id_' is required for get action.")
            resp = await client.get(f"{base_url}/{id_}")
            resp.raise_for_status()
            return resp.json()
        elif action == "update":
            if not (id_ and name and description):
                raise ValueError(
                    "'id_', 'name', and 'description' are required for update action."
                )
            data = {
                "id": id_,
                "name": name,
                "description": description,
            }
            resp = await client.put(f"{base_url}/{id_}", json=data)
            resp.raise_for_status()
            return resp.json()
        elif action == "delete":
            if not id_:
                raise ValueError("'id_' is required for delete action.")
            resp = await client.delete(f"{base_url}/{id_}")
            if resp.status_code in (200, 204):
                try:
                    return resp.json()
                except Exception:
                    return {}
            resp.raise_for_status()
            return resp.json()
        else:
            raise ValueError("Action must be one of: create, get, update, delete")

    @mcp.tool()
    async def manage_invitations(
        action: Annotated[
            Literal["get", "delete"],
            Field(description="The action to perform: get, delete"),
        ],
        id_: Annotated[
            Optional[str],
            Field(
                description="The invitation ID. Required for get and delete actions."
            ),
        ] = None,
    ):
        """Manage an invitation by action. The action must be one of 'get' or 'delete'."""
        base_url = "/invitations"
            
        if not id_:
            raise ValueError("'id_' is required for this action.")
        if action == "get":
            resp = await client.get(f"{base_url}/{id_}")
            resp.raise_for_status()
            return resp.json()
        elif action == "delete":
            resp = await client.delete(f"{base_url}/{id_}")
            if resp.status_code in (200, 204):
                try:
                    return resp.json()
                except Exception:
                    return {}
            resp.raise_for_status()
            return resp.json()
        else:
            raise ValueError("Action must be one of: get, delete")

    @mcp.tool()
    async def get_instance_info(
        info: Annotated[
            Literal["configuration", "license_info"],
            Field(description="The type of instance info to get"),
        ] = "configuration",
    ):
        """Get instance metadata: configuration (default) or license info. 
        - Use the configuration info to get your instance configuration. 
        - Use the license info to show your license type and expiration date."""
        api_version = await _detect_api_version()
        
        if info == "configuration":
            url = _root_api_url("/configs", client)
        elif info == "license_info":
            url = _root_api_url("/license-info", client)
        else:
            raise ValueError(
                "info must be one of: 'configuration', 'license_info'"
            )
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()

    @mcp.tool()
    async def generate_app(
        user_prompt: Annotated[
            str,
            Field(description="The user prompt describing what app to generate"),
        ],
        app_yaml: Annotated[
            str,
            Field(description="Existing app YAML to use as context for regeneration. Optional - if not provided, a new app will be generated from scratch."),
        ] = "",
        provider_id: Annotated[
            str,
            Field(description="Optional AI provider ID to use for generation. Use the list from GET /ai/providers if multiple providers are configured."),
        ] = "",
        auto_create: Annotated[
            bool,
            Field(description="If true, automatically create the app after generation. Defaults to false so you can review the YAML first."),
        ] = False,
    ) -> dict:
        """Generate or regenerate an app based on a prompt and existing app context.

        This tool uses Kestra's AI app generation endpoint to create or modify apps
        based on natural language descriptions and existing app definitions.

        This tool is available in Kestra 0.24 and later (Enterprise Edition).

        Returns the generated app YAML definition. If auto_create is true, the app is also created in Kestra."""
        payload = {
            "conversationId": str(uuid.uuid4()),
            "userPrompt": user_prompt,
        }
        if app_yaml:
            payload["yaml"] = app_yaml
        if provider_id:
            payload["providerId"] = provider_id

        resp = await client.post("/ai/generate/app", json=payload)
        resp.raise_for_status()
        generated_yaml = resp.text

        if auto_create:
            return await manage_apps(action="create", yaml_source=generated_yaml)

        return {"result": generated_yaml}

    @mcp.tool()
    async def generate_test(
        user_prompt: Annotated[
            str,
            Field(description="The user prompt describing what test suite to generate"),
        ],
        test_yaml: Annotated[
            str,
            Field(description="Existing test suite YAML to use as context for regeneration. Optional - if not provided, a new test suite will be generated from scratch."),
        ] = "",
        provider_id: Annotated[
            str,
            Field(description="Optional AI provider ID to use for generation. Use the list from GET /ai/providers if multiple providers are configured."),
        ] = "",
        auto_create: Annotated[
            bool,
            Field(description="If true, automatically create the test suite after generation. Defaults to false so you can review the YAML first."),
        ] = False,
    ) -> dict:
        """Generate or regenerate a unit test suite based on a prompt and existing test context.

        This tool uses Kestra's AI test suite generation endpoint to create or modify
        test suites based on natural language descriptions and existing test definitions.

        This tool is available in Kestra 0.24 and later (Enterprise Edition).

        Returns the generated test suite YAML definition. If auto_create is true, the test suite is also created in Kestra."""
        payload = {
            "conversationId": str(uuid.uuid4()),
            "userPrompt": user_prompt,
        }
        if test_yaml:
            payload["yaml"] = test_yaml
        if provider_id:
            payload["providerId"] = provider_id

        resp = await client.post("/ai/generate/test", json=payload)
        resp.raise_for_status()
        generated_yaml = resp.text

        if auto_create:
            return await manage_tests(action="create", yaml_source=generated_yaml)

        return {"result": generated_yaml}
