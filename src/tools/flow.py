from mcp.server.fastmcp import FastMCP
import httpx
import yaml
import uuid
from typing import Annotated, List, Literal
from pydantic import Field
from kestra.utils import _render_dependencies, _score_flow_match
from kestra.constants import _RESERVED_FLOW_IDS


def register_flow_tools(mcp: FastMCP, client: httpx.AsyncClient) -> None:
    @mcp.tool()
    async def search_flows(
        query: Annotated[
            str,
            Field(
                description="The query string for full-text search in flow names and contents"
            ),
        ],
        size: Annotated[
            int,
            Field(description="The number of flows to return per page. Default is 10."),
        ] = 10,
        page: Annotated[
            int, Field(description="The page number to return. Default is 1.")
        ] = 1,
    ) -> dict:
        """Search flows by name or content.
        Use this tool when the user asks to:
        - Find flows containing [specific text]
        - Search for flows with [plugin] in their code
        - Show me flows that use [function/feature/plugin]

        Format the result as markdown. For each search result, output:
        - Flow {{id}} in the namespace {{namespace}}

        Do not return other metadata in the result."""
        resp = await client.get(
            "/flows/search", params={"query": query, "size": size, "page": page}
        )
        return resp.json()

    @mcp.tool()
    async def find_flow(
        query: Annotated[
            str,
            Field(
                description="Natural language flow name to search for, e.g. 'hello world' or 'data pipeline'"
            ),
        ],
        namespace: Annotated[
            str,
            Field(description="Optional namespace to narrow the search"),
        ] = "",
    ) -> dict:
        """Find a flow by approximate or partial name. Use this when you know
        a flow's name (or part of it) but not its exact namespace and ID.

        Returns the best matching flow(s) with their namespace and flow_id
        ready to use in other tools like execute_flow or manage_flow.

        If exactly one strong match is found, returns it directly.
        If multiple matches exist, returns a ranked list for the user to pick from."""
        params: dict = {"q": query, "size": 20, "page": 1}
        if namespace:
            params["namespace"] = namespace

        resp = await client.get("/flows/search", params=params)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])

        if not results:
            return {
                "found": False,
                "message": f"No flows found matching '{query}'. Try a different search term or check available flows with search_flows.",
                "matches": [],
            }

        scored = []
        for flow in results:
            fid = flow.get("id", "")
            fns = flow.get("namespace", "")
            score = _score_flow_match(query, fid, fns)
            scored.append({
                "namespace": fns,
                "flow_id": fid,
                "score": round(score, 3),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)

        best = scored[0]
        if best["score"] >= 0.8 and (len(scored) == 1 or scored[1]["score"] < best["score"] - 0.1):
            return {
                "found": True,
                "message": f"Found flow '{best['flow_id']}' in namespace '{best['namespace']}'",
                "match": {"namespace": best["namespace"], "flow_id": best["flow_id"]},
                "matches": scored[:5],
            }

        return {
            "found": True,
            "message": f"Found {len(scored)} possible matches. The top results are listed below.",
            "matches": scored[:5],
        }

    @mcp.tool()
    async def list_flows_with_triggers(
        namespace: Annotated[
            str,
            Field(
                description="The namespace to list flows from. If not provided, all namespaces are scanned."
            ),
        ] = "",
        enabled_only: Annotated[
            bool,
            Field(
                description="Whether to include only flows with enabled triggers. Default is False."
            ),
        ] = False,
        disabled_only: Annotated[
            bool,
            Field(
                description="Whether to include only flows with disabled triggers. Default is False."
            ),
        ] = False,
    ) -> List[str]:
        """List all flows that have one or more triggers.
        Returns a markdown‑style list of lines, e.g.:

        Flow `myflow` in namespace `dev` with triggers:
        - `daily` (enabled) of type `io.kestra.plugin.core.trigger.Schedule`
            defined as:
            ```yaml
            id: daily
            type: io.kestra.plugin.core.trigger.Schedule
            cron: "0 0 * * *"
            ```
        - `api` (disabled) of type `io.kestra.plugin.core.trigger.Webhook`
            defined as:
            ```yaml
            id: api
            type: io.kestra.plugin.core.trigger.Webhook
            key: myKey123
            ```"""
        namespaces = [namespace] if namespace else []
        if not namespaces:
            ns_resp = await client.get("/flows/distinct-namespaces")
            ns_resp.raise_for_status()
            namespaces = ns_resp.json()

        results: List[str] = []
        for ns in namespaces:
            flows_resp = await client.get(f"/flows/{ns}")
            flows_resp.raise_for_status()
            for flow in flows_resp.json():
                triggers = flow.get("triggers", [])
                if not triggers:
                    continue

                filtered = []
                for t in triggers:
                    is_disabled = t.get("disabled", False)
                    if enabled_only and is_disabled:
                        continue
                    if disabled_only and not is_disabled:
                        continue
                    filtered.append(t)

                if not filtered:
                    continue

                lines = [f"Flow `{flow['id']}` in namespace `{ns}` with triggers:"]
                for t in filtered:
                    status = "disabled" if t.get("disabled", False) else "enabled"
                    t_id = t.get("id")
                    t_type = t.get("type")

                    definition = yaml.safe_dump(t, sort_keys=False).strip()
                    lines.append(
                        f"- `{t_id}` ({status}) of type `{t_type}` defined as:\n"
                        "```yaml\n"
                        f"{definition}\n"
                        "```"
                    )

                results.append("\n".join(lines))

        return results

    @mcp.tool()
    async def create_flow_from_yaml(
        yaml_definition: Annotated[
            str,
            Field(
                description="The raw YAML definition of the flow to create or update"
            ),
        ],
    ) -> dict:
        """Create or update a Kestra flow from a raw YAML definition. Returns the created or updated flow's JSON metadata."""
        doc = yaml.safe_load(yaml_definition)
        ns = doc.get("namespace")
        fid = doc.get("id")
        if not ns or not fid:
            raise ValueError("YAML must include 'namespace' and 'id' fields")

        if fid in _RESERVED_FLOW_IDS:
            allowed = ", ".join(sorted(_RESERVED_FLOW_IDS))
            raise ValueError(
                f"The flow ID `{fid}` is reserved and cannot be used. "
                f"Please choose a different ID (not: {allowed})."
            )

        headers = {"Content-Type": "application/x-yaml"}
        try:
            resp = await client.post("/flows", content=yaml_definition, headers=headers)
            resp.raise_for_status()
            return resp.json()

        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if code in (409, 422):
                upd = await client.put(
                    f"/flows/{ns}/{fid}", content=yaml_definition, headers=headers
                )
                upd.raise_for_status()
                return upd.json()
            raise

    @mcp.tool()
    async def manage_flow(
        action: Annotated[
            Literal[
                "enable",
                "disable",
                "list_revisions",
                "list_dependencies",
                "get_yaml",
                "delete",
            ],
            Field(
                description="The action to perform: enable, disable, list_revisions, list_dependencies, get_yaml, or delete"
            ),
        ],
        namespace: Annotated[str, Field(description="The namespace of the flow")],
        flow_id: Annotated[str, Field(description="The ID of the flow")],
    ):
        """
        Manage a flow by action. The action must be one of 'enable', 'disable', 'list_revisions', 'list_dependencies', 'get_yaml', or 'delete'.
        - 'enable': Enable a single flow via the bulk‑enable endpoint.
        - 'disable': Disable a single flow via the bulk‑disable endpoint.
        - 'list_revisions': Retrieve all revision metadata for a given flow. For each revision, SUMMARIZE IN ONE SENTENCE what changed in that revision compared to the previous one, e.g. new task added, some task removed, flow disabled, etc.
        - 'list_dependencies': Retrieve all flow dependencies for a given flow and render them as an ASCII dependency graph **using only the flow IDs**. Always return the legend after the graph.
        - 'get_yaml': Get the YAML definition of a flow. Make sure to output YAML, not JSON.
        - 'delete': Delete a flow by its namespace and ID.
        Returns:
        - for 'enable' and 'disable': the API JSON response;
        - for 'list_revisions': a list of revision metadata for the flow;
        - for 'list_dependencies': an ASCII dependency graph as a string, with a legend;
        - for 'get_yaml': the YAML definition of the flow (pruned);
        - for 'delete': empty dict on success.
        """
        if action == "enable":
            payload = [{"namespace": namespace, "id": flow_id}]
            resp = await client.post("/flows/enable/by-ids", json=payload)
            resp.raise_for_status()
            return resp.json()
        elif action == "disable":
            payload = [{"namespace": namespace, "id": flow_id}]
            resp = await client.post("/flows/disable/by-ids", json=payload)
            resp.raise_for_status()
            return resp.json()
        elif action == "list_revisions":
            resp = await client.get(f"/flows/{namespace}/{flow_id}/revisions")
            resp.raise_for_status()
            return resp.json()
        elif action == "list_dependencies":
            resp = await client.get(f"/flows/{namespace}/{flow_id}/dependencies")
            resp.raise_for_status()
            return await _render_dependencies(
                resp.json(),
                "Flows listed without arrows have no dependencies within this flow.",
            )
        elif action == "get_yaml":
            resp = await client.get(
                f"/flows/{namespace}/{flow_id}", params={"source": "true"}
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("source")
        elif action == "delete":
            resp = await client.delete(f"/flows/{namespace}/{flow_id}")
            if resp.status_code in (200, 204):
                try:
                    return resp.json()
                except Exception:
                    return {}
            resp.raise_for_status()
            return resp.json()
        else:
            raise ValueError(
                "Action must be one of: enable, disable, list_revisions, list_dependencies, get_yaml, delete"
            )

    @mcp.tool()
    async def generate_flow(
        user_prompt: Annotated[
            str,
            Field(description="The user prompt describing what flow to generate"),
        ],
        flow_yaml: Annotated[
            str,
            Field(description="Existing flow YAML to use as context for regeneration. Optional - if not provided, a new flow will be generated from scratch."),
        ] = "",
        namespace: Annotated[
            str,
            Field(description="Optional namespace to use as context for the generated flow"),
        ] = "",
        provider_id: Annotated[
            str,
            Field(description="Optional AI provider ID to use for generation. Use the list from GET /ai/providers if multiple providers are configured."),
        ] = "",
        auto_create: Annotated[
            bool,
            Field(description="If true, automatically create or update the flow after generation. Defaults to false so you can review the YAML first."),
        ] = False,
    ) -> str:
        """Generate or regenerate a flow based on a prompt and existing flow context.

        This tool uses Kestra's AI flow generation endpoint to create or modify flows
        based on natural language descriptions and existing flow definitions.

        This tool is available in Kestra 0.24 and later.

        Returns the generated flow YAML definition. If auto_create is true, the flow is also created or updated in Kestra."""
        payload = {
            "conversationId": str(uuid.uuid4()),
            "userPrompt": user_prompt,
        }
        if flow_yaml:
            payload["yaml"] = flow_yaml
        if namespace:
            payload["namespace"] = namespace
        if provider_id:
            payload["providerId"] = provider_id

        resp = await client.post("/ai/generate/flow", json=payload)
        resp.raise_for_status()
        generated_yaml = resp.text

        if auto_create:
            return await create_flow_from_yaml(generated_yaml)

        return generated_yaml

    @mcp.tool()
    async def generate_dashboard(
        user_prompt: Annotated[
            str,
            Field(description="The user prompt describing what dashboard to generate"),
        ],
        dashboard_yaml: Annotated[
            str,
            Field(description="Existing dashboard YAML to use as context for regeneration. Optional - if not provided, a new dashboard will be generated from scratch."),
        ] = "",
        provider_id: Annotated[
            str,
            Field(description="Optional AI provider ID to use for generation. Use the list from GET /ai/providers if multiple providers are configured."),
        ] = "",
        auto_create: Annotated[
            bool,
            Field(description="If true, automatically create the dashboard after generation. Defaults to false so you can review the YAML first."),
        ] = False,
    ) -> str:
        """Generate or regenerate a dashboard based on a prompt and existing dashboard context.

        This tool uses Kestra's AI dashboard generation endpoint to create or modify dashboards
        based on natural language descriptions and existing dashboard definitions.

        This tool is available in Kestra 0.24 and later.

        Returns the generated dashboard YAML definition. If auto_create is true, the dashboard is also created in Kestra."""
        payload = {
            "conversationId": str(uuid.uuid4()),
            "userPrompt": user_prompt,
        }
        if dashboard_yaml:
            payload["yaml"] = dashboard_yaml
        if provider_id:
            payload["providerId"] = provider_id

        resp = await client.post("/ai/generate/dashboard", json=payload)
        resp.raise_for_status()
        generated_yaml = resp.text

        if auto_create:
            headers = {"Content-Type": "application/x-yaml"}
            create_resp = await client.post("/dashboards", content=generated_yaml, headers=headers)
            create_resp.raise_for_status()
            return create_resp.json()

        return generated_yaml
