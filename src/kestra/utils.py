from typing import Dict, List, Set
from datetime import datetime
from difflib import SequenceMatcher
import httpx
import re


def _root_api_url(path: str, client: httpx.AsyncClient) -> str:
    # Remove tenant segment from base_url if present
    base = str(client.base_url)
    # Remove trailing slash
    base = base.rstrip("/")
    # Remove tenant segment if present (e.g. /api/v1/tenant -> /api/v1)
    base = re.sub(r"/api/v1/[^/]+$", "/api/v1", base)
    # Remove /api/v1 at the end if path already starts with /api/v1
    if path.startswith("/api/v1"):
        return re.sub(r"/api/v1$", "", base) + path
    return base + path


def _parse_iso(s: str) -> datetime:
    """Returns a timezone-aware datetime in UTC"""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


async def _render_dependencies(data: dict, legend_text: str) -> str:
    uid_to_id: Dict[str, str] = {
        n["uid"]: n.get("id", n["uid"]) for n in data.get("nodes", [])
    }
    all_ids: Set[str] = set(uid_to_id.values())
    triggers: Dict[str, List[str]] = {}
    tasks: Dict[str, List[str]] = {}
    incoming: Dict[str, Set[str]] = {uid_to_id[uid]: set() for uid in uid_to_id}

    for e in data.get("edges", []):
        src = uid_to_id.get(e["source"], e["source"])
        tgt = uid_to_id.get(e["target"], e["target"])
        all_ids.update({src, tgt})
        incoming.setdefault(tgt, set()).add(src)

        if e.get("relation") == "FLOW_TRIGGER":
            triggers.setdefault(src, []).append(tgt)
        else:
            tasks.setdefault(src, []).append(tgt)

    for node in all_ids:
        triggers.setdefault(node, [])
        tasks.setdefault(node, [])

    roots = [nid for nid in all_ids if not incoming.get(nid)]
    seen: Set[str] = set()

    def render(node: str, prefix: str = "") -> List[str]:
        lines: List[str] = []
        if node in seen:
            return []
        seen.add(node)
        edges = [(t, "────▶") for t in triggers.get(node, [])] + [
            (t, "====▶") for t in tasks.get(node, [])
        ]
        if not edges:
            lines.append(f"{prefix}{node}")
            return lines
        for child, arrow in edges:
            lines.append(f"{prefix}{node} {arrow} {child}")
            indent = len(prefix) + len(node) + 1 + len(arrow) + 1
            sub_prefix = " " * indent
            lines += render(child, prefix=sub_prefix)
        return lines

    output_lines: List[str] = []
    for root in sorted(roots):
        output_lines += render(root)

    legend = [
        "",
        "Legend:",
        "  ────▶ FLOW_TRIGGER  (flow-trigger-based dependency)",
        "  ====▶ FLOW_TASK     (subflow-task-based dependency)",
        legend_text,
    ]
    return "\n".join(output_lines + legend)


def _normalize_for_matching(text: str) -> str:
    """Lowercase, collapse separators (spaces/hyphens/dots) to underscore, strip non-alphanumeric."""
    text = text.lower().strip()
    text = re.sub(r"[\s\-\.]+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    return text


def _score_flow_match(query: str, flow_id: str, namespace: str = "") -> float:
    """Score 0.0–1.0 how well a user query matches a flow ID.

    Exact normalized match → 1.0, substring containment → 0.9,
    otherwise SequenceMatcher ratio with a small namespace bonus.
    """
    nq = _normalize_for_matching(query)
    nid = _normalize_for_matching(flow_id)

    if nq == nid:
        return 1.0

    if nq in nid or nid in nq:
        return 0.9

    score = SequenceMatcher(None, nq, nid).ratio()

    if namespace:
        nns = _normalize_for_matching(namespace)
        if nq in nns:
            score = min(score + 0.05, 1.0)

    return score


async def get_latest_execution(
    client: httpx.AsyncClient, namespace: str, flow_id: str, state: str = None
) -> dict:
    # Imported here rather than at module level because compat imports this module.
    from kestra.compat import executions_search_params

    params = await executions_search_params(
        client, namespace=namespace, flow_id=flow_id, state=state or None, size=25
    )
    resp = await client.get("/executions/search", params=params)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict):
        executions = data.get("results") or data.get("content") or []
    elif isinstance(data, list):
        executions = data
    else:
        executions = []

    # Double-check state filter in case API doesn't support it
    if state:
        executions = [
            e for e in executions if e.get("state", {}).get("current") == state
        ]

    if not executions:
        return {}

    latest = max(executions, key=lambda e: _parse_iso(e["state"]["startDate"]))
    return latest


async def _create_flow_from_yaml(client: httpx.AsyncClient, yaml_definition: str) -> dict:
    import yaml
    from kestra.constants import _RESERVED_FLOW_IDS

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


async def _create_app(client: httpx.AsyncClient, yaml_source: str) -> dict:
    import yaml

    headers = {"Content-Type": "application/x-yaml"}
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


async def _create_test(client: httpx.AsyncClient, yaml_source: str) -> dict:
    import yaml

    headers = {"Content-Type": "application/x-yaml"}
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
