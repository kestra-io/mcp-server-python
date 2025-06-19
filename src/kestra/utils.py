from typing import Dict, List, Set
from datetime import datetime
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


async def get_latest_execution(
    client: httpx.AsyncClient, namespace: str, flow_id: str, state: str = None
) -> dict:
    resp = await client.get(
        "/executions", params={"namespace": namespace, "flowId": flow_id}
    )
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict):
        executions = data.get("results") or data.get("content") or []
    elif isinstance(data, list):
        executions = data
    else:
        executions = []

    if state:
        executions = [
            e for e in executions if e.get("state", {}).get("current") == state
        ]

    if not executions:
        return {}

    latest = max(executions, key=lambda e: _parse_iso(e["state"]["startDate"]))
    return latest
