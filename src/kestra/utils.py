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
    params = {"namespace": namespace, "flowId": flow_id, "size": 25}
    if state:
        params["state"] = state
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
