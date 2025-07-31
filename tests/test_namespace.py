import pytest
from fastmcp import Client
from dotenv import load_dotenv
from pathlib import Path
from test_utils import mcp_server_config
import asyncio
import json

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env", override=True)


@pytest.mark.asyncio
async def test_namespace_actions():
    """Test namespace tools: list_namespaces, list_flows_in_namespace, list_namespace_dependencies."""
    async with Client(mcp_server_config) as client:
        # Test list_namespaces (all)
        namespaces = await client.call_tool("list_namespaces", {})
        namespace_texts = [ns.text for ns in namespaces]
        print(f"All namespaces: {namespace_texts}")
        assert isinstance(namespaces, list)
        assert all(isinstance(ns, str) for ns in namespace_texts)
        assert any("created" in ns or "not created" in ns for ns in namespace_texts)

        # Test list_namespaces with with_flows_only
        namespaces_with_flows = await client.call_tool(
            "list_namespaces", {"with_flows_only": True}
        )
        namespaces_with_flows_texts = [ns.text for ns in namespaces_with_flows]
        print(f"Namespaces with flows: {namespaces_with_flows_texts}")
        assert isinstance(namespaces_with_flows, list)
        assert all(isinstance(ns, str) for ns in namespaces_with_flows_texts)

        test_namespace = "company.team"
        # Test list_flows_in_namespace
        flows = await client.call_tool(
            "list_flows_in_namespace", {"namespace": test_namespace}
        )
        if flows:
            flow_dicts = [json.loads(flow.text) for flow in flows]
            print(f"Flows in namespace {test_namespace}: {flow_dicts}")
            assert all(isinstance(flow, dict) for flow in flow_dicts)
            assert all("id" in flow for flow in flow_dicts)
            assert all("namespace" in flow for flow in flow_dicts)

        # Test list_namespace_dependencies
        dependencies = await client.call_tool(
            "list_namespace_dependencies", {"namespace": test_namespace}
        )
        dep_text = dependencies[0].text
        print(f"Dependencies in namespace {test_namespace}: {dep_text}")
        assert isinstance(dep_text, str)
        assert "legend" in dep_text.lower() or "flows listed" in dep_text.lower()


if __name__ == "__main__":
    asyncio.run(test_namespace_actions())
