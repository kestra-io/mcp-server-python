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
        namespace_texts = [ns.text for ns in namespaces.content]
        print(f"All namespaces: {namespace_texts}")
        assert isinstance(namespaces.content, list)
        assert all(isinstance(ns, str) for ns in namespace_texts)

        # Test list_namespaces (with flows)
        namespaces_with_flows = await client.call_tool(
            "list_namespaces", {"with_flows": True}
        )
        namespaces_with_flows_texts = [ns.text for ns in namespaces_with_flows.content]
        print(f"Namespaces with flows: {namespaces_with_flows_texts}")
        assert isinstance(namespaces_with_flows.content, list)
        assert all(isinstance(ns, str) for ns in namespaces_with_flows_texts)

        test_namespace = "company.team"
        # Test list_flows_in_namespace
        flows = await client.call_tool(
            "list_flows_in_namespace", {"namespace": test_namespace}
        )
        if flows.content:
            flow_dicts = [json.loads(flow.text) for flow in flows.content]
            print(f"Flows in {test_namespace}: {flow_dicts}")
            assert isinstance(flow_dicts, list)
            # Verify each flow has required fields
            for flow in flow_dicts:
                assert "id" in flow
                assert "namespace" in flow

        # Test list_namespace_dependencies
        dependencies = await client.call_tool(
            "list_namespace_dependencies", {"namespace": test_namespace}
        )
        dep_text = dependencies.content[0].text
        print(f"Dependencies for {test_namespace}: {dep_text}")
        assert isinstance(dep_text, str)
        assert "legend" in dep_text.lower() or "flows listed" in dep_text.lower()


if __name__ == "__main__":
    asyncio.run(test_namespace_actions())
