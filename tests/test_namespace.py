import pytest
from dotenv import load_dotenv
from pathlib import Path
import json

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env", override=True)


@pytest.mark.asyncio
async def test_namespace_actions(kestra_client):
    """Test namespace tools: list_namespaces, list_flows_in_namespace, list_namespace_dependencies."""
    # Test list_namespaces (all)
    namespaces = await kestra_client.call_tool("list_namespaces", {})
    namespace_texts = [ns.text for ns in namespaces.content]
    print(f"All namespaces: {namespace_texts}")
    assert isinstance(namespaces.content, list)
    assert all(isinstance(ns, str) for ns in namespace_texts)

    # Test list_namespaces (with flows)
    namespaces_with_flows = await kestra_client.call_tool(
        "list_namespaces", {"with_flows_only": True}
    )
    namespaces_with_flows_texts = [ns.text for ns in namespaces_with_flows.content]
    print(f"Namespaces with flows: {namespaces_with_flows_texts}")
    assert isinstance(namespaces_with_flows.content, list)
    assert all(isinstance(ns, str) for ns in namespaces_with_flows_texts)

    test_namespace = "company.team"
    # Test list_flows_in_namespace
    flows = await kestra_client.call_tool(
        "list_flows_in_namespace", {"namespace": test_namespace}
    )
    if flows.content:
        flows_response = json.loads(flows.content[0].text)
        flow_list = flows_response.get("results", flows_response) if isinstance(flows_response, dict) else flows_response
        print(f"Flows in {test_namespace}: {flow_list}")
        assert isinstance(flow_list, list)
        # Verify each flow has required fields
        for flow in flow_list:
            assert "id" in flow
            assert "namespace" in flow

    # Test list_namespace_dependencies
    dependencies = await kestra_client.call_tool(
        "list_namespace_dependencies", {"namespace": test_namespace}
    )
    dep_text = dependencies.content[0].text
    print(f"Dependencies for {test_namespace}: {dep_text}")
    assert isinstance(dep_text, str)
    assert "legend" in dep_text.lower() or "flows listed" in dep_text.lower()
