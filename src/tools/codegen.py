from mcp.server.fastmcp import FastMCP
import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types
import os
from pathlib import Path
from typing import Annotated
from pydantic import Field
from kestra.codegen_utils import (
    read_plugin_documentation,
    fetch_plugins_to_csv,
    read_plugins,
    get_relevant_plugins,
)


load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_GEMINI_MODEL_CODEGEN = os.getenv("GOOGLE_GEMINI_MODEL_CODEGEN")
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY environment variable")
if not GOOGLE_GEMINI_MODEL_CODEGEN:
    raise ValueError(
        "Please set GOOGLE_GEMINI_MODEL_CODEGEN environment variable. Check the https://ai.google.dev/gemini-api/docs/models for available models."
    )

HELICONE_API_KEY = os.environ.get("HELICONE_API_KEY")

if HELICONE_API_KEY:
    gemini_client = genai.Client(
        api_key=GOOGLE_API_KEY,
        http_options={
            "base_url": "https://gateway.helicone.ai",
            "headers": {
                "helicone-auth": f"Bearer {HELICONE_API_KEY}",
                "helicone-target-url": "https://generativelanguage.googleapis.com",
            },
        },
    )
else:
    gemini_client = genai.Client(api_key=GOOGLE_API_KEY)


def register_codegen_tools(mcp: FastMCP, client: httpx.AsyncClient) -> None:
    @mcp.tool()
    async def generate_yaml(
        query: Annotated[
            str, Field(description="The user query to generate a flow from")
        ],
    ) -> str:
        """Generate, write, design, or code a flow based on the user query."""
        if not os.path.exists("plugins.csv"):
            await fetch_plugins_to_csv(client)
        plugins = read_plugins()
        relevant_plugins = get_relevant_plugins(gemini_client, query, plugins)
        plugin_docs = {}
        for plugin in relevant_plugins:
            plugin_docs[plugin] = await read_plugin_documentation(plugin, client)

        prompt = f"""Create a kestra flow YAML based on User Query using the provided Plugin Documentation. 
        Return only the YAML code. Do not include any explanations or comments in the YAML. 
        User Query: {query}
        Plugin Documentation:
        {chr(10).join(f"=== {plugin} ===\n{docs}\n" for plugin, docs in plugin_docs.items())}"""
        try:
            config = types.GenerateContentConfig(
                temperature=0.5, max_output_tokens=4096, response_mime_type="text/plain"
            )
            response = gemini_client.models.generate_content(
                model=GOOGLE_GEMINI_MODEL_CODEGEN,
                contents=prompt,
                config=config,
            )
            if not response.text:
                return "No detailed response could be generated."
            yaml_definition = response.text.strip()
            return yaml_definition
        except Exception as e:
            return f"Error generating detailed response: {str(e)}"
