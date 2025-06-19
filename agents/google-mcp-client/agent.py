from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.genai import types
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY environment variable")

root_agent = LlmAgent(
    model=os.getenv("GOOGLE_GEMINI_MODEL_AGENT"),
    name="kestra_agent",
    instruction="""You are an expert assistant for Kestra MCP. For every user query:
    1. Review the list of available MCP tools.
    2. Select the single most relevant tool that nest answers the question.
    3. Invoke only that one tool; do not call any others.
    4. Return the tool's result as your response directly with no additional tool calls.
    Return in markdown format.
    """,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.5,
        max_output_tokens=4096,
    ),
    tools=[
        MCPToolset(
            connection_params=StdioServerParameters(
                command="uv",
                args=["run", "../src/server.py"],
            )
            # Optional: Filter which tools from the MCP server are exposed
            # tool_filter=['backfill_executions', 'execute_flow']
        )
    ],
)
