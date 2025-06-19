"""
This is a simple example of how to use the Kestra MCP Server with the OpenAI API.

Before running it, make sure to install the openai-agents package:
uv pip install openai-agents

Then, run the script as follows from the root of the project:
uv run agents/openai-mcp-client/agent.py -p "List all namespaces"
"""

import os
import argparse
import asyncio
import shutil
import os
from agents import Agent, Runner, gen_trace_id, trace
from agents.mcp import MCPServer, MCPServerStdio
from agents.model_settings import ModelSettings
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


async def run(mcp_server: MCPServer, prompt: str, file_output: bool):
    agent = Agent(
        name="Kestra Assistant",
        instructions="""
        You are an expert user of Kestra. For every prompt:
        1. Review the list of available tools.
        2. Select the single most relevant tool that best answers the question.
        3. Invoke only that one tool; do not call any others.
        4. Return the tool's result as your response directly with no additional tool calls.
        """,
        mcp_servers=[mcp_server],
        model="gpt-4o",
        model_settings=ModelSettings(tool_choice="required"),
    )
    print(f"\n\nRunning: {prompt}\n")
    result = await Runner.run(starting_agent=agent, input=prompt)
    output = result.final_output
    print(output)

    if file_output:
        with open("output.md", "w", encoding="utf-8") as f:
            f.write(output)
        print("\n[Saved output to output.md]")


async def main(prompt: str, file_output: bool):
    async with MCPServerStdio(
        name="Kestra MCP Server",
        params={
            "command": "uv",
            "args": ["run", "./src/server.py"],
            "env": dict(os.environ),
        },
    ) as server:
        trace_id = gen_trace_id()
        with trace(workflow_name="SSE Kestra", trace_id=trace_id):
            print(
                f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}\n"
            )
            await run(server, prompt, file_output)


if __name__ == "__main__":
    if not shutil.which("uv"):
        raise RuntimeError(
            "uv is not installed. Please install it: https://docs.astral.sh/uv/getting-started/installation/"
        )

    parser = argparse.ArgumentParser(
        description="Run the Kestra Assistant with a given prompt."
    )
    parser.add_argument(
        "-p",
        "--prompt",
        required=True,
        help="The prompt to send to the Kestra Assistant",
    )
    parser.add_argument(
        "-f",
        "--file-output",
        action="store_true",
        help="Also write the assistant output to output.md",
    )
    args = parser.parse_args()

    asyncio.run(main(args.prompt, args.file_output))
