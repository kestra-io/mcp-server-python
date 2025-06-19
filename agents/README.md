## Google ADK

To launch the [Agent Development UI](https://google.github.io/adk-docs/get-started/quickstart/), run the following commands in your terminal from the `agents` directory:

```bash
source .venv/bin/activate  
adk web
```

Then, select `google-mcp-client` from the agent dropdown and start sending your prompts to interact with Kestra's MCP Server.

Best to enable the toggle "Token Streaming" to stream responses as they are generated.

For more information, check the official [adk-python](https://github.com/google/adk-python/) repository. 