import json
from pydantic import BaseModel, Field, RootModel
import re
from typing import Any, Dict, List, Optional
import httpx
import httpx
import csv
from google.genai import types
import json
import re
import sys
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from typing import Any, Dict, List, Set
from google import genai
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")


GOOGLE_GEMINI_MODEL_CODEGEN = os.environ.get("GOOGLE_GEMINI_MODEL_CODEGEN", "gemini-2.5-flash")


class PluginInfo(BaseModel):
    """Schema for plugin information."""

    pluginClass: str
    description: str


class PluginResponse(BaseModel):
    """Schema for the response containing relevant plugins."""

    relevant_plugins: List[str]


class PropertySchema(BaseModel):
    """Schema for a single property in the plugin."""

    type: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    default: Optional[Any] = None
    enum: Optional[List[str]] = None
    required: Optional[bool] = Field(None, alias="$required")
    dynamic: Optional[bool] = Field(None, alias="$dynamic")
    deprecated: Optional[bool] = Field(None, alias="$deprecated")
    properties: dict = None
    items: dict = None
    oneOf: Optional[List[Dict[str, Any]]] = None
    additionalProperties: dict = None
    ref: Optional[str] = Field(None, alias="$ref")
    format: Optional[str] = None
    minLength: Optional[int] = None
    pattern: Optional[str] = None
    minimum: Optional[int] = None


class Example(BaseModel):
    """Schema for a plugin example."""

    full: bool
    code: str
    lang: str
    title: str


class OutputProperty(BaseModel):
    """Schema for a single output property."""

    type: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    ref: Optional[str] = Field(None, alias="$ref")
    required: Optional[bool] = Field(None, alias="$required")
    format: Optional[str] = None
    enum: Optional[List[str]] = None
    properties: dict = None
    items: dict = None
    additionalProperties: dict = None
    default: Optional[Any] = None
    dynamic: Optional[bool] = Field(None, alias="$dynamic")
    oneOf: Optional[List[Dict[str, Any]]] = None


class OutputSchema(BaseModel):
    """Schema for plugin outputs."""

    properties: Optional[Dict[str, OutputProperty]] = Field(default_factory=dict)
    schema_version: Optional[str] = Field(None, alias="$schema")
    required: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, schema: Dict[str, Any]) -> "OutputSchema":
        """Create an OutputSchema instance from a dictionary."""
        properties = {}
        required = []

        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                properties[prop_name] = OutputProperty(
                    type=prop_schema.get("type", "object"),
                    title=prop_schema.get("title", prop_name),
                    description=prop_schema.get("description", ""),
                    required=prop_schema.get("$required", False),
                    ref=prop_schema.get("$ref"),
                    format=prop_schema.get("format"),
                    enum=prop_schema.get("enum"),
                    properties=prop_schema.get("properties"),
                    items=prop_schema.get("items"),
                    additionalProperties=prop_schema.get("additionalProperties"),
                    default=prop_schema.get("default"),
                    dynamic=prop_schema.get("$dynamic"),
                    oneOf=prop_schema.get("oneOf"),
                )

        if "required" in schema:
            required = schema["required"]

        return cls(
            properties=properties,
            schema_version=schema.get(
                "$schema", "https://json-schema.org/draft/2019-09/schema"
            ),
            required=required,
        )


class PluginSchema(BaseModel):
    """Schema for plugin JSON documentation."""

    properties: Dict[str, PropertySchema]
    required: List[str]
    title: str
    examples: Optional[List[Example]] = Field(None, alias="$examples")
    outputs: Optional[OutputSchema] = None
    definitions: Optional[Dict[str, Any]] = Field(None, alias="$defs")
    schema_version: Optional[str] = Field(None, alias="$schema")


def extract_json(text: str) -> Optional[str]:
    """Extract JSON array from text with multiple fallback strategies."""
    if not text:
        return None

    # Try direct JSON parse first
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Try to find JSON array with flexible pattern
    patterns = [
        r"(\[.*\])",  # Standard array
        r"(\[.*\])\s*$",  # Array at end of text
        r"^.*?(\[.*\])",  # Array after any prefix
        r"\[(.*?)\]",  # Most lenient pattern
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                # Validate the extracted JSON
                json.loads(match.group(1))
                return match.group(1)
            except json.JSONDecodeError:
                continue

    return None


class RelevantPlugins(RootModel):
    root: List[str]


def format_property_details(
    prop_details: PropertySchema,
    definitions: dict = None,
    indent: int = 0,
) -> str:
    """Format property details including references and oneOf options."""
    indent_str = "  " * indent
    details = []

    # Handle oneOf references
    if prop_details.oneOf:
        details.append(f"\n{indent_str}  Options:")
        for option in prop_details.oneOf:
            if "$ref" in option:
                ref_name = option["$ref"][8:]  # Remove #/$defs/ prefix
                if definitions and ref_name in definitions:
                    ref_def = definitions[ref_name]
                    if isinstance(ref_def, dict):
                        details.append(f"\n{indent_str}    - {ref_name}:")
                        if "type" in ref_def:
                            details.append(
                                f"\n{indent_str}      Type: {ref_def['type']}"
                            )
                        if "properties" in ref_def:
                            details.append(f"\n{indent_str}      Properties:")
                            for ref_prop_name, ref_prop in ref_def[
                                "properties"
                            ].items():
                                ref_type = ref_prop.get("type", "unknown")
                                ref_format = (
                                    f" ({ref_prop.get('format', '')})"
                                    if "format" in ref_prop
                                    else ""
                                )
                                ref_enum = (
                                    f" [{', '.join(ref_prop.get('enum', []))}]"
                                    if "enum" in ref_prop
                                    else ""
                                )
                                ref_required = (
                                    " (required)"
                                    if ref_prop.get("$required", False)
                                    else ""
                                )
                                ref_default = (
                                    f" (default: {ref_prop.get('default')})"
                                    if "default" in ref_prop
                                    else ""
                                )
                                details.append(
                                    f"\n{indent_str}        - {ref_prop_name}: {ref_type}{ref_format}{ref_enum}{ref_required}{ref_default}"
                                )
                        if "required" in ref_def:
                            details.append(
                                f"\n{indent_str}      Required: {', '.join(ref_def['required'])}"
                            )

    # Handle direct reference
    elif prop_details.ref:
        if prop_details.ref.startswith("#/$defs/"):
            ref_name = prop_details.ref[8:]  # Remove #/$defs/ prefix
            details.append(f"\n{indent_str}  References: {ref_name}")
            if definitions and ref_name in definitions:
                ref_def = definitions[ref_name]
                if isinstance(ref_def, dict):
                    if "type" in ref_def:
                        details.append(f"\n{indent_str}  Type: {ref_def['type']}")
                    if "properties" in ref_def:
                        details.append(f"\n{indent_str}  Properties:")
                        for ref_prop_name, ref_prop in ref_def["properties"].items():
                            ref_type = ref_prop.get("type", "unknown")
                            ref_format = (
                                f" ({ref_prop.get('format', '')})"
                                if "format" in ref_prop
                                else ""
                            )
                            ref_enum = (
                                f" [{', '.join(ref_prop.get('enum', []))}]"
                                if "enum" in ref_prop
                                else ""
                            )
                            ref_required = (
                                " (required)"
                                if ref_prop.get("$required", False)
                                else ""
                            )
                            ref_default = (
                                f" (default: {ref_prop.get('default')})"
                                if "default" in ref_prop
                                else ""
                            )
                            details.append(
                                f"\n{indent_str}    - {ref_prop_name}: {ref_type}{ref_format}{ref_enum}{ref_required}{ref_default}"
                            )
                    if "required" in ref_def:
                        details.append(
                            f"\n{indent_str}  Required: {', '.join(ref_def['required'])}"
                        )

    return "".join(details)


def format_output_property(
    output_name: str,
    output_details: OutputProperty,
    definitions: dict = None,
) -> str:
    """Format a single output property with its details."""
    indent_str = "  "
    details = []

    # Add title and type
    title = f": {output_details.title}" if output_details.title else ""
    required = " (required)" if output_details.required else ""
    details.append(f"- {output_name}{title}{required}")

    # Add type and format
    if output_details.type:
        format_str = f" ({output_details.format})" if output_details.format else ""
        details.append(f"{indent_str}  Type: {output_details.type}{format_str}")

    # Add enum values if present
    if output_details.enum:
        details.append(f"{indent_str}  Options: [{', '.join(output_details.enum)}]")

    # Handle references
    if output_details.ref:
        if output_details.ref.startswith("#/$defs/"):
            ref_name = output_details.ref[8:]  # Remove #/$defs/ prefix
            details.append(f"{indent_str}  References: {ref_name}")
            if definitions and ref_name in definitions:
                ref_def = definitions[ref_name]
                if isinstance(ref_def, dict):
                    if "type" in ref_def:
                        details.append(f"{indent_str}  Type: {ref_def['type']}")
                    if "properties" in ref_def:
                        details.append(f"{indent_str}  Properties:")
                        for ref_prop_name, ref_prop in ref_def["properties"].items():
                            ref_type = ref_prop.get("type", "unknown")
                            ref_format = (
                                f" ({ref_prop.get('format', '')})"
                                if "format" in ref_prop
                                else ""
                            )
                            ref_enum = (
                                f" [{', '.join(ref_prop.get('enum', []))}]"
                                if "enum" in ref_prop
                                else ""
                            )
                            ref_required = (
                                " (required)"
                                if ref_prop.get("$required", False)
                                else ""
                            )
                            ref_default = (
                                f" (default: {ref_prop.get('default')})"
                                if "default" in ref_prop
                                else ""
                            )
                            details.append(
                                f"{indent_str}    - {ref_prop_name}: {ref_type}{ref_format}{ref_enum}{ref_required}{ref_default}"
                            )
                    if "required" in ref_def:
                        details.append(
                            f"{indent_str}  Required: {', '.join(ref_def['required'])}"
                        )

    # Handle additional properties
    if output_details.additionalProperties:
        details.append(
            f"{indent_str}  Additional Properties: {output_details.additionalProperties}"
        )

    # Handle default value
    if output_details.default is not None:
        details.append(f"{indent_str}  Default: {output_details.default}")

    # Handle oneOf
    if output_details.oneOf:
        details.append(f"{indent_str}  Options:")
        for option in output_details.oneOf:
            if isinstance(option, dict):
                option_type = option.get("type", "unknown")
                option_format = (
                    f" ({option.get('format', '')})" if "format" in option else ""
                )
                details.append(f"{indent_str}    - {option_type}{option_format}")

    return "\n".join(details)


def format_property_block(
    name: str, details: PropertySchema, definitions: dict = None
) -> str:
    """Format a single property block with its details."""
    required = " (required)" if details.required else ""
    title = f": {details.title}" if details.title else ""
    description = f"\n  {details.description}" if details.description else ""
    default = f"\n  Default: {details.default}" if details.default is not None else ""
    enum = f"\n  Options: {', '.join(details.enum)}" if details.enum else ""
    format_str = f"\n  Format: {details.format}" if details.format else ""
    min_length = (
        f"\n  Min Length: {details.minLength}" if details.minLength is not None else ""
    )
    pattern = f"\n  Pattern: {details.pattern}" if details.pattern else ""
    minimum = f"\n  Minimum: {details.minimum}" if details.minimum is not None else ""
    details_str = format_property_details(details, definitions)

    return f"- {name}{required}{title}{description}{default}{enum}{format_str}{min_length}{pattern}{minimum}{details_str}\n"


async def read_plugin_documentation(
    plugin_class: str, client: httpx.AsyncClient
) -> str:
    """Fetch the JSON schema documentation for a specific plugin from the Kestra API."""
    try:
        resp = await client.get(f"/plugins/{plugin_class}", params={"all": "true"})
        resp.raise_for_status()
        data = resp.json()
        schema_json = data.get("schema")
        if not schema_json:
            return f"No JSON schema found for {plugin_class} via API."
        definitions = schema_json.get("definitions", {})
        props = schema_json.get("properties", {})
        metrics = props.get("$metrics", [])
        schema_dict = {
            "properties": props.get("properties", {}),
            "required": props.get("required", []),
            "title": props.get("title", ""),
            "$examples": props.get("$examples", []),
            "outputs": schema_json.get("outputs"),
            "$defs": definitions,
            "$schema": props.get("$schema"),
        }
        schema = PluginSchema(**schema_dict)
        formatted_docs = f"""Plugin: {schema.title}\n\nProperties:\n"""
        for name, details in schema.properties.items():
            formatted_docs += format_property_block(name, details, definitions)

        # Examples section
        if schema.examples:
            formatted_docs += "\nExamples:\n"
            for ex in schema.examples:
                formatted_docs += (
                    f"\n{ex.title or 'Example'}:\n```{ex.lang}\n{ex.code}\n```\n"
                )

        # Outputs section
        if schema.outputs and schema.outputs.properties:
            formatted_docs += "\nOutputs:\n"
            for out_name, out_detail in schema.outputs.properties.items():
                formatted_docs += (
                    format_output_property(out_name, out_detail, definitions) + "\n"
                )

        # Metrics section
        if metrics:
            formatted_docs += "\nMetrics:\n"
            for metric in metrics:
                name = metric.get("name", "")
                mtype = metric.get("type", "")
                unit = metric.get("unit", "")
                desc = metric.get("description", "")
                unit_str = f", unit={unit}" if unit else ""
                desc_str = f", description={desc}" if desc else ""
                formatted_docs += f"- {name}: type={mtype}{unit_str}{desc_str}\n"
        return formatted_docs
    except httpx.HTTPStatusError as e:
        return f"Error fetching schema for {plugin_class}: {e.response.text}"
    except Exception as e:
        return f"Error reading documentation for {plugin_class}: {str(e)}"


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True,
)
def get_relevant_plugins(
    gemini_client: genai.Client, query: str, plugins: List[PluginInfo]
) -> List[str]:
    """Use Gemini to find the most relevant plugins for the given query with retries."""
    prompt = f"""Find the most relevant Kestra plugin class names for a user's query.
    You MUST ALWAYS return at least one plugin.

    Respond ONLY with a valid JSON array of plugin class names, e.g.:
    [
    "io.kestra.plugin.example.PluginClass1",
    "io.kestra.plugin.example.PluginClass2"
    ]

    1. Consider both the pluginClass and its description
    2. Sort plugins by relevance to the query
    3. Return up to 10 most relevant plugins
    4. Be lenient in matching - consider matching by technology name e.g. MongoDB -> mongo, PostgreSQL -> postgres
    5. Consider synonyms for the query terms e.g. Query: "Load data into..." -> Match plugins with "load", "insert", "bulk", "query" in pluginClass or description
    7. IMPORTANT: Your response must be a valid JSON array, nothing else

    User Query: {query}

    Available Plugins:
    {chr(10).join(f"{p.pluginClass}: {p.description}" for p in plugins)}
    """

    try:
        config = types.GenerateContentConfig(
            temperature=0.9,
            max_output_tokens=1024,
            response_mime_type="application/json",
            response_schema=RelevantPlugins,
        )

        response = gemini_client.models.generate_content(
            model=GOOGLE_GEMINI_MODEL_CODEGEN,
            contents=prompt,
            config=config,
        )

        raw = response.text or ""

        # Try parsed response first
        if hasattr(response, "parsed") and response.parsed:
            return response.parsed.root

        # Fallback to JSON extraction
        json_text = extract_json(raw)
        if json_text:
            try:
                result = json.loads(json_text)
                if isinstance(result, list) and len(result) > 0:
                    return result
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Failed JSON text: {json_text}")

        print(f"No valid JSON array found in response. Raw response: {raw}")
        raise ValueError("Failed to get valid plugin list")
    except Exception as e:
        print(f"Error getting response from Gemini: {str(e)}")
        print(f"Full error details: {type(e).__name__}: {str(e)}")
        raise  # Re-raise for retry mechanism


async def fetch_plugins_to_csv(
    client: httpx.AsyncClient, output_file: str = "plugins.csv"
) -> str:
    """Fetch all Kestra plugins and their descriptions, writing them to a CSV file.
    Returns a success message with the number of plugins written.

    Args:
        output_file: Path to the output CSV file (default: plugins.csv)"""

    async def get_plugin_classes(plugins_response: List[Dict[str, Any]]) -> Set[str]:
        """Extract plugin classes from the API response."""
        plugin_classes = set()

        for plugin in plugins_response:
            # Extract from tasks
            if "tasks" in plugin and isinstance(plugin["tasks"], list):
                plugin_classes.update(plugin["tasks"])
            # Extract from triggers
            if "triggers" in plugin and isinstance(plugin["triggers"], list):
                plugin_classes.update(plugin["triggers"])
            # Extract from taskRunners
            if "taskRunners" in plugin and isinstance(plugin["taskRunners"], list):
                plugin_classes.update(plugin["taskRunners"])

        return plugin_classes

    async def get_plugin_description(plugin_class: str) -> str:
        """Get the description from plugin's markdown frontmatter."""
        try:
            plugin_url = f"/plugins/{plugin_class}"
            plugin_response = await client.get(plugin_url)

            if plugin_response.status_code != 200:
                return ""

            plugin_data = plugin_response.json()
            markdown_content = plugin_data.get("markdown", "")

            # Extract description from frontmatter
            description_match = re.search(r'description:\s*"([^"]*)"', markdown_content)
            if description_match:
                return description_match.group(1)
            return ""
        except Exception:
            return ""

    # First API call to get list of plugins
    response = await client.get("/plugins/groups/subgroups?includeDeprecated=false")
    if response.status_code != 200:
        raise ValueError(
            f"API request failed with status code {response.status_code}: {response.text}"
        )

    plugins_response = response.json()

    # Get plugin classes
    plugin_classes = await get_plugin_classes(plugins_response)

    # Sort plugin classes alphabetically
    sorted_plugin_classes = sorted(plugin_classes)

    # Get descriptions for each plugin
    plugin_data = []
    for plugin_class in sorted_plugin_classes:
        description = await get_plugin_description(plugin_class)
        plugin_data.append((plugin_class, description))

    # Write to CSV file
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pluginClass", "description"])  # Header
        writer.writerows(plugin_data)

    return f"Successfully wrote {len(plugin_data)} plugin entries to {output_file}"


def read_plugins() -> List[PluginInfo]:
    """Read plugins from the local plugins.csv file."""
    try:
        plugins = []
        with open("plugins.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                plugins.append(
                    PluginInfo(
                        pluginClass=str(row["pluginClass"]),
                        description=str(row["description"]),
                    )
                )
        return plugins
    except FileNotFoundError:
        print("Error: plugins.csv file not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading plugins.csv: {str(e)}")
        sys.exit(1)
