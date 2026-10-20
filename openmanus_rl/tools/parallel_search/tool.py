"""Anonymous Parallel Search MCP tool for one GAIA research task."""

import asyncio
import uuid
from urllib.parse import urlsplit

import anyio
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from openmanus_rl.tools.base import BaseTool

MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "OpenManus-RL/parallel-search (+https://github.com/OpenManus/OpenManus-RL)"


class ParallelSearchTool(BaseTool):
    """Search and fetch using the same conversation ID until the task resets."""

    def __init__(self, timeout=30):
        super().__init__(
            tool_name="parallel_search",
            tool_description=(
                "Search the web without an API key, or fetch relevant excerpts "
                "from HTTP/HTTPS URLs. Supply query for search, urls for fetch."
            ),
            tool_version="1.0.0",
            input_types={
                "query": "str - Search query, or optional focus for fetching URLs.",
                "urls": "list[str] - Optional HTTP/HTTPS URLs to fetch (up to 20).",
            },
            output_type="dict - Source URLs, titles and excerpts, or an error.",
        )
        self.timeout = timeout
        self.start_session()

    def start_session(self):
        """Called by the owning ToolUseEnv when it starts a new task."""
        self.session_id = uuid.uuid4().hex

    async def _call(self, name, arguments):
        # Connections are short lived; conversation identity is supplied by the
        # task-owned tool, independently of the MCP transport's session ID.
        with anyio.fail_after(self.timeout):
            async with httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT},
                timeout=self.timeout,
            ) as client, streamable_http_client(
                MCP_URL,
                http_client=client,
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments)
                    if result.isError:
                        return {
                            "error": "\n".join(
                                block.text
                                for block in result.content
                                if block.type == "text"
                            )
                        }
                    if result.structuredContent is not None:
                        return result.structuredContent
                    return {
                        "output": "\n".join(
                            block.text
                            for block in result.content
                            if block.type == "text"
                        )
                    }

    def execute(self, query="", urls=None):
        if not isinstance(query, str):
            return {"error": "query must be a string"}
        if urls is not None:
            if not isinstance(urls, list) or not 1 <= len(urls) <= 20:
                return {"error": "urls must contain 1 to 20 HTTP/HTTPS URLs"}
            for url in urls:
                try:
                    parsed = urlsplit(url) if isinstance(url, str) else None
                    if (
                        parsed is None
                        or parsed.scheme not in ("http", "https")
                        or not parsed.hostname
                        or parsed.username
                        or parsed.password
                    ):
                        return {
                            "error": "Only HTTP/HTTPS URLs without credentials are accepted"
                        }
                except ValueError:
                    return {"error": "Invalid URL"}
            name = "web_fetch"
            arguments = {"urls": urls}
            if query.strip():
                arguments["objective"] = query[:200]
        else:
            if not query.strip():
                return {"error": "A nonempty query is required for search"}
            name = "web_search"
            arguments = {"objective": query, "search_queries": [query]}
        arguments["session_id"] = self.session_id
        try:
            return asyncio.run(self._call(name, arguments))
        except TimeoutError:
            return {"error": "Parallel request timed out"}
        except Exception as exc:
            # Do not retry another provider or regenerate the conversation ID.
            while getattr(exc, "exceptions", None) and len(exc.exceptions) == 1:
                exc = exc.exceptions[0]
            return {"error": f"Parallel request failed: {exc}"}
