"""Web-search tooling used by the ``*_tools`` workflow variants.

The tool definition and the research helper are direct ports of the notebook
cells.  Imports of the search providers happen lazily so the rest of the
package remains importable without the optional dependencies installed.

Bring-your-own-key: ``build_tools`` accepts a user-supplied Tavily API key.  An
*empty string* means "the user selected a tools workflow but did not provide a
key", in which case the tool is built in a disabled state that tells the user
how to enable it — and, crucially, never falls back to a host key.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

# Returned by the ``web_search`` tool when no Tavily key was supplied.  It is a
# normal string (not an exception) so the workflow still completes and the LLM
# can relay the limitation to the user.
MISSING_TAVILY_KEY_MESSAGE = (
    "WEB_SEARCH_UNAVAILABLE: no Tavily API key was provided. "
    "Paste your own Tavily API key in the sidebar to enable web search."
)


def _make_web_search(tavily_api_key: Optional[str] = None) -> Any:
    # Imported lazily: Tavily is only required for the *_tools workflows.
    from langchain_core.tools import tool

    @tool
    def web_search(query: str) -> str:
        """Searches the web for factual information regarding scientists,
        their history, and their achievements.

        Tavily is used as the external search provider.
        """
        # Explicitly-empty key: the user picked a tools workflow but gave no
        # key, so return a helpful message instead of searching (and instead of
        # silently using a server-side key).
        if tavily_api_key == "":
            return MISSING_TAVILY_KEY_MESSAGE

        try:
            from langchain_tavily import TavilySearch

            # Small, fast config: one result, no raw HTML, with a summary.
            search_kwargs: Dict[str, Any] = {
                "max_results": 1,
                "topic": "general",
                "search_depth": "basic",
                "include_answer": True,
                "include_raw_content": False,
            }
            # A user-supplied key always wins; ``None`` keeps the library's
            # ``TAVILY_API_KEY`` environment fallback (used by the local CLI).
            if tavily_api_key:
                search_kwargs["tavily_api_key"] = tavily_api_key
            search_tool = TavilySearch(**search_kwargs)
            result = search_tool.invoke({"query": query})
            if not result:
                return "No search results were returned."

            # Re-format the raw Tavily payload into a compact research report.
            research_parts: List[str] = []
            answer = result.get("answer")
            if answer:
                research_parts.append(f"SEARCH SUMMARY:\n{answer}")

            for index, item in enumerate(result.get("results", []), start=1):
                title = item.get("title", "Untitled")
                url = item.get("url", "")
                content = item.get("content", "")
                research_parts.append(
                    "\n".join(
                        [
                            f"SOURCE {index}",
                            f"Title: {title}",
                            f"URL: {url}",
                            "Content:",
                            content,
                        ]
                    ).strip()
                )

            if not research_parts:
                return "No useful external research was returned."
            return "\n\n".join(research_parts)
        except Exception as exc:  # pragma: no cover - network dependent
            # Never fail the workflow because search broke: surface the error
            # as text the downstream LLM can still reason about.
            return f"WEB_SEARCH_ERROR: {type(exc).__name__}: {exc}"

    return web_search


def build_tools(
    tavily_api_key: Optional[str] = None,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Return ``(tools, tools_by_name)`` for the web search tool.

    ``tavily_api_key`` controls which key the tool bills:

    * a non-empty string -> the end user's own key;
    * an empty string -> disabled: the tool returns a "paste your key" message;
    * ``None`` -> fall back to ``TAVILY_API_KEY`` in the environment (CLI/local).
    """
    # ``tools`` goes to ``bind_tools``; ``tools_by_name`` lets the research
    # runner dispatch on the tool name chosen by the model.
    tools = [_make_web_search(tavily_api_key)]
    tools_by_name = {tool_obj.name: tool_obj for tool_obj in tools}
    return tools, tools_by_name


def make_research_fn(
    llm_with_tools: Any, tools_by_name: Dict[str, Any]
) -> Callable[[str], str]:
    """Build ``perform_research_with_tools`` bound to the given tool window.

    Gemini decides whether ``web_search`` is required; Python executes the
    requested tool calls and returns the combined research string.
    """

    def perform_research_with_tools(query_prompt: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        # System prompt steers the model towards using the tool only when it
        # would materially improve factual accuracy.
        messages = [
            SystemMessage(
                content=(
                    "You are a research assistant. "
                    "Decide whether web_search is necessary to answer "
                    "the user's request accurately. "
                    "Use web_search when external factual information "
                    "is needed or would materially improve accuracy. "
                    "If web search is not necessary, answer directly."
                )
            ),
            HumanMessage(content=query_prompt),
        ]

        try:
            # Let the model decide whether to call web_search.
            response = llm_with_tools.invoke(messages)
            tool_calls = getattr(response, "tool_calls", None)
            if not tool_calls:
                # No tool needed: the model answered directly.
                return str(response.content)

            # Execute each requested tool call and collect the observations.
            research_results: List[str] = []
            for tool_call in tool_calls:
                # Tool payloads can arrive as dicts; guard against surprises.
                if not isinstance(tool_call, dict):
                    continue
                tool_name = tool_call.get("name")
                if not tool_name:
                    continue
                tool_obj = tools_by_name.get(tool_name)
                if tool_obj is None:
                    research_results.append(f"Unknown tool requested: {tool_name}")
                    continue

                # Normalise arguments: JSON string or dict -> expected ``dict``.
                tool_args = tool_call.get("args", {})
                if tool_args is None:
                    tool_args = {}
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        # Fall back to treating the string as the query itself.
                        tool_args = {"query": tool_args}
                if not isinstance(tool_args, dict):
                    tool_args = {"query": str(tool_args)}
                if not tool_args.get("query"):
                    tool_args["query"] = query_prompt

                try:
                    observation = tool_obj.invoke(tool_args)
                    if observation:
                        research_results.append(str(observation))
                except Exception as exc:  # pragma: no cover - network dependent
                    research_results.append(
                        f"WEB_SEARCH_ERROR: {type(exc).__name__}: {exc}"
                    )

            if research_results:
                return "\n\n".join(research_results)
            return "No useful external research was returned."
        except Exception as exc:
            # Keep the generation nodes from crashing when research fails.
            return f"Research unavailable: {type(exc).__name__}: {exc}"

    return perform_research_with_tools
