"""Fake LLM runnables so workflows can be tested without network/API keys.

The real ``Models`` bundle is a plain dataclass, which is exactly what makes
this possible: every field can be swapped for a fake runnable that returns
fixed, deterministic structured outputs.
"""

from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.runnables import RunnableLambda

from scripting.llms import Models
from scripting.schemas import (
    EvaluatorCritic,
    OutlinePlan,
    RefinedArticleOutput,
    Route,
    ScientistAchievementsOutput,
    ScientistArticleOutput,
    ScientistHistoryOutput,
    Section,
)


class Response:
    """Minimal stand-in for a LangChain ``AIMessage``.

    Nodes only read ``content`` and ``tool_calls`` off chat responses, so a
    tiny namespace object is enough instead of constructing real messages.
    Using a plain object (rather than a real ``AIMessage``) keeps the suite
    free of LangChain message plumbing and any token/credential handling.
    """

    def __init__(
        self,
        content: str = "",
        tool_calls: Optional[list] = None,
        **kwargs: Any,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls
        # Allow arbitrary extra attributes (e.g. response_metadata) so callers
        # can mimic whichever fields a node happens to touch.
        for key, value in kwargs.items():
            setattr(self, key, value)


class ToolLLM:
    """Fake tool-bound chat model (returns no tool calls by default).

    ``make_research_fn`` calls ``invoke`` and inspects ``tool_calls``; without
    any tool calls it just returns the baked-in content, so the tools workflow
    runs offline without Tavily.
    """

    def __init__(self, content: str = "RESEARCH", tool_calls: Optional[list] = None):
        # Defaults: no tool calls (so no real web search is attempted) and a
        # fixed research string that ends up in the generated sections.
        self._content = content
        self._tool_calls = tool_calls

    def invoke(self, _messages, *args: Any, **kwargs: Any) -> Response:
        # Ignore the prompt entirely; always return the same deterministic
        # response.  The signature mirrors ``BaseChatModel.invoke`` loosely so
        # it can be substituted transparently.
        return Response(content=self._content, tool_calls=self._tool_calls)


def make_fake_models(
    *,
    route_step: str = "article",
    eval_status: str = "ACCEPTED",
    eval_feedback: str = "Looks good.",
    article: str = "This is a short generated article.",
    refined: str = "This is the refined article.",
) -> Models:
    """Return a ``Models`` bundle wired to deterministic fakes.

    ``RunnableLambda`` is a real LangChain runnable, so chaining still works
    (``prompt | fake_runnable``) exactly as with a live model.  Structured
    schemas are returned directly because nodes read attributes off them.
    """
    return Models(
        # Generic chat model slot; content is never inspected for this fake.
        google_llm=RunnableLambda(lambda _: Response(content="Generated content.")),
        # Evaluator variant used by the feedback subgraph; status/feedback are
        # parameterised so tests can force ACCEPTED or REJECTED flows.
        google_llm_evaluator=RunnableLambda(
            lambda _: EvaluatorCritic(status=eval_status, feedback=eval_feedback)
        ),
        # Fixed history text: tests assert this exact sentinel value survives
        # the graph (e.g. test_routing_decision_paths).
        history_llm=RunnableLambda(
            lambda _: ScientistHistoryOutput(history="Fake history.")
        ),
        # Exactly five achievements so downstream word/len assumptions hold.
        achievements_llm=RunnableLambda(
            lambda _: ScientistAchievementsOutput(
                achievements=["A1", "A2", "A3", "A4", "A5"]
            )
        ),
        # The raw article.  Parameterised so callers can pick long/short text.
        article_llm=RunnableLambda(
            lambda _: ScientistArticleOutput(article=article)
        ),
        # Short article default -> the 1000-word refinement branch stays off.
        refined_article_llm=RunnableLambda(
            lambda _: RefinedArticleOutput(refined_article=refined)
        ),
        # Deterministic two-section outline for the orchestrator's plan node.
        planner_llm=RunnableLambda(
            lambda _: OutlinePlan(
                sections=[
                    Section(title="History", description="Early life and context."),
                    Section(
                        title="Achievements", description="Major scientific work."
                    ),
                ]
            )
        ),
        # Drives the routing workflow down a chosen branch (history/etc.).
        router=RunnableLambda(lambda _: Route(step=route_step)),
        # Standalone evaluator runnable used outside the combined subgraph.
        evaluator_llm=RunnableLambda(
            lambda _: EvaluatorCritic(status=eval_status, feedback=eval_feedback)
        ),
        # Tool-bound model + empty tool registry: research nodes that would
        # otherwise call Tavily get a no-op offline model instead.
        llm_with_tools=ToolLLM(),
        tools=[],
        tools_by_name={},
    )


__all__: List[str] = ["Response", "ToolLLM", "make_fake_models"]
