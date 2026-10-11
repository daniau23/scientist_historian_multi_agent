"""Converted from ``prompt_chaining_with_feedback_tools.ipynb``.

Prompt chaining + feedback loop + web-search (tool) grounding. Same graph structure
as prompt_chaining_feedback, but history and achievements nodes are created with
a research function (passed via node factories) so they can ground generation using
web search. Feedback loop: evaluate_feedback_llm routes ACCEPTED->END or
REJECTED->refine_article_with_feedback (bounded by MAX_FEEDBACK_RETRIES).

ASCII flow diagram (actual node names from build_graph):
    START -> generate_history -> generate_achievements -> generate_article -> generate_refined_article -> evaluate_feedback_llm
                                                                                                            ├--> END (ACCEPTED)
                                                                                                            └--> refine_article_with_feedback -> evaluate_feedback_llm (loop, REJECTED)
                                                                                                                 (bounded by MAX_FEEDBACK_RETRIES)

Nodes:
- generate_history: Produce history (web-search grounded via research fn).
- generate_achievements: Produce achievements (web-search grounded via research fn).
- generate_article: Compose article.
- generate_refined_article: Refine article.
- refine_article_with_feedback: Revise using feedback.
- evaluate_feedback_llm: Evaluate; routes ACCEPTED/REJECTED.

"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .. import nodes
from ..llms import build_models
from ..schemas import FeedbackState, ScientistInput
from ..tools import make_research_fn

NAME = "prompt_chaining_feedback_tools"
KIND = "prompt_chaining"
HAS_FEEDBACK = True
HAS_TOOLS = True
INPUT_KIND = "scientist"
DESCRIPTION = "Sequential chain + feedback + web-search grounded generation."


def initial_state(payload: dict) -> dict:
    """Initialize state from payload. Expects 'scientist' (str); validates via
    ScientistInput and sets scientist in state (FeedbackState shape)."""
    validated = ScientistInput.model_validate(
        {"scientist": payload.get("scientist", "")}
    )
    return {"scientist": validated.scientist}


def build_graph(models=None):
    models = models or build_models(feedback=True, tools=True)
    research = make_research_fn(models.llm_with_tools, models.tools_by_name)

    workflow = StateGraph(FeedbackState)
    # Language/tool grounding: history and achievements first query the web
    # (see ``research``), then the feedback loop refines the article.

    # Node role: Generate history (web-search grounded via research fn).
    workflow.add_node(
        "generate_history", nodes.make_generate_history(models, research)
    )
    # Node role: Generate achievements (web-search grounded via research fn).
    workflow.add_node(
        "generate_achievements", nodes.make_generate_achievements(models, research)
    )
    # Node role: Compose article.
    workflow.add_node("generate_article", nodes.make_generate_article(models))
    # Node role: Refine composed article.
    workflow.add_node(
        "generate_refined_article", nodes.make_generate_refined_article(models)
    )
    # Node role: Revise using evaluator feedback.
    workflow.add_node(
        "refine_article_with_feedback",
        nodes.make_refine_article_with_feedback(models),
    )
    # Node role: Evaluate state; routes ACCEPTED/REJECTED (bounded by MAX_FEEDBACK_RETRIES).
    workflow.add_node("evaluate_feedback_llm", nodes.make_llm_evaluator(models))

    # Linear flow with feedback loop.
    workflow.add_edge(START, "generate_history")
    workflow.add_edge("generate_history", "generate_achievements")
    workflow.add_edge("generate_achievements", "generate_article")
    workflow.add_edge("generate_article", "generate_refined_article")
    workflow.add_edge("generate_refined_article", "evaluate_feedback_llm")
    # Loop back from revision to evaluation.
    workflow.add_edge("refine_article_with_feedback", "evaluate_feedback_llm")
    # Feedback loop: stay on the revise path until the critic accepts.
    workflow.add_conditional_edges(
        "evaluate_feedback_llm",
        nodes.route_feedback,
        {
            "ACCEPTED": END,
            "REJECTED": "refine_article_with_feedback",
        },
    )

    return workflow.compile()


def run(payload: dict, models=None, callbacks=None) -> dict:
    # Feedback + tools: evaluator LLM and a tool-bound model are both needed.
    models = models or build_models(feedback=True, tools=True)
    graph = build_graph(models)
    config = {"callbacks": callbacks} if callbacks else {}
    return graph.invoke(initial_state(payload), config=config)
