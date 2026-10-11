"""Converted from ``prompt_chaining_with_feedback.ipynb``.

Sequential chain architecture plus an evaluator feedback loop. After generating
a refined article, the evaluator (evaluate_feedback_llm via nodes.route_feedback)
routes to END if ACCEPTED, or to refine_article_with_feedback if REJECTED. The
revision path loops back to the evaluator, bounded by nodes.MAX_FEEDBACK_RETRIES.

ASCII flow diagram (actual node names from build_graph):
    START -> generate_history -> generate_achievements -> generate_article -> generate_refined_article -> evaluate_feedback_llm
                                                                                                            ├--> END (ACCEPTED)
                                                                                                            └--> refine_article_with_feedback -> evaluate_feedback_llm (loop, REJECTED)
                                                                                                                 (bounded by MAX_FEEDBACK_RETRIES)

Nodes:
- generate_history: Produce scientist history.
- generate_achievements: Produce scientist achievements.
- generate_article: Compose article from history+achievements.
- generate_refined_article: Refine composed article.
- refine_article_with_feedback: Revise article using feedback from evaluator (loop entry on REJECTED).
- evaluate_feedback_llm: Evaluate article/feedback state; routes ACCEPTED->END or REJECTED->refine_article_with_feedback.

"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .. import nodes
from ..llms import build_models
from ..schemas import FeedbackState, ScientistInput

NAME = "prompt_chaining_feedback"
KIND = "prompt_chaining"
HAS_FEEDBACK = True
HAS_TOOLS = False
INPUT_KIND = "scientist"
DESCRIPTION = "Sequential chain plus evaluator feedback loop."


def initial_state(payload: dict) -> dict:
    """Initialize state from payload. Expects 'scientist' (str); validates via
    ScientistInput and sets state['scientist']=validated.scientist (FeedbackState shape)."""
    validated = ScientistInput.model_validate(
        {"scientist": payload.get("scientist", "")}
    )
    return {"scientist": validated.scientist}


def build_graph(models=None):
    models = models or build_models(feedback=True)

    workflow = StateGraph(FeedbackState)
    # Generation chain followed by an evaluator loop: REJECTED articles are
    # revised until the critic accepts or MAX_FEEDBACK_RETRIES is reached.

    # Node role: Generate history.
    workflow.add_node("generate_history", nodes.make_generate_history(models))
    # Node role: Generate achievements.
    workflow.add_node(
        "generate_achievements", nodes.make_generate_achievements(models)
    )
    # Node role: Compose article from history+achievements.
    workflow.add_node("generate_article", nodes.make_generate_article(models))
    # Node role: Refine composed article.
    workflow.add_node(
        "generate_refined_article", nodes.make_generate_refined_article(models)
    )
    # Node role: Revise article using evaluator feedback (loop target on REJECTED).
    workflow.add_node(
        "refine_article_with_feedback",
        nodes.make_refine_article_with_feedback(models),
    )
    # Node role: Evaluate article state; returns ACCEPTED/REJECTED (bounded by MAX_FEEDBACK_RETRIES).
    workflow.add_node("evaluate_feedback_llm", nodes.make_llm_evaluator(models))

    # Linear generation edges.
    workflow.add_edge(START, "generate_history")
    workflow.add_edge("generate_history", "generate_achievements")
    workflow.add_edge("generate_achievements", "generate_article")
    # Feedback loop: the evaluator verdict decides between ending and revising.
    workflow.add_edge("generate_article", "generate_refined_article")
    workflow.add_edge("generate_refined_article", "evaluate_feedback_llm")
    # Loop back: revised article re-enters evaluation.
    workflow.add_edge("refine_article_with_feedback", "evaluate_feedback_llm")
    # Conditional edges: route_feedback maps evaluator verdict to next node.
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
    # ``feedback=True`` attaches the evaluator LLM to the models bundle.
    models = models or build_models(feedback=True)
    graph = build_graph(models)
    config = {"callbacks": callbacks} if callbacks else {}
    return graph.invoke(initial_state(payload), config=config)
