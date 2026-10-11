"""Converted from ``parallelization_with_feedback.ipynb``.

Parallel history+achievements architecture plus evaluator feedback loop. After
parallel generation converges at article, then refine, the evaluator routes
ACCEPTED->END or REJECTED->refine_article_with_feedback (looping back to evaluator),
bounded by nodes.MAX_FEEDBACK_RETRIES.

ASCII flow diagram (actual node names from build_graph):
    START --┬--> generate_history ----------┐
           └--> generate_achievements -----+--> generate_article -> generate_refined_article -> evaluate_feedback_llm
                                                                                               ├--> END (ACCEPTED)
                                                                                               └--> refine_article_with_feedback -> evaluate_feedback_llm (loop, REJECTED)
                                                                                                    (bounded by MAX_FEEDBACK_RETRIES)

Nodes:
- generate_history: Produce history (parallel).
- generate_achievements: Produce achievements (parallel).
- generate_article: Compose article from both branches.
- generate_refined_article: Refine composed article.
- refine_article_with_feedback: Revise using evaluator feedback.
- evaluate_feedback_llm: Evaluate state; routes ACCEPTED/REJECTED.

"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .. import nodes
from ..llms import build_models
from ..schemas import FeedbackState, ScientistInput

NAME = "parallelization_feedback"
KIND = "parallelization"
HAS_FEEDBACK = True
HAS_TOOLS = False
INPUT_KIND = "scientist"
DESCRIPTION = "Parallel generation plus evaluator feedback loop."


def initial_state(payload: dict) -> dict:
    """Initialize state from payload. Expects 'scientist' (str); validates via
    ScientistInput, returns FeedbackState-compatible dict with scientist."""
    validated = ScientistInput.model_validate(
        {"scientist": payload.get("scientist", "")}
    )
    return {"scientist": validated.scientist}


def build_graph(models=None):
    models = models or build_models(feedback=True)

    workflow = StateGraph(FeedbackState)
    # Parallel generation, then the shared refinement + evaluator feedback loop.

    # Node role: Generate history (parallel).
    workflow.add_node("generate_history", nodes.make_generate_history(models))
    # Node role: Generate achievements (parallel).
    workflow.add_node(
        "generate_achievements", nodes.make_generate_achievements(models)
    )
    # Node role: Compose article from both branches.
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
    # Node role: Evaluate state; route ACCEPTED/REJECTED.
    workflow.add_node("evaluate_feedback_llm", nodes.make_llm_evaluator(models))

    # Parallel branches converge at generate_article, then feedback loop.
    workflow.add_edge(START, "generate_history")
    workflow.add_edge(START, "generate_achievements")
    workflow.add_edge("generate_history", "generate_article")
    workflow.add_edge("generate_achievements", "generate_article")
    workflow.add_edge("generate_article", "generate_refined_article")
    workflow.add_edge("generate_refined_article", "evaluate_feedback_llm")
    # Loop: revised article re-evaluated.
    workflow.add_edge("refine_article_with_feedback", "evaluate_feedback_llm")
    # Conditional routing by evaluator.
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
