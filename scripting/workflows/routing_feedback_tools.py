"""Converted from ``routing_with_feedback_tools.ipynb``.

Routing + feedback loop with web-search grounding for all generation nodes.
Top-level: extract scientist name -> llm_routing branches to history/accomplishments/article_workflow.
All generation nodes (history/achievements in top-level branches and inside article_workflow)
are created with research fn. Article subgraph has feedback loop: evaluate -> ACCEPTED exits to subgraph END,
or REJECTED -> refine_article_with_feedback -> evaluate (bounded by MAX_FEEDBACK_RETRIES).

ASCII flow diagram (actual node names from build_graph):
    START -> scientist_name_extraction -> llm_routing
                                           ├--> generate_history (grounded) -> END
                                           ├--> generate_achievements (grounded) -> END
                                           └--> article_workflow -> END
                                                (article_workflow, with feedback, grounded):
                                                    START --┬--> generate_history (grounded) ----------┐
                                                           └--> generate_achievements (grounded) -----+--> generate_article -> generate_refined_article -> evaluate_feedback_llm
                                                                                                                                                      ├--> END (ACCEPTED)
                                                                                                                                                      └--> refine_article_with_feedback -> evaluate_feedback_llm (loop, REJECTED)
                                                                                                                                                           (bounded by MAX_FEEDBACK_RETRIES)

Nodes (top-level):
- scientist_name_extraction: Extract scientist name from free-text input.
- llm_routing: LLM decides branch.
- generate_history: Terminal branch (web-search grounded).
- generate_achievements: Terminal branch (web-search grounded).
- article_workflow: Nested subgraph (grounded + feedback loop).

Nodes (article_workflow):
- generate_history, generate_achievements (parallel, grounded)
- generate_article, generate_refined_article
- refine_article_with_feedback, evaluate_feedback_llm

"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .. import nodes
from ..llms import build_models
from ..schemas import RoutingState
from ..tools import make_research_fn
from .routing import make_llm_routing, route_decision

NAME = "routing_feedback_tools"
KIND = "routing"
HAS_FEEDBACK = True
HAS_TOOLS = True
INPUT_KIND = "input"
DESCRIPTION = "Routing + feedback + web-search grounded generation."


def initial_state(payload: dict) -> dict:
    """Initialize state from payload. Expects 'input' (str). Returns {'input': ...}
    matching RoutingState shape."""
    return {"input": payload.get("input", "")}


def build_article_workflow(models, research):
    """Build nested article subgraph with feedback loop and web-search grounding
    for generation nodes. Parallel history+achievements (grounded) from START inside
    subgraph, converge at generate_article, refine, evaluate; on REJECTED loop to
    refine_article_with_feedback, on ACCEPTED exit to subgraph END."""
    article_workflow = StateGraph(RoutingState)
    # Node roles inside nested subgraph (grounded):
    # generate_history: produce history (web-search grounded)
    article_workflow.add_node(
        "generate_history", nodes.make_generate_history(models, research)
    )
    # generate_achievements: produce achievements (web-search grounded)
    article_workflow.add_node(
        "generate_achievements", nodes.make_generate_achievements(models, research)
    )
    # generate_article: compose article
    article_workflow.add_node("generate_article", nodes.make_generate_article(models))
    # generate_refined_article: refine composed article
    article_workflow.add_node(
        "generate_refined_article", nodes.make_generate_refined_article(models)
    )
    # refine_article_with_feedback: revise using feedback
    article_workflow.add_node(
        "refine_article_with_feedback",
        nodes.make_refine_article_with_feedback(models),
    )
    # evaluate_feedback_llm: evaluate; routes ACCEPTED/REJECTED
    article_workflow.add_node(
        "evaluate_feedback_llm", nodes.make_llm_evaluator(models)
    )

    # Parallel fan-out inside subgraph.
    article_workflow.add_edge(START, "generate_history")
    article_workflow.add_edge(START, "generate_achievements")
    # Join at generate_article.
    article_workflow.add_edge("generate_history", "generate_article")
    article_workflow.add_edge("generate_achievements", "generate_article")
    # To refinement then evaluation.
    article_workflow.add_edge("generate_article", "generate_refined_article")
    article_workflow.add_edge("generate_refined_article", "evaluate_feedback_llm")
    # Loop edge from revision back to evaluation.
    article_workflow.add_edge(
        "refine_article_with_feedback", "evaluate_feedback_llm"
    )
    # Feedback loop inside the article subgraph: revise until accepted.
    article_workflow.add_conditional_edges(
        "evaluate_feedback_llm",
        nodes.route_feedback,
        {
            "ACCEPTED": END,
            "REJECTED": "refine_article_with_feedback",
        },
    )

    return article_workflow.compile()


def build_graph(models=None):
    models = models or build_models(routing=True, feedback=True, tools=True)
    research = make_research_fn(models.llm_with_tools, models.tools_by_name)

    workflow = StateGraph(RoutingState)
    # Routing shell whose article subgraph is both web-search grounded and
    # wrapped in the evaluator feedback loop.

    # Node role: Extract scientist name from free-text input.
    workflow.add_node(
        "scientist_name_extraction", nodes.make_scientist_name_extraction(models)
    )
    # Node role: Generate history (terminal branch, web-search grounded).
    workflow.add_node(
        "generate_history", nodes.make_generate_history(models, research)
    )
    # Node role: Generate achievements (terminal branch, web-search grounded).
    workflow.add_node(
        "generate_achievements", nodes.make_generate_achievements(models, research)
    )
    # Node role: Nested subgraph with grounded generation + feedback loop.
    workflow.add_node("article_workflow", build_article_workflow(models, research))
    # Node role: LLM routing decision.
    workflow.add_node("llm_routing", make_llm_routing(models))

    # Top-level flow.
    workflow.add_edge(START, "scientist_name_extraction")
    workflow.add_edge("scientist_name_extraction", "llm_routing")
    # Conditional routing.
    workflow.add_conditional_edges(
        "llm_routing",
        route_decision,
        {
            "generate_history": "generate_history",
            "generate_achievements": "generate_achievements",
            "article_workflow": "article_workflow",
        },
    )
    # Terminal branches.
    workflow.add_edge("generate_history", END)
    workflow.add_edge("generate_achievements", END)
    workflow.add_edge("article_workflow", END)

    return workflow.compile()


def run(payload: dict, models=None, callbacks=None) -> dict:
    # Routing + feedback + tools: every optional model is required here.
    models = models or build_models(routing=True, feedback=True, tools=True)
    graph = build_graph(models)
    config = {"callbacks": callbacks} if callbacks else {}
    return graph.invoke(initial_state(payload), config=config)
