"""Converted from ``routing_with_feedback.ipynb``.

Routing architecture with evaluator feedback loop inside the article subgraph.
Top-level: extract scientist name -> llm_routing branches to history/accomplishments/article_workflow.
When routed to article_workflow (nested subgraph), the subgraph runs parallel
history+achievements -> article -> refine -> evaluate_feedback_llm; on REJECTED
loops to refine_article_with_feedback (bounded by MAX_FEEDBACK_RETRIES), on ACCEPTED exits subgraph to END.

ASCII flow diagram (actual node names from build_graph):
    START -> scientist_name_extraction -> llm_routing
                                           ├--> generate_history -> END
                                           ├--> generate_achievements -> END
                                           └--> article_workflow -> END
                                                (article_workflow, with feedback):
                                                    START --┬--> generate_history ----------┐
                                                           └--> generate_achievements -----+--> generate_article -> generate_refined_article -> evaluate_feedback_llm
                                                                                                                                                ├--> END (ACCEPTED)
                                                                                                                                                └--> refine_article_with_feedback -> evaluate_feedback_llm (loop, REJECTED)
                                                                                                                                                     (bounded by MAX_FEEDBACK_RETRIES)

Nodes (top-level):
- scientist_name_extraction: Extract scientist name from free-text input.
- llm_routing: LLM decides branch (history/accomplishment/article).
- generate_history: Terminal history branch.
- generate_achievements: Terminal achievements branch.
- article_workflow: Nested subgraph with feedback loop.

Nodes (article_workflow):
- generate_history, generate_achievements (parallel from START in subgraph)
- generate_article, generate_refined_article
- refine_article_with_feedback, evaluate_feedback_llm

"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .. import nodes
from ..llms import build_models
from ..schemas import RoutingState
from .routing import make_llm_routing, route_decision

NAME = "routing_feedback"
KIND = "routing"
HAS_FEEDBACK = True
HAS_TOOLS = False
INPUT_KIND = "input"
DESCRIPTION = "Routing workflow with evaluator feedback loop."


def initial_state(payload: dict) -> dict:
    """Initialize state from payload. Expects 'input' (str). Returns {'input': ...}
    matching RoutingState shape."""
    return {"input": payload.get("input", "")}


def build_article_workflow(models):
    """Build nested article subgraph with feedback loop. Parallel history+achievements
    from START inside subgraph, converge at generate_article, refine, then evaluate;
    on REJECTED loop to refine_article_with_feedback, on ACCEPTED exit to subgraph END."""
    article_workflow = StateGraph(RoutingState)
    # Node roles inside nested subgraph:
    # generate_history: produce history (parallel)
    article_workflow.add_node("generate_history", nodes.make_generate_history(models))
    # generate_achievements: produce achievements (parallel)
    article_workflow.add_node(
        "generate_achievements", nodes.make_generate_achievements(models)
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
    models = models or build_models(routing=True, feedback=True)

    workflow = StateGraph(RoutingState)
    # Routing shell: the feedback loop lives inside the article subgraph.

    # Node role: Extract scientist name from free-text input.
    workflow.add_node(
        "scientist_name_extraction", nodes.make_scientist_name_extraction(models)
    )
    # Node role: Generate history (terminal branch).
    workflow.add_node("generate_history", nodes.make_generate_history(models))
    # Node role: Generate achievements (terminal branch).
    workflow.add_node(
        "generate_achievements", nodes.make_generate_achievements(models)
    )
    # Node role: Nested subgraph with feedback loop for full article.
    workflow.add_node("article_workflow", build_article_workflow(models))
    # Node role: LLM routing decision.
    workflow.add_node("llm_routing", make_llm_routing(models))

    # Top-level flow.
    workflow.add_edge(START, "scientist_name_extraction")
    workflow.add_edge("scientist_name_extraction", "llm_routing")
    # Conditional routing by decision.
    workflow.add_conditional_edges(
        "llm_routing",
        route_decision,
        {
            "generate_history": "generate_history",
            "generate_achievements": "generate_achievements",
            "article_workflow": "article_workflow",
        },
    )
    # Terminal branches exit to END.
    workflow.add_edge("generate_history", END)
    workflow.add_edge("generate_achievements", END)
    workflow.add_edge("article_workflow", END)

    return workflow.compile()


def run(payload: dict, models=None, callbacks=None) -> dict:
    # Routing + feedback: the router and the evaluator LLM are both required.
    models = models or build_models(routing=True, feedback=True)
    graph = build_graph(models)
    config = {"callbacks": callbacks} if callbacks else {}
    return graph.invoke(initial_state(payload), config=config)
