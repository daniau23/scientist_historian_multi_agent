"""Converted from ``orchestrator_worker_with_feedback_tools.ipynb``.

Orchestrator-worker + feedback loop with web-search grounding. Same structure as
orchestrator_worker_feedback, but worker_section_writer is created with research
fn so each worker grounds its section using web search. Feedback loop:
evaluate_feedback_llm routes ACCEPTED->END or REJECTED->refine_article_with_feedback
(looping back to evaluator), bounded by nodes.MAX_FEEDBACK_RETRIES. Workers spawned
via Send (nodes.assign_workers); completed_sections uses operator.add reducer.

ASCII flow diagram (actual node names from build_graph):
    START -> scientist_name_extraction -> orchestrator --Send--> worker_section_writer (xN, grounded) -> synthesizer -> evaluate_feedback_llm
                                                                                                             ├--> END (ACCEPTED)
                                                                                                             └--> refine_article_with_feedback -> evaluate_feedback_llm (loop, REJECTED)
                                                                                                                  (bounded by MAX_FEEDBACK_RETRIES)

Nodes:
- scientist_name_extraction: Extract scientist name from free-text input.
- orchestrator: Plan sections dynamically.
- worker_section_writer: Write one grounded section per planned section (parallel via Send; research fn provided).
- synthesizer: Synthesize completed_sections into article (feedback_fields=True).
- refine_article_with_feedback: Revise using evaluator feedback.
- evaluate_feedback_llm: Evaluate article state; routes ACCEPTED/REJECTED.

"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .. import nodes
from ..llms import build_models
from ..schemas import OrchestratorFeedbackState
from ..tools import make_research_fn

NAME = "orchestrator_worker_feedback_tools"
KIND = "orchestrator_worker"
HAS_FEEDBACK = True
HAS_TOOLS = True
INPUT_KIND = "input"
DESCRIPTION = "Orchestrator-worker + feedback + web-search grounded sections."


def initial_state(payload: dict) -> dict:
    """Initialize state from payload. Expects 'input' (str). Sets
    {'input': ..., 'completed_sections': []} matching OrchestratorFeedbackState."""
    return {"input": payload.get("input", ""), "completed_sections": []}


def build_graph(models=None):
    models = models or build_models(orchestrator=True, feedback=True, tools=True)
    research = make_research_fn(models.llm_with_tools, models.tools_by_name)

    workflow = StateGraph(OrchestratorFeedbackState)
    # Orchestrator-worker with research-grounded sections, followed by the
    # evaluator feedback loop over the synthesised article.

    # Node role: Extract scientist name from free-text input.
    workflow.add_node(
        "scientist_name_extraction", nodes.make_scientist_name_extraction(models)
    )
    # Node role: Plan sections dynamically.
    workflow.add_node("orchestrator", nodes.make_orchestrator(models))
    # Node role: Write one section per planned section (parallel via Send; grounded by research fn).
    workflow.add_node(
        "worker_section_writer", nodes.make_worker_section_writer(models, research)
    )
    # Node role: Synthesize completed_sections into article (feedback_fields=True).
    workflow.add_node(
        "synthesizer", nodes.make_synthesizer(models, feedback_fields=True)
    )
    # Node role: Revise article using evaluator feedback.
    workflow.add_node(
        "refine_article_with_feedback",
        nodes.make_refine_article_with_feedback(models),
    )
    # Node role: Evaluate article state; routes ACCEPTED/REJECTED.
    workflow.add_node("evaluate_feedback_llm", nodes.make_llm_evaluator(models))

    # Orchestrator-worker flow, then evaluator feedback loop.
    workflow.add_edge(START, "scientist_name_extraction")
    workflow.add_edge("scientist_name_extraction", "orchestrator")
    # Conditional edge: fan-out via Send API (nodes.assign_workers) to worker_section_writer (xN).
    workflow.add_conditional_edges(
        "orchestrator", nodes.assign_workers, ["worker_section_writer"]
    )
    # Workers join at synthesizer after all parallel sections complete.
    workflow.add_edge("worker_section_writer", "synthesizer")
    # Synthesized article -> evaluation.
    workflow.add_edge("synthesizer", "evaluate_feedback_llm")
    # Loop: revised article re-evaluated.
    workflow.add_edge("refine_article_with_feedback", "evaluate_feedback_llm")
    # Conditional routing by evaluator verdict.
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
    # Orchestrator + feedback + tools: planner, evaluator and tool model needed.
    models = models or build_models(orchestrator=True, feedback=True, tools=True)
    graph = build_graph(models)
    config = {"callbacks": callbacks} if callbacks else {}
    return graph.invoke(initial_state(payload), config=config)
