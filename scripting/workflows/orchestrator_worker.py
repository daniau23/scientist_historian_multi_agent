"""Converted from ``orchestrator_worker.ipynb``.

Orchestrator-worker architecture: extract scientist name from input, orchestrator
plans sections dynamically, then uses LangGraph Send API (via nodes.assign_workers)
to fan out one worker_section_writer per planned section in parallel. Each worker
writes its assigned section; results accumulate into completed_sections (reducer
operator.add in state). Finally synthesizer combines sections into a coherent article.

ASCII flow diagram (actual node names from build_graph):
    START -> scientist_name_extraction -> orchestrator --Send--> worker_section_writer (xN) -> synthesizer -> END
                                                                    (fan-out via assign_workers)

Nodes:
- scientist_name_extraction: Extracts scientist name from free-text input (input -> Routing/Orchestrator style).
- orchestrator: Plans article sections dynamically based on input/scientist.
- worker_section_writer: Writes one section per planned section (spawned in parallel via Send; takes research fn optionally).
- synthesizer: Synthesizes completed_sections into final article (may include refinement logic depending on variant).

Note: conditional edge from orchestrator uses nodes.assign_workers to emit Send objects targeting worker_section_writer.

"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .. import nodes
from ..llms import build_models
from ..schemas import OrchestratorState

NAME = "orchestrator_worker"
KIND = "orchestrator_worker"
HAS_FEEDBACK = False
HAS_TOOLS = False
INPUT_KIND = "input"
DESCRIPTION = "Dynamic orchestrator plans sections; workers write them in parallel."


def initial_state(payload: dict) -> dict:
    """Initialize state from payload. Expects 'input' (str); sets
    {'input': ..., 'completed_sections': []} matching OrchestratorState shape
    (completed_sections uses operator.add reducer)."""
    return {"input": payload.get("input", ""), "completed_sections": []}


def build_graph(models=None, research=None):
    models = models or build_models(orchestrator=True)

    workflow = StateGraph(OrchestratorState)
    # The orchestrator plans sections; the conditional edge fans out to one
    # worker per section (via Send), then the synthesizer combines the results.

    # Node role: Extract scientist name from free-text input.
    workflow.add_node(
        "scientist_name_extraction", nodes.make_scientist_name_extraction(models)
    )
    # Node role: Plan article sections dynamically.
    workflow.add_node("orchestrator", nodes.make_orchestrator(models))
    # Node role: Write one section per planned section; spawned in parallel by Send.
    workflow.add_node(
        "worker_section_writer", nodes.make_worker_section_writer(models, research)
    )
    # Node role: Synthesize completed_sections into coherent article.
    workflow.add_node("synthesizer", nodes.make_synthesizer(models))

    # Edge: START -> extract scientist name.
    workflow.add_edge(START, "scientist_name_extraction")
    # Edge: Extracted name -> orchestrator plans sections.
    workflow.add_edge("scientist_name_extraction", "orchestrator")
    # Conditional edge: orchestrator uses nodes.assign_workers (Send API) to fan out
    # one worker_section_writer per planned section; all Send targets are 'worker_section_writer'.
    workflow.add_conditional_edges(
        "orchestrator", nodes.assign_workers, ["worker_section_writer"]
    )
    # Edge: each worker writes its section and appends to completed_sections (reducer.add),
    # then joins at synthesizer (after all parallel sends complete).
    workflow.add_edge("worker_section_writer", "synthesizer")
    # Edge: synthesizer outputs final article -> END.
    workflow.add_edge("synthesizer", END)

    return workflow.compile()


def run(payload: dict, models=None, callbacks=None) -> dict:
    # ``orchestrator=True`` attaches the planner LLM to the models bundle.
    models = models or build_models(orchestrator=True)
    graph = build_graph(models)
    config = {"callbacks": callbacks} if callbacks else {}
    return graph.invoke(initial_state(payload), config=config)
