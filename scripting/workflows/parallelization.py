"""Converted from ``parallelization.ipynb``.

Parallel architecture: history and achievements are generated concurrently from START,
then their outputs converge at the article node before refinement. This pattern
demonstrates fan-out from START and join (convergence) at a downstream node.

ASCII flow diagram (actual node names from build_graph):
    START --┬--> generate_history ----------┐
           └--> generate_achievements -----+--> generate_article -> generate_refined_article -> END

Nodes:
- generate_history: Produces scientist's biographical history (runs in parallel).
- generate_achievements: Produces scientist's major achievements (runs in parallel).
- generate_article: Composes article using both history and achievements (waits for both branches).
- generate_refined_article: Refines the composed article for clarity/coherence.

"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .. import nodes
from ..llms import build_models
from ..schemas import ScientistInput, ScientistState

NAME = "parallelization"
KIND = "parallelization"
HAS_FEEDBACK = False
HAS_TOOLS = False
INPUT_KIND = "scientist"
DESCRIPTION = "History and achievements generated in parallel, then article + refine."


def initial_state(payload: dict) -> dict:
    """Initialize state from payload. Expects 'scientist' (str); validates via
    ScientistInput and sets state['scientist']=validated.scientist."""
    validated = ScientistInput.model_validate(
        {"scientist": payload.get("scientist", "")}
    )
    return {"scientist": validated.scientist}


def build_graph(models=None):
    models = models or build_models()

    workflow = StateGraph(ScientistState)
    # Fan out: history and achievements run in parallel from START and
    # converge on the article node.

    # Node role: Generate scientist history (parallel branch from START).
    workflow.add_node("generate_history", nodes.make_generate_history(models))
    # Node role: Generate scientist achievements (parallel branch from START).
    workflow.add_node(
        "generate_achievements", nodes.make_generate_achievements(models)
    )
    # Node role: Compose article using outputs from both parallel branches.
    workflow.add_node("generate_article", nodes.make_generate_article(models))
    # Node role: Refine composed article for clarity/coherence.
    workflow.add_node(
        "generate_refined_article", nodes.make_generate_refined_article(models)
    )

    # Edges: fan-out from START to parallel branches (concurrent execution).
    workflow.add_edge(START, "generate_history")
    workflow.add_edge(START, "generate_achievements")
    # Edges: join both parallel branches at generate_article (waits for both).
    workflow.add_edge("generate_history", "generate_article")
    workflow.add_edge("generate_achievements", "generate_article")
    # Edge: article feeds refinement.
    workflow.add_edge("generate_article", "generate_refined_article")
    # Edge: refined article exits.
    workflow.add_edge("generate_refined_article", END)

    return workflow.compile()


def run(payload: dict, models=None, callbacks=None) -> dict:
    # Tests inject fake models here to avoid any network / API-key dependency.
    models = models or build_models()
    graph = build_graph(models)
    config = {"callbacks": callbacks} if callbacks else {}
    return graph.invoke(initial_state(payload), config=config)
