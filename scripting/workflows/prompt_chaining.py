"""Converted from ``prompt_chaining.ipynb``.

Sequential pipeline architecture for generating a biographical article about a scientist.
This is a linear (chain) workflow that builds up the article step by step: first
gathers the scientist's history, then their achievements, composes an article,
and finally refines it.

ASCII flow diagram (actual node names from build_graph):
    START -> generate_history -> generate_achievements -> generate_article -> generate_refined_article -> END

Nodes:
- generate_history: Produces the scientist's biographical history.
- generate_achievements: Produces the scientist's major achievements.
- generate_article: Composes a full article using history and achievements.
- generate_refined_article: Refines the composed article for clarity/coherence.

"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .. import nodes
from ..llms import build_models
from ..schemas import ScientistInput, ScientistState

NAME = "prompt_chaining"
KIND = "prompt_chaining"
HAS_FEEDBACK = False
HAS_TOOLS = False
INPUT_KIND = "scientist"
DESCRIPTION = "Sequential chain of history, achievements, article and refinement."


def initial_state(payload: dict) -> dict:
    """Initialize state from payload. Expects payload with 'scientist' (str);
    validates via ScientistInput.model_validate({'scientist': ...}) and sets
    state['scientist'] = validated.scientist. Returns state dict matching
    ScientistState shape expectations."""
    validated = ScientistInput.model_validate(
        {"scientist": payload.get("scientist", "")}
    )
    return {"scientist": validated.scientist}


def build_graph(models=None):
    models = models or build_models()

    workflow = StateGraph(ScientistState)
    # Linear pipeline: history -> achievements -> article -> refinement.

    # Node role: Generate the scientist's biographical history.
    workflow.add_node("generate_history", nodes.make_generate_history(models))
    # Node role: Generate the scientist's major achievements.
    workflow.add_node(
        "generate_achievements", nodes.make_generate_achievements(models)
    )
    # Node role: Compose a full article using history and achievements.
    workflow.add_node("generate_article", nodes.make_generate_article(models))
    # Node role: Refine the composed article for clarity and coherence.
    workflow.add_node(
        "generate_refined_article", nodes.make_generate_refined_article(models)
    )

    # Edge: Entry point to first step of linear chain.
    workflow.add_edge(START, "generate_history")
    # Edge: History output feeds achievements generation (sequential data flow).
    workflow.add_edge("generate_history", "generate_achievements")
    # Edge: Achievements output feeds article composition (sequential).
    workflow.add_edge("generate_achievements", "generate_article")
    # Edge: Article output feeds refinement step.
    workflow.add_edge("generate_article", "generate_refined_article")
    # Edge: Final refined article exits the graph.
    workflow.add_edge("generate_refined_article", END)

    # Compile the graph so it can be invoked like a normal Runnable.
    return workflow.compile()


def run(payload: dict, models=None, callbacks=None) -> dict:
    # Allow tests to inject fake models instead of building real LLM chains.
    models = models or build_models()
    graph = build_graph(models)
    # Optional LangChain callbacks (e.g. tracing) are forwarded to the invoke.
    config = {"callbacks": callbacks} if callbacks else {}
    return graph.invoke(initial_state(payload), config=config)
