"""End-to-end workflow tests using fake, deterministic LLMs.

Builds each workflow graph with ``tests.fakes.make_fake_models`` and exercises
the full ``run`` paths offline (no API keys, no network).  Covers node
existence by archetype, routing branch selection, feedback loop termination/
acceptance, orchestrator fan-out, and initial state validation.
"""

from __future__ import annotations

import pytest

from scripting.workflows import MODULES
from tests.fakes import make_fake_models

SCIENTIST = "Isaac Newton"
INPUT = "Write an article about Marie Curie"

# Which nodes every architecture must register at the top level.  Feedback
# nodes (``evaluate_feedback_llm``, ``refine_article_with_feedback``) are
# checked separately because routing keeps them inside the ``article_workflow``
# subgraph (so they do not appear as top-level nodes).  The ``TOP_LEVEL_ONLY``
# set captures those routing feedback variants where top-level inspection is
# intentionally limited to the subgraph node itself.
NODE_EXPECTATIONS = {
    "prompt_chaining": {
        "generate_history",
        "generate_achievements",
        "generate_article",
        "generate_refined_article",
    },
    "parallelization": {
        "generate_history",
        "generate_achievements",
        "generate_article",
        "generate_refined_article",
    },
    "routing": {
        "scientist_name_extraction",
        "generate_history",
        "generate_achievements",
        "article_workflow",
        "llm_routing",
    },
    "orchestrator_worker": {
        "scientist_name_extraction",
        "orchestrator",
        "worker_section_writer",
        "synthesizer",
    },
}

FEEDBACK_NODES = {"evaluate_feedback_llm", "refine_article_with_feedback"}

# Routing feedback nodes live inside the ``article_workflow`` subgraph, so a
# top-level node inspection only sees ``article_workflow`` itself (and for the
# base routing graph, feedback is not present).  Treating these three names
# specially prevents requiring feedback nodes to exist at the top level.
TOP_LEVEL_ONLY = {"routing", "routing_feedback", "routing_feedback_tools"}


def get_module(name):
    for module in MODULES:
        if module.NAME == name:
            return module
    raise AssertionError(f"module {name} not found")


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.NAME)
def test_build_graph_has_expected_nodes(module):
    """Each graph registers the core nodes for its archetype; feedback nodes
    are expected at top level for non-routing feedback variants."""
    models = make_fake_models(
        route_step="article", eval_status="ACCEPTED"
    )
    graph = module.build_graph(models)
    nodes = set(graph.get_graph().nodes)
    expected = set(NODE_EXPECTATIONS[module.KIND])
    if module.HAS_FEEDBACK and module.NAME not in TOP_LEVEL_ONLY:
        expected |= FEEDBACK_NODES
    assert expected <= nodes


def _payload_for(module):
    if module.INPUT_KIND == "scientist":
        return {"scientist": SCIENTIST}
    return {"input": INPUT}


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.NAME)
def test_run_uses_fake_models(module):
    """Running with deterministic fakes always produces a ``refined_article``
    (and never makes external calls)."""
    models = make_fake_models(route_step="article", eval_status="ACCEPTED")
    result = module.run(_payload_for(module), models=models)
    assert result["refined_article"], f"missing refined_article for {module.NAME}"


@pytest.mark.parametrize("route_step", ["history", "accomplishment", "article"])
def test_routing_decision_paths(route_step):
    """The router LLM's decision deterministically branches the graph:
    ``history`` stops after generating history; ``accomplishment`` stops after
    achievements; ``article`` runs through to a refined article."""
    metrics_models = make_fake_models(route_step=route_step)
    module = get_module("routing")
    result = module.run({"input": INPUT}, models=metrics_models)
    if route_step == "history":
        assert result.get("history") == "Fake history."
        assert "refined_article" not in result
    elif route_step == "accomplishment":
        assert result.get("achievements")
    else:
        assert result.get("refined_article")


def test_routing_invalid_decision_fails():
    """A malformed router decision (not one of the allowed literals) raises
    ``ValueError`` so the graph fails fast."""
    from scripting.workflows.routing import route_decision

    with pytest.raises(ValueError):
        route_decision({"decision": "bogus"})


@pytest.mark.parametrize(
    "name",
    # Every chain/parallel/routing feedback variant (they share the loop).
    ["prompt_chaining_feedback", "parallelization_feedback", "routing_feedback"],
)
def test_feedback_loop_terminates_after_max_retries(name):
    """With a permanently REJECTED critic, the feedback loop stops at
    ``MAX_FEEDBACK_RETRIES`` (3) and still returns a ``refined_article``
    (liveness/termination guard)."""
    module = get_module(name)
    payload = _payload_for(module)
    models = make_fake_models(eval_status="REJECTED", route_step="article")
    result = module.run(payload, models=models)
    assert result["feedback_count"] == 3
    assert result["refined_article"]


def test_feedback_loop_accepts_immediately():
    """An evaluator returning ACCEPTED on the first iteration exits the loop
    with ``feedback_count == 0``."""
    module = get_module("prompt_chaining_feedback")
    models = make_fake_models(eval_status="ACCEPTED")
    result = module.run({"scientist": SCIENTIST}, models=models)
    assert result["feedback_count"] == 0
    assert result["accepted_rejected"] == "ACCEPTED"


def test_orchestrator_runs_parallel_workers():
    """The orchestrator splits work into sections and workers run in parallel
    (appending their outputs), producing ``completed_sections`` and a final
    synthesized/refined article."""
    module = get_module("orchestrator_worker")
    result = module.run({"input": INPUT}, models=make_fake_models())
    assert result["completed_sections"]
    assert "History" in result["article"]
    assert result["refined_article"]


def test_initial_state_validation_rejects_bad_scientist():
    """Initial state validation catches a malformed scientist name early
    (using the same Pydantic constraints)."""
    module = get_module("prompt_chaining")
    with pytest.raises(Exception):
        module.initial_state({"scientist": "12"})