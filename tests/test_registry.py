"""Tests for the workflow registry.

The registry is the single lookup table mapping workflow names to metadata and
build/run entry points.  These tests pin the *shape* of the catalogue (how many
workflows, variants and tools exist) and the lookup/error behaviour.
"""

from __future__ import annotations

from collections import Counter

import pytest

from scripting.registry import get_workflow, list_workflows, run_workflow

EXPECTED_NAMES = [
    "prompt_chaining",
    "parallelization",
    "routing",
    "orchestrator_worker",
    "prompt_chaining_feedback",
    "parallelization_feedback",
    "routing_feedback",
    "orchestrator_worker_feedback",
    "prompt_chaining_feedback_tools",
    "parallelization_feedback_tools",
    "routing_feedback_tools",
    "orchestrator_worker_feedback_tools",
]


def test_all_workflows_registered():
    """Exactly the twelve expected workflows are registered: four archetypes
    each in base / feedback / feedback+tools flavours."""
    workflows = list_workflows()
    assert len(workflows) == 12
    assert {w.name for w in workflows} == set(EXPECTED_NAMES)


def test_each_archetype_has_three_variants():
    """Every archetype (kind) appears exactly three times."""
    kinds = Counter(w.kind for w in list_workflows())
    assert kinds == Counter(
        {
            "prompt_chaining": 3,
            "parallelization": 3,
            "routing": 3,
            "orchestrator_worker": 3,
        }
    )


def test_tool_variants_marked_with_tools():
    """Only the four ``*_feedback_tools`` variants have ``has_tools=True``."""
    with_tools = [w for w in list_workflows() if w.has_tools]
    assert len(with_tools) == 4
    assert {w.name for w in with_tools} == {
        name + "_feedback_tools" for name in ["prompt_chaining", "parallelization", "routing", "orchestrator_worker"]
    }


def test_input_kinds():
    """Scientist-name inputs use ``scientist``; graph-centric flows use ``input``."""
    by_name = {w.name: w for w in list_workflows()}
    for name in ("prompt_chaining", "parallelization"):
        assert by_name[name].input_kind == "scientist"
    for name in ("routing", "orchestrator_worker"):
        assert by_name[name].input_kind == "input"
    assert by_name["prompt_chaining_feedback_tools"].input_kind == "scientist"


def test_get_workflow_unknown_raises():
    """``get_workflow`` raises ``KeyError`` for non-existent names."""
    with pytest.raises(KeyError):
        get_workflow("does_not_exist")


def test_get_workflow_known():
    """``get_workflow`` returns metadata and exposes the underlying module."""
    info = get_workflow("prompt_chaining")
    assert info.kind == "prompt_chaining"
    assert info.label == "Prompt Chaining"
    assert info.module.NAME == "prompt_chaining"


def test_run_workflow_unknown_raises():
    """``run_workflow`` rejects unknown workflow names."""
    with pytest.raises(KeyError):
        run_workflow("nope", {})