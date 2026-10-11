"""Tests for the shared pydantic schemas / validation logic.

These schemas are the contract between the outside world and every workflow:
they validate user input (scientist names) and constrain the structured output
of each LLM so the graph edges can switch on exact literal values.  The tests
here pin down that validation behaviour independently of the graphs.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from scripting.schemas import (
    EvaluatorCritic,
    OutlinePlan,
    Route,
    ScientistInput,
    Section,
)


def test_scientist_input_valid():
    """A well-formed name is accepted and surrounding whitespace is stripped."""
    result = ScientistInput.model_validate({"scientist": "  Isaac Newton  "})
    assert result.scientist == "Isaac Newton"


@pytest.mark.parametrize(
    "value",
    [
        "ab",
        "",
        "  ",
        "12345",  # no letters
        "a" * 101,
    ],
)
def test_scientist_input_invalid(value):
    """Each malformed name is rejected: too short/empty/whitespace-only,
    digit-only (no letters), and longer than the 100-char max_length."""
    with pytest.raises(ValidationError):
        ScientistInput.model_validate({"scientist": value})


def test_route_status_validation():
    """``Route.step`` only allows the literal branches the graph understands."""
    assert Route(step="history").step == "history"
    assert Route(step="article").step == "article"
    with pytest.raises(ValidationError):
        Route(step="unknown")


def test_evaluator_critic_status_validation():
    """The evaluator verdict is a two-value literal; anything else is invalid."""
    assert EvaluatorCritic(status="ACCEPTED", feedback="ok").status == "ACCEPTED"
    with pytest.raises(ValidationError):
        EvaluatorCritic(status="MAYBE", feedback="ok")


def test_outline_plan_round_trip():
    """An outline keeps its ordered list of sections intact (order matters to
    the orchestrator/worker fan-out)."""
    plan = OutlinePlan(
        sections=[
            Section(title="History", description="Early life."),
            Section(title="Achievements", description="Major work."),
        ]
    )
    assert len(plan.sections) == 2
    assert plan.sections[1].title == "Achievements"