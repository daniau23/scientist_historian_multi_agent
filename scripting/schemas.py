"""Pydantic models and LangGraph state definitions shared by all workflows.

These mirror the ``Pydantic: Validation Logic`` cells of the notebooks.
"""

from __future__ import annotations

import re
from operator import add
from typing import Annotated, List, Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Pydantic models: validate / structure every LLM input and output.
# ---------------------------------------------------------------------------


class ScientistInput(BaseModel):
    """Validated external input: the name of a scientist."""

    scientist: str = Field(
        min_length=3, max_length=100, description="The name of the scientist"
    )

    @field_validator("scientist")
    @classmethod
    def validate_scientist(cls, value: str) -> str:
        # Normalise whitespace and reject names that contain no letters.
        value = value.strip()
        if not re.search("[a-zA-Z]", value):
            raise ValueError("Scientist name must contain letters.")
        return value.strip()


class ScientistHistoryOutput(BaseModel):
    """Structured output of the history generation node."""

    history: str = Field(description="The history of the scientist")


class ScientistAchievementsOutput(BaseModel):
    """Structured output of the achievements generation node."""

    achievements: list[str] = Field(
        description="Five major achievements of the scientist"
    )


class ScientistArticleOutput(BaseModel):
    """Structured output of the article generation node."""

    article: str = Field(description="Detailed article about the scientist")


class RefinedArticleOutput(BaseModel):
    """Structured output of the refinement node."""

    refined_article: str = Field(
        description="Refined article about the scientist under 1000 words"
    )


class EvaluatorCritic(BaseModel):
    """Structured verdict used by the evaluator/reflection loop.

    ``status`` is constrained to a literal so the conditional edge can switch
    on exactly ``ACCEPTED`` / ``REJECTED``.
    """

    status: Literal["ACCEPTED", "REJECTED"] = Field(
        description="Decide if the Refined Article is acceptable or not"
    )
    feedback: str = Field(
        description=(
            "Give a constructive reason why the article is not good enough "
            "with points for improvement"
        )
    )


class Route(BaseModel):
    """Structured routing decision produced by the router LLM."""

    step: Literal["history", "accomplishment", "article"] = Field(
        description="The next step of the workflow process."
    )


class Section(BaseModel):
    """A single planned section with its description (orchestrator)."""

    title: str = Field(
        ...,
        description="Title of the section, e.g., 'Early Life', 'Major Achievements'.",
    )
    description: str = Field(
        ..., description="Brief overview of what should be covered in this section."
    )


class OutlinePlan(BaseModel):
    """The full set of sections planned by the orchestrator LLM."""

    sections: List[Section] = Field(
        ..., description="Dynamically planned sections for the scientist report."
    )


# ---------------------------------------------------------------------------
# Graph states: TypedDicts describing the values each node can read/write.
# ---------------------------------------------------------------------------


class ScientistState(TypedDict):
    """State for prompt chaining / parallelization workflows."""

    scientist: str
    history: NotRequired[str]
    achievements: NotRequired[list[str]]
    article: NotRequired[str]
    refined_article: NotRequired[str]


class FeedbackState(ScientistState):
    """State with the evaluator feedback loop fields."""

    feedback: NotRequired[str]
    accepted_rejected: NotRequired[str]
    feedback_count: NotRequired[int]


class RoutingState(TypedDict):
    """State for the routing workflow."""

    input: str
    decision: NotRequired[str]
    scientist: NotRequired[str]
    history: NotRequired[str]
    achievements: NotRequired[list[str]]
    article: NotRequired[str]
    refined_article: NotRequired[str]
    feedback: NotRequired[str]
    accepted_rejected: NotRequired[str]
    feedback_count: NotRequired[int]


class OrchestratorState(TypedDict):
    """State for the orchestrator-worker workflow."""

    input: str
    scientist: NotRequired[str]
    sections: NotRequired[List[Section]]
    # ``Annotated[List[str], add]`` registers ``operator.add`` as the reducer:
    # parallel workers append their section instead of overwriting each other.
    completed_sections: Annotated[List[str], add]
    article: NotRequired[str]
    refined_article: NotRequired[str]


class OrchestratorFeedbackState(OrchestratorState):
    """Orchestrator-worker state with the evaluator feedback loop fields."""

    feedback: NotRequired[str]
    accepted_rejected: NotRequired[str]
    feedback_count: NotRequired[int]


class WorkerState(TypedDict):
    """Individual worker state passed through the ``Send`` API."""

    scientist: str
    section: Section