"""Registry of all converted workflows."""

from __future__ import annotations

# Import every workflow module up-front so the registry can enumerate them
# without importing on demand (keeps the public API explicit and discoverable).
from . import (
    orchestrator_worker,
    orchestrator_worker_feedback,
    orchestrator_worker_feedback_tools,
    parallelization,
    parallelization_feedback,
    parallelization_feedback_tools,
    prompt_chaining,
    prompt_chaining_feedback,
    prompt_chaining_feedback_tools,
    routing,
    routing_feedback,
    routing_feedback_tools,
)

# Registration order also defines the display order: the base architectures
# first, then their feedback and feedback+tools variants.
MODULES = [
    prompt_chaining,
    parallelization,
    routing,
    orchestrator_worker,
    prompt_chaining_feedback,
    parallelization_feedback,
    routing_feedback,
    orchestrator_worker_feedback,
    prompt_chaining_feedback_tools,
    parallelization_feedback_tools,
    routing_feedback_tools,
    orchestrator_worker_feedback_tools,
]

__all__ = ["MODULES"]