"""Workflow registry: discover and run converted workflows by name."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from .workflows import MODULES


@dataclass(frozen=True)
class WorkflowInfo:
    """Metadata describing a single workflow."""

    # Canonical name (also the registry key), e.g. ``prompt_chaining``.
    name: str
    # Architecture family: prompt_chaining | parallelization | routing |
    # orchestrator_worker.  Several workflows share the same kind.
    kind: str
    has_feedback: bool
    has_tools: bool
    # ``scientist`` workflows take a name; ``input`` workflows take free text.
    input_kind: str
    description: str
    # The workflow module exposing ``build_graph`` / ``run`` / ``initial_state``.
    module: Any

    @property
    def label(self) -> str:
        """Human-readable label, e.g. ``Prompt Chaining``."""
        return self.name.replace("_", " ").title()

    def as_dict(self) -> Dict[str, Any]:
        """JSON-friendly view of the metadata (used by callers / tests)."""
        return {
            "name": self.name,
            "kind": self.kind,
            "has_feedback": self.has_feedback,
            "has_tools": self.has_tools,
            "input_kind": self.input_kind,
            "description": self.description,
        }


# Build the lookup table once from every module registered in
# ``scripting.workflows``.  Each module declares its metadata as module-level
# constants, which keeps the registry data-driven (no duplication).
WORKFLOWS: Dict[str, WorkflowInfo] = {
    module.NAME: WorkflowInfo(
        name=module.NAME,
        kind=module.KIND,
        has_feedback=module.HAS_FEEDBACK,
        has_tools=module.HAS_TOOLS,
        input_kind=module.INPUT_KIND,
        description=module.DESCRIPTION,
        module=module,
    )
    for module in MODULES
}


def list_workflows() -> List[WorkflowInfo]:
    """Return every registered workflow in registration order."""
    return list(WORKFLOWS.values())


def get_workflow(name: str) -> WorkflowInfo:
    """Look up a workflow by name.

    Raises ``KeyError`` with a helpful message when the name is unknown.
    """
    try:
        return WORKFLOWS[name]
    except KeyError:
        available = ", ".join(sorted(WORKFLOWS))
        raise KeyError(
            f"Unknown workflow '{name}'. Available workflows: {available}"
        ) from None


def run_workflow(
    name: str,
    payload: Dict[str, Any],
    models: Optional[Any] = None,
    callbacks: Optional[Iterable] = None,
) -> Dict[str, Any]:
    """Run a registered workflow with the supplied payload.

    ``models`` is injectable so the CLI and the backend can pass a built bundle
    or (in tests) fakes; defaulting to ``None`` lets each workflow build its own.
    """
    info = get_workflow(name)
    return info.module.run(payload, models=models, callbacks=callbacks)