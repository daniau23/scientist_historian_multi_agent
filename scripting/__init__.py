"""Scripting package for the scientist multi-agent workflows.

This package contains converted (notebook -> python) versions of every
workflow found in this repository, a registry to discover them and a CLI to
run them.
"""

import warnings


def _import_langgraph_quietly() -> None:
    """Import the LangGraph stack with its import-time warnings suppressed.

    A module on LangGraph's import path (via ``langsmith``) calls a bare
    ``warnings.simplefilter("default")`` while importing, which resets the
    global filter list and lets LangGraph's own pending-deprecation warning
    (``langgraph.checkpoint...allowed_objects``) slip through.  Temporarily
    neutralising the filter-mutating helpers for the duration of the import
    keeps our ``ignore`` filter authoritative; the originals are restored
    immediately afterwards.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _filterwarnings, _simplefilter = (
            warnings.filterwarnings,
            warnings.simplefilter,
        )
        warnings.filterwarnings = lambda *a, **k: None
        warnings.simplefilter = lambda *a, **k: None
        try:
            import langgraph.graph  # noqa: F401
        finally:
            warnings.filterwarnings = _filterwarnings
            warnings.simplefilter = _simplefilter


# Import once, quietly, so the rest of the package (and the CLI) is silent.
_import_langgraph_quietly()

# Keep LangChain/LangGraph deprecation noise suppressed for the rest of the run.
warnings.filterwarnings("ignore", message=r".*allowed_objects.*")
try:
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

    warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)
except Exception:  # pragma: no cover - defensive against version drift
    pass

# Re-export the registry helpers so consumers only need ``import scripting``.
from .registry import WORKFLOWS, get_workflow, list_workflows, run_workflow

__all__ = [
    "WORKFLOWS",
    "get_workflow",
    "list_workflows",
    "run_workflow",
]