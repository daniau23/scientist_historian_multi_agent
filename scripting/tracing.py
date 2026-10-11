"""LangSmith tracing helpers shared by the CLI and the Modal backend."""

from __future__ import annotations

import os
from typing import Any, Callable

from . import config


def configure_langsmith(project: str | None = None) -> bool:
    """Ensure LangSmith env vars are present.

    Returns ``True`` when tracing looks enabled and an API key is available.
    """
    # Load ``.env`` so the local CLI behaves like the notebooks.
    config.ensure_env()
    if project:
        os.environ.setdefault("LANGSMITH_PROJECT", project)

    # Tracing must be explicitly switched on AND have a key to be usable.
    api_key = os.getenv("LANGSMITH_API_KEY")
    if api_key and os.getenv("LANGSMITH_TRACING", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        return True
    return False


def traceable(*dargs: Any, **dkwargs: Any) -> Callable:
    """Thin wrapper around ``langsmith.traceable``.

    Falls back to a no-op decorator when LangSmith is unavailable so callers do
    not need to guard every call site.
    """
    try:
        from langsmith import traceable as _traceable

        return _traceable(*dargs, **dkwargs)
    except Exception:  # pragma: no cover - langsmith optional
        # Support both ``@traceable`` (bare) and ``@traceable(...)`` forms.
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return dargs[0]

        # No-op decorator: the decorated function just passes through.
        def decorator(func: Callable) -> Callable:
            return func

        return decorator