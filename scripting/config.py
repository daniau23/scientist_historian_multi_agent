"""Environment / configuration loading for the scripting package."""

from __future__ import annotations

import os
from pathlib import Path

from . import models

# Guard flag: once loaded we never re-load the ``.env`` file, so later calls
# cannot override values that the environment already set.
_DOTENV_LOADED = False

# The repository root holds the ``.env`` file and the source notebooks.
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env(path: str | os.PathLike[str] | None = None) -> bool:
    """Load variables from a ``.env`` file.

    Mirrors ``load_dotenv('.env')`` used throughout the notebooks.  Returns
    ``True`` when a file was found and loaded.
    """
    global _DOTENV_LOADED

    # Imported lazily: ``python-dotenv`` is a runtime dependency only.
    from dotenv import load_dotenv

    env_path = Path(path) if path is not None else REPO_ROOT / ".env"
    if not env_path.exists():
        return False
    # ``override=False`` keeps explicitly set environment variables, which is
    # safer for deployed (Modal) environments where secrets are injected.
    loaded = load_dotenv(str(env_path), override=False)
    _DOTENV_LOADED = True
    return bool(loaded)


def google_api_key() -> str | None:
    """Return the Google Generative AI API key used by both models.

    ``models.GOOGLE_API_KEY`` is the import-time snapshot; re-reading the
    environment here also picks up values loaded from ``.env`` at build time.
    """
    return (
        models.GOOGLE_API_KEY
        or os.getenv("GOOGLE_AI_STUDIO_GENERATIVE_MODEL_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )


def langsmith_enabled() -> bool:
    """Whether LangSmith tracing has been requested through the environment."""
    return os.getenv("LANGSMITH_TRACING", "false").lower() in {"1", "true", "yes"}


def ensure_env() -> None:
    """Best-effort loading of the repository ``.env`` file (once)."""
    if not _DOTENV_LOADED:
        load_env()