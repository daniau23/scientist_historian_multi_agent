"""Shared pytest fixtures and path setup for the whole test suite.

This module is imported automatically by pytest before any test module.  Its
only job is to guarantee that the repository root is on ``sys.path`` so that
``import scripting`` (the workflow engine) and ``import deployment`` (the
service/frontend layer) work identically regardless of the current working
directory pytest is invoked from.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repository root importable so ``import scripting`` and
# ``import deployment`` resolve no matter where pytest launches from.
# ``Path(__file__).resolve().parent.parent`` walks tests/ -> repo root; the
# ``not in sys.path`` guard keeps us from inserting a duplicate entry when
# pytest is already launched from the project root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))