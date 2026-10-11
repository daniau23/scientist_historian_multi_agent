"""Tests for the notebook -> Python converter.

The converter strips markdown, drops empty cells and removes common notebook
magics/commands (``%timeit`` etc.) so that exported notebooks become directly
runnable Python files.  It also walks directories to batch-convert all
``.ipynb`` files.
"""

from __future__ import annotations

import json

from scripting.convert import convert_all, convert_notebook, notebook_to_source


def _make_notebook():
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "source": ["# A title that must be skipped"],
            },
            {
                "cell_type": "code",
                "source": [
                    "import os\n",
                    "print(os.environ)\n",
                    "%timeit pass\n",
                ],
            },
            {
                "cell_type": "code",
                "source": ["!pip install foo\n", "x = 1\n"],
            },
            {"cell_type": "code", "source": ["\n"]},
        ]
    }


def test_notebook_to_source_strips_magics_and_skips_markdown():
    """Markdown cells are omitted; magic lines (``%timeit``) and shell commands
    (``!pip``) are removed; non-empty code cells are retained and annotated
    with ``# cell[N]`` markers."""
    source = notebook_to_source(_make_notebook())
    assert "# A title" not in source
    assert "%timeit" not in source
    assert "!pip" not in source
    assert "import os" in source
    assert "x = 1" in source
    # Two non-empty code cells (markdown + empty code cells are skipped).
    assert source.count("# cell[") == 2


def test_convert_notebook_writes_py_file(tmp_path):
    """Converting a single ``.ipynb`` file writes an equivalently named ``.py``
    to the requested output directory (creating it if needed)."""
    nb = _make_notebook()
    ipynb = tmp_path / "sample.ipynb"
    ipynb.write_text(json.dumps(nb), encoding="utf-8")

    out = convert_notebook(ipynb, tmp_path / "out")
    assert out.name == "sample.py"
    text = out.read_text(encoding="utf-8")
    assert "import os" in text
    assert "x = 1" in text


def test_convert_all_handles_directory(tmp_path):
    """``convert_all`` batches all notebooks under a root and writes their
    converted scripts into ``output_dir``, returning their paths."""
    for name in ("alpha.ipynb", "beta.ipynb"):
        (tmp_path / name).write_text(
            json.dumps(_make_notebook()), encoding="utf-8"
        )
    out_dir = tmp_path / "out"
    outputs = convert_all(tmp_path, out_dir)
    assert {p.name for p in outputs} == {"alpha.py", "beta.py"}
    for path in outputs:
        assert path.exists()