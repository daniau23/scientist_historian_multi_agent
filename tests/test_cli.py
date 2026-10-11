"""Tests for the CLI surface (list / convert / payload building).

Covers argument parsing, the ``list`` and ``convert`` subcommands, payload
construction by input kind, formatting of the final result for different
workflow variants, and error handling when an unknown workflow is requested.
"""

from __future__ import annotations

import argparse

from scripting.cli import build_parser, build_payload, format_result, main
from scripting.registry import get_workflow


def _result(refined: str, **extra) -> dict:
    """Minimal workflow-state dict with the final article present.

    Used to exercise ``format_result`` with just enough fields to cover all
    branches (base / feedback / orchestrator variants).  Extra keys (e.g.
    ``feedback_count``, ``completed_sections``) are passed through explicitly.
    """
    base = {
        "scientist": "Isaac Newton",
        "history": "history text",
        "achievements": ["a1"],
        "article": "full article",
        "refined_article": refined,
    }
    base.update(extra)
    return base


def test_list_runs_and_prints(capsys):
    """Running ``list`` prints the registered workflows and exits successfully."""
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "prompt_chaining" in out
    assert "orchestrator_worker_feedback_tools" in out


def test_convert_subcommand(tmp_path, capsys):
    """``convert`` subcommand writes the converted Python script to output dir."""
    source = tmp_path / "demo.ipynb"
    source.write_text(
        '{"cells": [{"cell_type": "code", "source": ["print(1)"]}]}',
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    assert main(["convert", "--root", str(tmp_path), "--output", str(out_dir)]) == 0
    assert (out_dir / "demo.py").exists()


def test_build_payload_scientist_workflow():
    """For scientist-kind workflows the CLI prefers the ``scientist`` field."""
    parser = build_parser()
    args = argparse.Namespace(scientist="Marie Curie", input=None)
    payload = build_payload(get_workflow("prompt_chaining"), args)
    assert payload == {"scientist": "Marie Curie"}


def test_build_payload_input_workflow():
    """For input-kind workflows the payload must carry ``input``."""
    parser = build_parser()
    args = argparse.Namespace(scientist=None, input="give me an article")
    payload = build_payload(get_workflow("routing"), args)
    assert payload == {"input": "give me an article"}


def test_run_unknown_workflow_exits_nonzero(capsys):
    """Calling ``main`` with an unknown workflow name propagates ``KeyError``
    (the CLI layer lets registry lookups bubble up)."""
    import pytest

    with pytest.raises(KeyError):
        main(["run", "not_a_workflow"])


def test_format_result_base_workflow_shows_only_final_output():
    """A plain (non-feedback, non-orchestrator) workflow shows just the article
    text, with no extra metadata lines."""
    text = format_result(_result("final markdown"), get_workflow("prompt_chaining"))
    assert "final markdown" in text
    assert "Workers:" not in text
    assert "Feedback rounds:" not in text


def test_format_result_feedback_workflow_shows_rounds():
    """Feedback variants prepend the number of evaluator rounds before the
    final refined article."""
    info = get_workflow("prompt_chaining_feedback")
    text = format_result(_result("v2", feedback_count=2), info)
    assert "Feedback rounds: 2" in text
    assert text.strip().endswith("v2")


def test_format_result_orchestrator_shows_workers_and_rounds():
    """Orchestrator-worker variants report workers, then feedback, then the
    joined final output (in that order)."""
    info = get_workflow("orchestrator_worker_feedback")
    completed = ["### One\n\nbody one", "### Two\n\nbody two"]
    text = format_result(
        _result("joined article", completed_sections=completed, feedback_count=1),
        info,
    )
    assert text.index("Workers: 2") < text.index("Feedback rounds: 1")
    assert text.index("Feedback rounds: 1") < text.index("joined article")


def test_format_result_without_info_renders_final_key():
    """Formatting falls back to the common ``refined_article`` key when no
    workflow metadata is supplied."""
    text = format_result(_result("solo"))
    assert text.strip().endswith("solo")