"""Command line interface for running the converted workflows.

Examples
--------
List available workflows::

    python -m scripting list

Run a workflow non-interactively (prints only the final output)::

    python -m scripting run prompt_chaining --scientist "Isaac Newton"
    python -m scripting run routing --input "Give me an article about Marie Curie"

Convert the notebooks back to Python files::

    python -m scripting convert
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from . import config, registry


def _json_safe(value: Any) -> Any:
    """Recursively make any value JSON-serialisable (handles pydantic models)."""
    # Dictionaries may contain pydantic model values -> recurse into each key.
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    # pydantic v2 models expose ``model_dump()`` for a plain-dict view.
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


# Keys (in priority order) that hold the workflow's final rendered output.
_FINAL_OUTPUT_KEYS = ("refined_article", "article", "final_article")


def _final_output(result: Dict[str, Any]) -> Optional[str]:
    """Return the workflow's final rendered text, or ``None`` if absent."""
    for key in _FINAL_OUTPUT_KEYS:
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def format_result(
    result: Dict[str, Any],
    info: Optional[registry.WorkflowInfo] = None,
) -> str:
    """Human readable rendering of a workflow result.

    Only the final output is shown.  Before it, workflow-specific summary
    lines are printed:
    - orchestrator-worker workflows: the number of worker sections;
    - feedback workflows: the number of feedback rounds the evaluator ran.

    ``--json`` remains available for the full, machine-readable state.
    """
    lines: List[str] = []
    # Orchestrator-worker: report how many parallel workers produced sections.
    if info is not None and info.kind == "orchestrator_worker":
        completed = result.get("completed_sections")
        worker_count = len(completed) if isinstance(completed, (list, tuple)) else 0
        lines.append(f"Workers: {worker_count}")

    # Feedback variants: report how many evaluator retries happened.
    if info is not None and info.has_feedback:
        lines.append(f"Feedback rounds: {result.get('feedback_count', 0)}")

    final = _final_output(result)
    if final is not None:
        # Blank line separates the summary header from the article body.
        if lines:
            lines.append("")
        lines.append(final)
        return "\n".join(lines)

    # Fallback (partial / error states without a final article): render every
    # remaining key so nothing important is silently hidden.
    parts: List[str] = []
    for key, value in result.items():
        header = key.upper()
        parts.append(header)
        if isinstance(value, str):
            parts.append(value.strip())
        else:
            parts.append(str(value))
    return "\n\n".join(parts)


def build_payload(info: registry.WorkflowInfo, args: argparse.Namespace) -> Dict[str, Any]:
    """Build the workflow payload from CLI args (prompting when omitted)."""
    # Scientist workflows take a name; the rest take free-text input.
    if info.input_kind == "scientist":
        scientist = args.scientist
        if not scientist:
            scientist = input("Scientist name: ").strip()
        return {"scientist": scientist}
    user_input = args.input
    if not user_input:
        user_input = input("Input: ").strip()
    return {"input": user_input}


def cmd_list(_: argparse.Namespace) -> int:
    """Implement ``cli list``: print every workflow with its feature tags."""
    for info in registry.list_workflows():
        # Collect feature tags like [feedback/tools] for at-a-glance filtering.
        tags = []
        if info.has_feedback:
            tags.append("feedback")
        if info.has_tools:
            tags.append("tools")
        tag_str = f" [{'/'.join(tags)}]" if tags else ""
        print(f"{info.name}{tag_str}\n    {info.description}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Implement ``cli run``: execute one workflow and render its output."""
    info = registry.get_workflow(args.workflow)

    # Optional LangSmith tracing: load env and warn if it is not enabled.
    callbacks = None
    if args.trace:
        config.ensure_env()
        if not config.langsmith_enabled():
            print(
                "Warning: --trace requested but LANGSMITH_TRACING is not enabled.",
                file=sys.stderr,
            )

    payload = build_payload(info, args)
    result = registry.run_workflow(args.workflow, payload, callbacks=callbacks)

    # Emit raw JSON instead of the pretty text when the user asked for it.
    if args.json:
        print(json.dumps(_json_safe(result), indent=2))
    else:
        print(format_result(result, info))
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    """Implement ``cli convert``: notebooks -> ``.py`` in ``converted/``."""
    from .convert import convert_all

    outputs = convert_all(args.root, args.output)
    # Print the written paths so users can find the generated files.
    for path in outputs:
        print(path)
    print(f"Converted {len(outputs)} notebook(s).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Assemble the argument parser with the three subcommands."""
    parser = argparse.ArgumentParser(
        prog="scripting",
        description="Run the converted scientist multi-agent workflows.",
    )
    # ``required=True`` forces the user to pick a subcommand explicitly.
    sub = parser.add_subparsers(dest="command", required=True)

    # Subcommand: list ------------------------------------------------
    list_parser = sub.add_parser("list", help="List available workflows")
    list_parser.set_defaults(func=cmd_list)

    # Subcommand: run -------------------------------------------------
    run_parser = sub.add_parser("run", help="Run a workflow")
    run_parser.add_argument("workflow", help="Workflow name (see 'list')")
    run_parser.add_argument(
        "--scientist", help="Scientist name for scientist workflows"
    )
    run_parser.add_argument(
        "--input", help="Free-text input for routing/orchestrator workflows"
    )
    run_parser.add_argument("--json", action="store_true", help="Emit JSON output")
    run_parser.add_argument("--trace", action="store_true", help="Enable LangSmith tracing")
    run_parser.set_defaults(func=cmd_run)

    # Subcommand: convert ---------------------------------------------
    convert_parser = sub.add_parser("convert", help="Convert notebooks to .py files")
    convert_parser.add_argument(
        "--root", default=None, help="Directory containing notebooks"
    )
    convert_parser.add_argument("--output", default=None, help="Output directory")
    convert_parser.set_defaults(func=cmd_convert)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point: parse args and dispatch to the matching subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover - only reachable via ``python -m scripting``
    raise SystemExit(main())