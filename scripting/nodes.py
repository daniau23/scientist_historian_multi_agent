"""Reusable LangGraph node factories.

Building nodes as factories that close over a ``Models`` bundle lets tests
inject fake LLMs and lets every workflow share the same well-tested logic.
"""

from __future__ import annotations

from typing import Callable, Optional

from .prompts import (
    ACHIEVEMENTS_PROMPT,
    ACHIEVEMENTS_RESEARCH_PROMPT,
    ARTICLE_PROMPT,
    FEEDBACK_PROMPT,
    HISTORY_PROMPT,
    HISTORY_RESEARCH_PROMPT,
    REFINE_PROMPT,
    SCIENTIST_NAME_EXTRACTION_PROMPT,
    WORKER_SECTION_PROMPT,
    WORKER_SECTION_RESEARCH_PROMPT,
)
from .schemas import FeedbackState, ScientistState, WorkerState


# ---------------------------------------------------------------------------
# Generation nodes
# ---------------------------------------------------------------------------


def make_generate_history(
    models, research: Optional[Callable[[str], str]] = None
) -> Callable[[ScientistState], dict]:
    def generate_history(state: ScientistState) -> dict:
        scientist = state.get("scientist", "")
        if research is not None:
            # Tools variant: gather web research, pass it into the prompt.
            research_data = research(
                f"Research key biographical history of {scientist}."
            )
            prompt = HISTORY_RESEARCH_PROMPT
            info = {"scientist": scientist, "research": research_data}
        else:
            # Plain variant: no external research is needed.
            prompt = HISTORY_PROMPT
            info = {"scientist": scientist}

        # ``prompt | llm`` composes into a chain; the structured LLM returns
        # a ``ScientistHistoryOutput`` so we can read ``response.history``.
        response = (prompt | models.history_llm).invoke(info)
        return {"history": response.history}

    return generate_history


def make_generate_achievements(
    models, research: Optional[Callable[[str], str]] = None
) -> Callable[[ScientistState], dict]:
    def generate_achievements(state: ScientistState) -> dict:
        scientist = state.get("scientist", "")
        if research is not None:
            research_data = research(
                f"Research the 5 most important achievements of {scientist}."
            )
            prompt = ACHIEVEMENTS_RESEARCH_PROMPT
            info = {"scientist": scientist, "research": research_data}
        else:
            prompt = ACHIEVEMENTS_PROMPT
            info = {"scientist": scientist}

        # The structured model returns a list; flatten it to bullet text so
        # downstream prompts can embed it directly inside the article request.
        response = (prompt | models.achievements_llm).invoke(info)
        achievements_list_text = "\n".join(
            f"- {achievement}" for achievement in response.achievements
        )
        return {"achievements": achievements_list_text}

    return generate_achievements


def make_generate_article(models) -> Callable[[ScientistState], dict]:
    def generate_article(state: ScientistState) -> dict:
        # Collect the earlier pipeline outputs into the article prompt inputs.
        information_data = {
            "scientist": state["scientist"],
            "history": state.get("history", ""),
            "achievements": state.get("achievements", ""),
        }
        response = (ARTICLE_PROMPT | models.article_llm).invoke(information_data)
        return {"article": response.article}

    return generate_article


def make_generate_refined_article(models) -> Callable[[ScientistState], dict]:
    def generate_refined_article(state: ScientistState) -> dict:
        generated_article = state.get("article", "")
        current_count = state.get("feedback_count", 0)

        if not generated_article:
            # Reflect a bad upstream generation instead of crashing the graph.
            return {"refined_article": "Error: Article was not generated properly."}

        # Only rewrite when the article is over 1000 words (per the notebook
        # spec); shorter articles are already concise enough.
        if len(generated_article.split()) > 1000:
            refined_article_data = {
                "scientist": state["scientist"],
                "article": generated_article,
            }
            response = (REFINE_PROMPT | models.refined_article_llm).invoke(
                refined_article_data
            )
            return {
                "refined_article": response.refined_article,
                "feedback_count": current_count,
            }

        return {
            "refined_article": generated_article,
            "feedback_count": current_count,
        }

    return generate_refined_article


# ---------------------------------------------------------------------------
# Feedback nodes
# ---------------------------------------------------------------------------


def make_refine_article_with_feedback(
    models,
) -> Callable[[FeedbackState], dict]:
    def refine_article_with_feedback(state: FeedbackState) -> dict:
        generated_article = state.get("article", "")
        # If the article was never refined, use the raw article as the base.
        existing_refined = state.get("refined_article") or generated_article
        current_count = state.get("feedback_count", 0)

        feedback_data = {
            "refined_article": existing_refined,
            "feedback": state.get("feedback", ""),
        }
        response = (FEEDBACK_PROMPT | models.refined_article_llm).invoke(
            feedback_data
        )
        # Guard against a structured response without the field: keep the
        # previous text rather than overwriting with something broken.
        refined_text = (
            response.refined_article
            if (response and getattr(response, "refined_article", None))
            else existing_refined
        )
        # Count this revision so the loop is guaranteed to terminate.
        current_count += 1
        return {
            "refined_article": refined_text,
            "feedback_count": current_count,
        }

    return refine_article_with_feedback


def make_llm_evaluator(models) -> Callable[[FeedbackState], dict]:
    def llm_evaluator(state: FeedbackState) -> dict:
        from langchain_core.messages import HumanMessage, SystemMessage

        # Evaluate the refinement if present, else fall back to the article.
        refined_text = state.get("refined_article") or state.get("article", "")
        try:
            # The structured evaluator returns an ``EvaluatorCritic`` verdict.
            critic = models.evaluator_llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are an expert critic. "
                            "Evaluate the provided article."
                        )
                    ),
                    HumanMessage(content=f"Article:\n{refined_text}"),
                ]
            )
            # Defensive defaults keep the graph flowing on bad outputs.
            status = getattr(critic, "status", "REJECTED")
            feedback = getattr(critic, "feedback", "Evaluation error.")
        except Exception as exc:
            # A broken evaluator must not crash the workflow: reject so the
            # feedback loop can retry rather than silently passing.
            status = "REJECTED"
            feedback = f"Error: {exc}"

        return {"accepted_rejected": status, "feedback": feedback}

    return llm_evaluator


MAX_FEEDBACK_RETRIES = 3


def route_feedback(state: FeedbackState) -> str:
    """Return ``ACCEPTED`` once the retry budget is exhausted or the critic
    accepts the article."""
    if state.get("feedback_count", 0) >= MAX_FEEDBACK_RETRIES:
        return "ACCEPTED"
    return state.get("accepted_rejected", "ACCEPTED")


# ---------------------------------------------------------------------------
# Orchestrator-worker nodes
# ---------------------------------------------------------------------------


def _scrub_name(value: str, truncate: bool) -> str:
    """Strip wrapper punctuation; optionally cut a long value at a word
    boundary.  Truncation is only used for the raw-user fallback, which is not
    a real name and just needs to fit ``ScientistInput``'s 100-char cap."""
    cleaned = value.strip().strip('"').strip("'")
    if truncate and len(cleaned) > 100:
        cleaned = cleaned[:100].rsplit(" ", 1)[0].rstrip(" \t\r\n,.;:'\"")
    return cleaned


def _salvage_scientist(candidate: str, fallback: str) -> str:
    """Return a ``ScientistInput``-valid scientist name.

    Model-based extraction can occasionally return filler instead of the bare
    name (e.g. "No scientist's name found ..."), which would fail
    ``ScientistInput``'s length/letter checks and crash the whole graph with a
    500.  The model's answer is used only if it validates as-is; otherwise fall
    back to the raw user input (truncated if needed) and only raise a clear,
    traceable error if both fail.
    """
    from .schemas import ScientistInput

    for cleaned in (
        _scrub_name(candidate, truncate=False),
        _scrub_name(fallback, truncate=True),
    ):
        if not cleaned:
            continue
        try:
            return ScientistInput.model_validate({"scientist": cleaned}).scientist
        except Exception:
            continue
    # ``!r`` renders each value via repr(), so the error message shows the
    # strings quoted ('...'), making whitespace/punctuation visible in logs.
    raise ValueError(
        "Could not extract a valid scientist name from the request "
        f"{fallback!r} (model returned {candidate!r})."
    )


def make_scientist_name_extraction(models) -> Callable[[dict], dict]:
    def scientist_name_extraction(state: dict) -> dict:
        # Free-text input -> LLM extracts the scientist -> validated again by
        # the same pydantic model the notebooks use.  ``_salvage_scientist``
        # swallows a messy extraction instead of crashing the graph.
        response = (SCIENTIST_NAME_EXTRACTION_PROMPT | models.google_llm).invoke(
            {"input": state["input"]}
        )
        return {
            "scientist": _salvage_scientist(
                str(response.content).strip(), state["input"]
            )
        }

    return scientist_name_extraction


def make_orchestrator(models, sections: int = 2) -> Callable[[dict], dict]:
    from langchain_core.messages import HumanMessage, SystemMessage

    # Static system message; only the scientist changes between invocations.
    system = SystemMessage(
        content=(
            "You are an expert editor. Create an outline plan of "
            f"{sections} distinct sections for an article about the given "
            "scientist. Which covers the history and the achievements of the "
            "scientist"
        )
    )

    def orchestrator(state: dict) -> dict:
        # The planner LLM returns an ``OutlinePlan`` of ``Section`` objects.
        plan = models.planner_llm.invoke(
            [system, HumanMessage(content=f"Scientist: {state['scientist']}")]
        )
        # ``completed_sections`` is reset here: the reducer in the state schema
        # lets every parallel worker append its own section.
        return {"sections": plan.sections, "completed_sections": []}

    return orchestrator


def make_worker_section_writer(
    models, research: Optional[Callable[[str], str]] = None
) -> Callable[[WorkerState], dict]:
    # The tools variant uses the research-grounded prompt; the plain variant
    # uses the short one.  Both are imported from ``scripting.prompts``.
    worker_prompt = (
        WORKER_SECTION_RESEARCH_PROMPT if research is not None else WORKER_SECTION_PROMPT
    )

    def worker_section_writer(state: WorkerState) -> dict:
        # ``WorkerState`` is the per-worker Send payload (see ``assign_workers``).
        scientist = state["scientist"]
        title = state["section"].title
        description = state["section"].description

        info = {
            "scientist": scientist,
            "title": title,
            "description": description,
        }
        if research is not None:
            # Ground this section with a targeted web search.
            info["research"] = research(
                f"Research the section '{title}' about {scientist}: {description}"
            )

        response = (worker_prompt | models.google_llm).invoke(info)
        # Return one pre-formatted markdown section; the reducer appends it to
        # ``completed_sections`` from any number of parallel workers.
        formatted_section = f"### {title}\n\n{str(response.content).strip()}"
        return {"completed_sections": [formatted_section]}

    return worker_section_writer


def make_synthesizer(models, feedback_fields: bool = False) -> Callable[[dict], dict]:
    def synthesizer(state: dict) -> dict:
        completed_sections = state.get("completed_sections", [])
        current_count = state.get("feedback_count", 0)
        # Feedback variants must carry the counter through this node.
        base_out = {"feedback_count": current_count} if feedback_fields else {}

        if not completed_sections:
            # No worker produced anything: surface a readable error.
            return {
                "article": "",
                "refined_article": "Error: Article was not generated properly.",
                **base_out,
            }

        # Join every worker's markdown into one article.
        generated_article = "\n\n".join(
            sec.strip() for sec in completed_sections if sec and sec.strip()
        )
        article_length = len(generated_article.split())

        # Mirror the refinement policy of the chain workflows: refine only
        # articles longer than 1000 words.
        if article_length > 1000:
            refined_article_data = {
                "scientist": state["scientist"],
                "article": generated_article,
            }
            try:
                response = (REFINE_PROMPT | models.refined_article_llm).invoke(
                    refined_article_data
                )
                refined_text = (
                    response.refined_article
                    if (response and getattr(response, "refined_article", None))
                    else generated_article
                )
            except Exception:
                # Refinement is best-effort; keep the raw article if it fails.
                refined_text = generated_article

            return {
                "article": generated_article,
                "refined_article": refined_text,
                **base_out,
            }

        # Short enough: the article is its own refined version.
        return {
            "article": generated_article,
            "refined_article": generated_article,
            **base_out,
        }

    return synthesizer


def assign_workers(state: dict):
    """Conditional edge using the ``Send`` API to spawn parallel workers."""
    from langgraph.types import Send

    # One ``Send`` event per planned section -> ``worker_section_writer`` is
    # invoked concurrently for each section while sharing the group state.
    return [
        Send(
            "worker_section_writer",
            {"scientist": state["scientist"], "section": section},
        )
        for section in state["sections"]
    ]
