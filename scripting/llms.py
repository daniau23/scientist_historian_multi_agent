"""Lazy construction of the LLMs / structured chains used by the workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from langchain_core.runnables import RunnableLambda

from . import config, models, schemas


def _chat(
    model: str,
    temperature: float = 0.1,
    max_output_tokens: int = 4096,
    api_key: str | None = None,
) -> Any:
    """Build a ``ChatGoogleGenerativeAI`` instance.

    Imported lazily so that importing the package (and the tests) never require
    the Google SDK or a configured API key.

    ``api_key`` lets callers run on a *user-supplied* key (the Modal backend's
    bring-your-own-key path).  When it is ``None`` we fall back to the key from
    the environment, which is what local CLI/notebook use wants.
    """
    import warnings

    # The Google SDK still ships the (deprecated) ``google.generativeai``
    # package and warns about it on import; that is out of our control.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        # A caller-provided key always wins; otherwise resolve from the env.
        api_key=api_key or config.google_api_key(),
    )


def _structured(base: Any, schema: Any, attempts: int = 3) -> Any:
    """Structured-output runnable that retries empty (``None``) results.

    Gemini occasionally answers the first structured request with plain text,
    which the schema parser converts to ``None``; a retry reliably returns the
    expected object.  Wrapping this here keeps every workflow node simple.
    """
    chain = base.with_structured_output(schema)

    def _invoke(value: Any, config: Any = None) -> Any:
        for _ in range(attempts):
            result = chain.invoke(value, config=config)
            if result is not None:
                return result
        raise RuntimeError(
            f"Structured model returned no result after {attempts} attempts"
        )

    return RunnableLambda(_invoke)


@dataclass
class Models:
    """Bundle of every LLM/runnable a workflow may need.

    Attributes are optional because each workflow only uses a subset.  Tests
    build this dataclass directly with fake runnables, which is why the
    fields are not constructed eagerly here.
    """

    google_llm: Any
    google_llm_evaluator: Any
    history_llm: Any
    achievements_llm: Any
    article_llm: Any
    refined_article_llm: Any
    # Workflow-specific models; ``None`` unless the workflow opts in.
    planner_llm: Any = None  # orchestrator_worker: section planning
    router: Any = None  # routing: decision step
    evaluator_llm: Any = None  # feedback variants: critic verdict
    llm_with_tools: Any = None  # *_tools variants: tool-calling loop
    tools: List[Any] = field(default_factory=list)
    tools_by_name: dict = field(default_factory=dict)


def build_models(
    *,
    tools: bool = False,
    feedback: bool = False,
    routing: bool = False,
    orchestrator: bool = False,
    model: str | None = None,
    evaluator_model: str | None = None,
    temperature: float = models.TEMPERATURE,
    max_output_tokens: int = models.MAX_OUTPUT_TOKENS,
    api_key: str | None = None,
    tavily_api_key: str | None = None,
) -> Models:
    """Construct the bundle of structured LLMs for a workflow.

    ``api_key`` (when given) is the end-user's own Google AI Studio key; every
    model built here will bill that key instead of the host's.

    ``tavily_api_key`` is the end-user's own Tavily key for the ``*_tools``
    variants.  ``None`` (the default) keeps the CLI's environment fallback;
    an empty string disables web search rather than using a host key.
    """
    # Loading the .env is cheap (once) and makes every downstream piece
    # identical whether it runs locally or inside Modal.
    config.ensure_env()

    # Two underlying chat instances: one for generation, one for evaluation.
    # Model identifiers and hyper-parameters come from ``scripting.models``.
    base = _chat(
        model or models.GOOGLE_MODEL,
        temperature=models.TEMPERATURE,
        max_output_tokens=models.MAX_OUTPUT_TOKENS,
        api_key=api_key,
    )
    evaluator = _chat(
        evaluator_model or models.GOOGLE_EVALUATOR_MODEL,
        temperature=models.TEMPERATURE,
        max_output_tokens=models.MAX_OUTPUT_TOKENS,
        api_key=api_key,
    )

    # ``with_structured_output`` pins each chain to a pydantic schema, so the
    # node functions can read attributes like ``response.history`` directly.
    bundle = Models(
        google_llm=base,
        google_llm_evaluator=evaluator,
        history_llm=_structured(base, schemas.ScientistHistoryOutput),
        achievements_llm=_structured(base, schemas.ScientistAchievementsOutput),
        article_llm=_structured(base, schemas.ScientistArticleOutput),
        refined_article_llm=_structured(base, schemas.RefinedArticleOutput),
    )

    # Attach the optional runtimes requested by the workflow variant.
    if routing:
        bundle.router = _structured(base, schemas.Route)
    if orchestrator:
        bundle.planner_llm = _structured(base, schemas.OutlinePlan)
    if feedback:
        bundle.evaluator_llm = _structured(evaluator, schemas.EvaluatorCritic)
    if tools:
        from .tools import build_tools

        # ``bind_tools`` advertises the web_search tool to the Gemini model.
        # The Tavily key is billed to the end user (see ``build_tools``).
        bundle.tools, bundle.tools_by_name = build_tools(
            tavily_api_key=tavily_api_key
        )
        bundle.llm_with_tools = base.bind_tools(bundle.tools)

    return bundle