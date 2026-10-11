"""All prompt templates used by the workflow nodes.

Keeping the prompts in one module makes them easy to review, version and tune
without touching node logic.  Every template is a `langchain` prompt so it
composes directly with LLM runnables, e.g. `HISTORY_PROMPT | models.history_llm`.
"""

from __future__ import annotations

from langchain.prompts import PromptTemplate


# ---------------------------------------------------------------------------
# History generation
# ---------------------------------------------------------------------------

HISTORY_PROMPT = PromptTemplate.from_template(
    "Provide a brief history of {scientist}."
)

# Research-grounded variant used by the `*_tools` workflows.
HISTORY_RESEARCH_PROMPT = PromptTemplate.from_template(
    """
    Provide a brief and accurate history of {scientist}.

    Use the following research information where relevant:

    {research}

    Do not mention the research process.
    """
)


# ---------------------------------------------------------------------------
# Achievements generation
# ---------------------------------------------------------------------------

ACHIEVEMENTS_PROMPT = PromptTemplate.from_template(
    """Provide 5 top achievements of {scientist} in bullet points.
    Example:
    - Achievement 1
    - Achievement 2
    - Achievement 3
    - Achievement 4
    - Achievement 5
    """
)

# Research-grounded variant used by the `*_tools` workflows.
ACHIEVEMENTS_RESEARCH_PROMPT = PromptTemplate.from_template(
    """Provide 5 top achievements of {scientist} in bullet points.
    Example:
    - Achievement 1
    - Achievement 2
    - Achievement 3
    - Achievement 4
    - Achievement 5

    Use the extra information from a research if needed:
    {research}
    """
)


# ---------------------------------------------------------------------------
# Article generation / refinement / feedback
# ---------------------------------------------------------------------------

ARTICLE_PROMPT = PromptTemplate.from_template(
    """Provide an in depth article about {scientist} based on the following information:
    History: {history}
    Achievements: {achievements}
    Final Article must be in Markdown format with different sections. For example;
    ## **Title**

    ### **History**

    ### **Achievements**

    ### **Main Contributions**
    """
)

REFINE_PROMPT = PromptTemplate.from_template(
    """Refine the article about {scientist} based on the feedback received.

    Original Article:
    {article}

    Feedback:
    {feedback}
    """
)

FEEDBACK_PROMPT = PromptTemplate.from_template(
    """Provide constructive feedback on the article about {scientist}.

    The article is:
    {article}

    Provide the feedback in bullet points.
    """
)


# ---------------------------------------------------------------------------
# Routing (history/accomplishment/article)
# ---------------------------------------------------------------------------

ROUTING_PROMPT = PromptTemplate.from_template(
    """Route the workflow based on the input provided by the user.
    - If the input is asking for a history of a scientist, route to 'history'.
    - If the input is asking for achievements of a scientist, route to 'accomplishment'.
    - If the input is asking for an article about a scientist, route to 'article'. Especially if the input is asking 
for both history and achievements, route to 'article'.
    The input is: {input}.
    """
)


# ---------------------------------------------------------------------------
# Orchestrator-worker
# ---------------------------------------------------------------------------

# Turns free-text user input into a validated scientist name.  Instructs the
# model to reply with only the name so extraction rarely produces a long
# refusal/comment string (which would fail `ScientistInput`'s 100-char cap).
SCIENTIST_NAME_EXTRACTION_PROMPT = PromptTemplate.from_template(
    """Extract the scientist's name from the text below and reply with ONLY the
    name and nothing else - no explanation, no punctuation, no commentary.

    Text: {input}
    Scientist name:
    """
)

# Plain worker variant: writes one markdown section for a planned `Section`.
WORKER_SECTION_PROMPT = PromptTemplate.from_template(
    """Write a comprehensive section for an article about {scientist}.
        Section Title: {title}
        Section Focus: {description}

    Return the section content in Markdown format.
    """
)

# Research-grounded worker variant used by the `orchestrator_worker_feedback_tools`
# workflow: the LLM must ground its answer in the supplied research snippets.
WORKER_SECTION_RESEARCH_PROMPT = PromptTemplate.from_template(
    """Write a comprehensive section for an article about {scientist}.
        Section Title: {title}
        Section Focus: {description}

    Use the following research information where relevant:

    {research}

    Do not mention the research process.
    Return the section content in Markdown format.
    """
)
