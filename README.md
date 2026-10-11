# Scientist Multi-Agent Workflows

Twelve LangGraph multi-agent workflows that research and write articles about
scientists, exposed three ways:

1. **`notebooks/`** — the original Jupyter notebooks (the source of truth).
2. **`scripting/`** — a clean, runnable Python package + CLI (`python -m scripting`).
3. **`deployment/`** — a self-contained production app: a **Modal** backend and a
   **Streamlit** frontend, with **LangSmith** tracing and **PII redaction**.

A reader who is new to the project can follow this README top-to-bottom to
understand the design, run everything locally, and reproduce the deployment.

---

## Table of contents

- [The four architectures](#the-four-architectures)
- [The three variants of each](#the-three-variants-of-each)
- [Repository layout](#repository-layout)
- [How the pieces fit together](#how-the-pieces-fit-together)
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Running the workflows (CLI)](#running-the-workflows-cli)
- [Testing](#testing)
- [Converting notebooks to Python](#converting-notebooks-to-python)
- [Deployment](#deployment)
- [Design notes and conventions](#design-notes-and-conventions)
- [Troubleshooting / FAQ](#troubleshooting--faq)
- [Extending: adding a new workflow](#extending-adding-a-new-workflow)

---

## The four architectures

Each architecture solves "write an article about a scientist" a different way.
Reading them in this order makes the progression
easy to follow: `prompt_chaining` -> `parallelization` -> `routing` ->
`orchestrator_worker`.

### 1. Prompt chaining (`prompt_chaining`)

A straight line: every step feeds the next. Simple, predictable, easy to debug.

```
START -> generate_history -> generate_achievements -> generate_article -> generate_refined_article -> END
```

### 2. Parallelization (`parallelization`)

Two independent steps run at the same time and converge, cutting latency.

```
START ─┬─> generate_history ──────┐
       └─> generate_achievements ─┴─> generate_article -> generate_refined_article -> END
```

### 3. Routing (`routing`)

An LLM acts as a switchboard: it classifies the request and sends it down the
cheapest sufficient path (`history`, `accomplishment`, or the full `article`
subgraph).

```
START -> scientist_name_extraction -> llm_routing ─┬─> generate_history -> END
                                                   ├─> generate_achievements -> END
                                                   └─> article_workflow (subgraph) -> END
                                                          (parallel history+achievements -> article -> refine)
```

### 4. Orchestrator-worker (`orchestrator_worker`)

A planner decides *how many* sections are needed, then LangGraph's `Send` API
fans out one worker per section **in parallel**; a synthesizer stitches the
sections back together.

```
START -> scientist_name_extraction -> orchestrator ──Send──> worker_section_writer (xN) -> synthesizer -> END
```

## The three variants of each

| Variant   | Suffix             | What it adds |
| --------- | ------------------ | ------------ |
| Base      | *(none)*           | The architecture as described above. |
| Feedback  | `_feedback`        | An LLM **critic** evaluates the article. `REJECTED` articles are revised and re-evaluated, up to `MAX_FEEDBACK_RETRIES` (3). |
| Feedback + Tools | `_feedback_tools` | Same as feedback, but generation steps are **grounded with a Tavily web search** before writing. |

Feedback loop (added to every architecture):

```
... -> evaluate_feedback_llm ─┬─> END                    (ACCEPTED, or retries exhausted)
                              └─> refine_article_with_feedback -> evaluate_feedback_llm   (REJECTED)
```

Full matrix (12 workflows):

| Architecture        | Base                  | Feedback                  | Feedback + Tools                  |
| ------------------- | --------------------- | ------------------------- | --------------------------------- |
| Prompt chaining     | `prompt_chaining`     | `prompt_chaining_feedback`| `prompt_chaining_feedback_tools`  |
| Parallelization     | `parallelization`     | `parallelization_feedback`| `parallelization_feedback_tools`  |
| Routing             | `routing`             | `routing_feedback`        | `routing_feedback_tools`          |
| Orchestrator-worker | `orchestrator_worker` | `orchestrator_worker_feedback` | `orchestrator_worker_feedback_tools` |

---

## Repository layout

```
├── notebooks/                      # Original .ipynb workflows (source of truth)
├── scripting/                      # Runnable Python package + CLI
│   ├── __init__.py                 # Quiet LangGraph import + re-exports registry
│   ├── __main__.py                 # `python -m scripting` entry point
│   ├── cli.py                      # `list` / `run` / `convert` subcommands
│   ├── config.py                   # .env loading + env helpers
│   ├── models.py                   # Model ids + hyper-parameters (generator/evaluator)
│   ├── schemas.py                  # Pydantic models + LangGraph state TypedDicts
│   ├── llms.py                     # Lazy LLM / structured-chain construction (`Models`)
│   ├── nodes.py                    # Reusable LangGraph node factories
│   ├── prompts.py                  # All prompt templates
│   ├── tools.py                    # Tavily web-search tooling (for *_tools)
│   ├── pii.py                      # Regex + optional Presidio redaction
│   ├── tracing.py                  # LangSmith helpers (no-op fallback)
│   ├── registry.py                 # Data-driven workflow discovery / dispatch
│   ├── convert.py                  # Notebook (.ipynb) -> .py converter
│   ├── workflows/                  # One module per runnable workflow (12)
│   └── converted/                  # Auto-generated notebook dumps (do not edit)
├── deployment/                     # Self-contained (vendors the engine; no `scripting` imports)
│   ├── modal_app.py                # Modal backend (endpoints, tracing, redaction)
│   ├── service.py                  # Modal-independent request handling (`execute`)
│   ├── frontend.py                 # Streamlit frontend (imports `catalog` + `byok_gate`)
│   ├── catalog.py                  # Dependency-free workflow metadata for the UI
│   ├── redaction.py                # Redaction re-exports
│   ├── fakes.py                    # Fake LLMs for offline tests / local runs
│   ├── _quiet.py                   # LangGraph import-time warning suppression
│   ├── config.py, models.py, schemas.py, prompts.py, pii.py, tools.py,
│   │   tracing.py, llms.py, nodes.py, registry.py, workflows/   # Vendored engine
│   ├── requirements.txt            # Full backend dependencies
│   └── requirements-streamlit.txt  # Lightweight frontend-only dependencies
├── byok_gate/                      # Standalone "bring your own key" gate (no app deps)
│   ├── gate.py                     # Streamlit gate: prompt, verify, store the user's key
│   ├── verification.py             # Local format check + optional online verification
│   └── README.md                   # Usage / how to extract into its own package
└── tests/                          # pytest suite (network-free)
    ├── fakes.py                    # Deterministic fake LLMs
    └── fake_streamlit.py           # Minimal Streamlit test double
```

## How the pieces fit together

```
notebooks/*.ipynb  ──(convert.py)──▶  scripting/converted/*.py   (reference dumps)
        │
        │  hand-written equivalents
        ▼
scripting/workflows/*.py  ──▶  scripting/registry.py  ──▶  scripting/cli.py  ──▶  python -m scripting run ...
        │
        │  vendored copy (kept byte-identical)
        ▼
deployment/workflows/*.py  ──▶  deployment/registry.py ──▶ deployment/service.py ──▶ deployment/modal_app.py (Modal)
                                                                                  └─▶ deployment/frontend.py (Streamlit)
```

Key idea: **the workflows, nodes, prompts, schemas and LLMs are shared logic**.
The CLI and the deployment are just two front doors onto the same engine. The
`deployment/` copy exists so the online app can be pushed to its own repository
and deployed without dragging in the notebooks, CLI or tests.

### Bring your own key (BYOK)

The online app never runs on the host's quota. The Streamlit frontend first
shows a `byok_gate` prompt; the visitor must paste **their own** Google AI
Studio API key, which is verified against Google and then forwarded with every
request. The Modal backend uses that key for the request and **rejects requests
without one**, so users can only ever spend their own quota. The key is never
logged or traced. See `byok_gate/README.md` for the standalone package.

The `*_tools` workflows also use Tavily web search. When one of those is
selected, the sidebar asks for the visitor's own **Tavily** key. This one is
*optional*: with a key the search runs on the user's account; without one the
workflow still runs, web search is **skipped**, and the app warns the visitor to
paste a key. It never falls back to a host Tavily key.

---

## Prerequisites

- **Python 3.11+** (developed against 3.11; `schemas.py` uses `typing.NotRequired`).
- A **Google Generative AI (Gemini)** API key — required to actually run a workflow
  (for the CLI/local use). In the **deployed** app each end user supplies their own
  key through the `byok_gate` prompt, so the host does not need one.
- **`TAVILY_API_KEY`** — for the `*_tools` workflows in CLI/local runs. In the
  **deployed** app each user pastes their own Tavily key (optional; without it
  web search is skipped).
- **Modal** + **LangSmith** accounts — only for the deployment.
- `modal` and `streamlit` CLIs are installed as Python packages (see below).

## Quick start

```bash
# 1. Create and activate a virtual environment (any shell)
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# 2. Install dependencies (full set: engine + Modal + Streamlit + Presidio)
pip install -r deployment/requirements.txt

# 3. Provide your API keys
cp .env.example .env      # macOS/Linux   (Windows: copy .env.example .env)
#   ...then edit .env — see "Configuration" below

# 4. Verify the package imports with no keys / no network (should list 12 workflows)
python -m scripting list
```

The exact runtime versions are pinned in `deployment/requirements.txt`. The
**frontend-only** dependency set (used when you deploy the Streamlit app on its
own) is `deployment/requirements-streamlit.txt`. A `.env.example` template is
provided at the repository root.

## Configuration

Configuration is read from a **`.env` file in the repository root** (matching the
notebooks) or from the process environment. `python-dotenv` strips surrounding
quotes; the environment always wins over `.env` (so Modal secrets override local
values).

| Variable | Purpose | Default |
| -------- | ------- | ------- |
| `GOOGLE_AI_STUDIO_GENERATIVE_MODEL_API_KEY` (or `GOOGLE_API_KEY`) | Gemini API key (required) | — |
| `GOOGLE_MODEL` | Generator model | `gemini-2.5-flash` |
| `GOOGLE_EVALUATOR_MODEL` | Evaluator (critic) model | `gemini-3.7-flash` |
| `TAVILY_API_KEY` | Web search for the CLI/local runs (only `*_tools`) | — |
| `LANGSMITH_API_KEY` | LangSmith tracing | — |
| `LANGSMITH_PROJECT` | LangSmith project name | — |
| `LANGSMITH_TRACING` | Enable tracing (`true`/`false`) | `false` |
| `MODAL_URL` | **Frontend only** — the deployed Modal `run` endpoint | — |
| `BYOK_SKIP_ONLINE_VERIFY` | **Frontend only** — set `1`/`true` to accept keys on format alone (hosts that cannot reach Google) | `false` |

Model ids and shared hyper-parameters (`temperature=0.1`, `max_output_tokens=4096`)
live in `scripting/models.py` and mirror the `ChatGoogleGenerativeAI` blocks in
the notebooks; the deployment copy is `deployment/models.py`.

The `GOOGLE_AI_STUDIO_GENERATIVE_MODEL_API_KEY` and `TAVILY_API_KEY` are used by
the **CLI and local runs**. The deployed backend does **not** read them: the
Streamlit frontend collects each user's own keys and passes them per request.

Example `.env`:

```dotenv
GOOGLE_AI_STUDIO_GENERATIVE_MODEL_API_KEY="your-gemini-key"
GOOGLE_MODEL="gemini-2.5-flash"
GOOGLE_EVALUATOR_MODEL="gemini-3.7-flash"
TAVILY_API_KEY="your-tavily-key"
LANGSMITH_API_KEY="your-langsmith-key"
LANGSMITH_PROJECT="scientist-workflows"
LANGSMITH_TRACING="true"
MODAL_URL="https://<workspace>--researcher-workflow.modal.run"
```

## Running the workflows (CLI)

List everything that is registered:

```bash
python -m scripting list
```

Scientist-name workflows take `--scientist`; routing/orchestrator workflows take
free-text `--input`:

```bash
python -m scripting run prompt_chaining --scientist "Isaac Newton"
python -m scripting run parallelization_feedback_tools --scientist "Marie Curie"
python -m scripting run routing --input "Give me an article about Alan Turing"
python -m scripting run orchestrator_worker --input "Write about Einstein"
```

Useful flags:

| Flag | Effect |
| ---- | ------ |
| `--json` | Print the full workflow state as JSON instead of the rendered article. |
| `--trace` | Enable LangSmith tracing for this run (warns if `LANGSMITH_TRACING` is off). |

If you omit `--scientist`/`--input`, the CLI prompts for it interactively.

By default only the **final output** is printed, preceded by workflow-specific
summary lines:

```text
Workers: 2
Feedback rounds: 1

## **Albert Einstein: A Revolutionary Mind**
...
```

## Testing

```bash
python -m pytest -q
```

The suite is **network-free**: `tests/fakes.py` (and `deployment/fakes.py`) supply
deterministic fake LLMs, so every workflow graph builds and runs end-to-end
offline with no API keys. Coverage includes:

- schema validation rules (`ScientistInput`, `Route`, `EvaluatorCritic`, `OutlinePlan`);
- PII regex redaction (non-mutating; pass-through for non-strings);
- registry discovery and metadata;
- graph wiring and execution for all 12 workflows;
- routing decision paths and evaluator retry loops;
- notebook -> Python conversion (magics stripped, markdown skipped);
- CLI subcommands, deployment service logic, and the Modal app module;
- consistency between the UI `catalog` and the engine `registry`.

## Converting notebooks to Python

The notebooks are the source of truth. To regenerate reference `.py` dumps
(written to `scripting/converted/`, no extra dependency needed):

```bash
python -m scripting convert
```

IPython magics (`%...`) and shell escapes (`!...`) are stripped so the output is
valid Python. **`scripting/converted/` is auto-generated — do not edit it by
hand.** The hand-written, tested versions live in `scripting/workflows/`.

---

## Deployment

The app has two halves:

```
     ┌───────────────┐  POST (JSON + user's Google key) ┌───────────────────────────────┐
     │   Streamlit   │ ─────────────────────────────────▶ │   Modal (FastAPI endpoint)     │
     │  frontend.py  │   byok_gate collects/verifies it   │   modal_app.py                 │
     │   (browser)   │ ◀──────────── JSON result ───────── │    └─ service.execute           │
     └──────┬────────┘                                     │         └─ registry            │
            │  key typed by the user                       │              └─ workflow graph │
            ▼   (verify: generativelanguage...)            └───────────┬───────────────────┘
       Google AI Studio                                                 │ per-request Models
       (issues the key)                                                 ▼
                                                          Gemini  +  Tavily (tools)
                                                                        │
                                                                        ▼
                                                          LangSmith traces (PII-redacted;
                                                          the user's key is excluded)
```

The `MODAL_URL` is resolved from the environment or `.streamlit/secrets.toml`,
never hardcoded or shown to end users.

### Modal backend

`deployment/modal_app.py` defines a class-based Modal app (`WorkflowService`)
modelled on `server.py`. On container start it builds a shared `Redactor`.
`Models` bundles are built **per request** on the caller's key (never cached, so
a user's key is not retained in memory). It exposes:

| Method | Label | URL | Purpose |
| ------ | ----- | --- | ------- |
| `GET`  | `health` | `https://<workspace>--health.modal.run` | Liveness probe (development/monitoring only). |
| `POST` | `researcher-workflow` | `https://<workspace>--researcher-workflow.modal.run` | Accepts `{"workflow": "...", "scientist"\|"input": "...", "google_api_key": "AIza...", "tavily_api_key": "tvly-..."}`. Requests without `google_api_key` are rejected; a missing `tavily_api_key` only disables web search. |

The URL is derived from the endpoint **label**, not a `/run` path. The
`POST` label is what you put in `MODAL_URL`.

Every request is traced to LangSmith; anything logged (inputs **and** outputs)
passes through `deployment.pii.Redactor`, so emails, phones, credit cards,
SSNs/PPSNs, zip codes and passports are masked before they reach LangSmith. The
user's `google_api_key` and `tavily_api_key` are stripped from the request
before it is recorded, so they never appear in traces or logs.

Deploy it:

```bash
# 1. Authenticate once
modal token new

# 2. Create the secret with the non-key settings (no Google/Tavily keys stored)
modal secret create researcher-scientist-agent-workflows \
  LANGSMITH_API_KEY=... LANGSMITH_PROJECT=... LANGSMITH_TRACING=true

# 3. Deploy (prints the URLs for both endpoints)
modal deploy deployment/modal_app.py

# 4. Optional local smoke test (pass your own keys)
modal run deployment/modal_app.py --api-key "$GOOGLE_AI_STUDIO_GENERATIVE_MODEL_API_KEY"
```

> **Note:** the `@app.cls(...)` block still uses Modal's pre-1.0 parameter names
> (`container_idle_timeout`, `allow_concurrent_inputs`). Modal emits deprecation
> warnings for these; they were intentionally left as-is. See
> <https://modal.com/docs/guide/modal-1-0-migration>.

### Streamlit frontend

`deployment/frontend.py` talks **only** to the public `POST` endpoint. The URL is
never hardcoded or shown in the UI: it is resolved from `MODAL_URL`, first in the
environment (picked up from `.env` during local dev) and then in Streamlit
secrets. The `GET /health` endpoint is intentionally not exposed to end users.

Before any workflow UI appears, the app renders the **BYOK gate**
(`byok_gate.require_google_api_key`): the visitor enters their own Google AI
Studio API key, which is format-checked and (by default) verified against Google
before the app continues. A small sidebar badge shows the active key and offers
**Change key**. Set `BYOK_SKIP_ONLINE_VERIFY=1` to accept keys on format alone.

When a `*_tools` workflow is selected, the sidebar also shows a **Tavily API
key** field. If it is left empty the app still runs the workflow, skips web
search, and warns the visitor to paste a key.

Run it locally:

```bash
# Frontend-only (light) dependencies:
pip install -r deployment/requirements-streamlit.txt
streamlit run deployment/frontend.py
```

For a deployed app, instead of `.env`, set the secret:

```toml
# .streamlit/secrets.toml
MODAL_URL = "https://<workspace>--researcher-workflow.modal.run"
```

Then select a workflow type (the dropdown is populated from the dependency-free
`deployment/catalog.py`), fill in the field, and click **Run workflow**. The
**Full workflow state** expander shows the raw state: feedback workflows expose
their `feedback_count` directly, and orchestrator-worker workflows get a derived
`workers` count (the length of `completed_sections`), mirroring the CLI's
`Workers: N` summary line.

### Reproducing the deployment from scratch

1. `pip install -r deployment/requirements.txt`
2. `modal token new`
3. `modal secret create researcher-scientist-agent-workflows LANGSMITH_API_KEY=... LANGSMITH_PROJECT=... LANGSMITH_TRACING=true`
   (no Google/Tavily keys — end users bring their own)
4. `modal deploy deployment/modal_app.py` and copy the `POST` URL
5. Put that URL in `.streamlit/secrets.toml` (or export `MODAL_URL`)
6. `pip install -r deployment/requirements-streamlit.txt`
7. `streamlit run deployment/frontend.py`

Because `deployment/` vendors the engine, you can push just that folder to a
separate repository for the Streamlit app. When you do, ship `byok_gate/` too (or
install it as a package): the frontend imports it at runtime.

---

## Design notes and conventions

- **Import safety.** Importing any module requires no API keys, no network and no
  optional dependencies. `presidio`, `tavily` and the Google SDK are imported
  lazily. This is what lets the tests run offline and lets the frontend import
  only `catalog.py`.
- **Dependency injection via `Models`.** Node factories close over an injectable
  `Models` bundle (`scripting/llms.py`), which is exactly what lets tests swap in
  fakes and keeps workflows free of global state.
- **Structured output with retry.** Gemini occasionally answers the first
  structured request with plain text, which the schema parser turns into `None`.
  `llms._structured(..., attempts=3)` retries and returns the expected object.
- **Robust scientist-name extraction.** `nodes._salvage_scientist` guards the
  free-text -> name step: if the model returns a long refusal/comment (which
  would fail `ScientistInput`'s 100-char cap and 500 the request), it falls back
  to the raw user input instead of crashing the graph.
- **Quiet LangGraph import.** A module on LangGraph's import path resets the
  global warnings filter, letting a deprecation warning leak through. Both
  `scripting/__init__.py` and `deployment/_quiet.py` import LangGraph with the
  filter-mutating helpers temporarily neutralised.
- **PII redaction is non-mutating.** `Redactor.hybrid_anonymize_data` deep-copies
  its input before masking, because the same payload is also sent to the
  workflow (`deployment/redaction.py:redact_payload` documents this).
- **Single source of truth for metadata.** Each workflow module declares
  `NAME` / `KIND` / `HAS_FEEDBACK` / `HAS_TOOLS` / `INPUT_KIND` / `DESCRIPTION`;
  `registry.py` builds its table from these, and `catalog.py` mirrors them for
  the UI. `tests/test_deployment.py` asserts the two agree.

## Troubleshooting / FAQ

**"The first Gemini call returns `None` / a validation error."**
Known transient behaviour; `_structured` retries up to 3 times. If it persists,
re-run or try a different `GOOGLE_MODEL`.

**"500 error: `ScientistInput` / `string_too_long`."**
The name-extraction model returned a whole sentence instead of a name. This is
handled by `_salvage_scientist`; make sure you are running the current code.
(The deployed container only picks up the fix after `modal deploy`.)

**"`from scripting import registry` fails when I run the frontend."**
Run the Streamlit app against `deployment/` (which vendors the engine). When run
as a script, `frontend.py` bootstraps `sys.path` itself, so
`streamlit run deployment/frontend.py` works from the repo root.

**LangGraph deprecation warnings on startup.**
Suppressed by the quiet import. If you still see them, your environment may be
resetting `warnings` — check `scripting/__init__.py` / `deployment/_quiet.py`.

**"Presidio is not installed."**
Optional. The redactor silently degrades to the regex layer only. To enable full
NER redaction: `pip install presidio-analyzer presidio-anonymizer` and
`python -m spacy download en_core_web_sm`.

**`*_tools` workflows return `WEB_SEARCH_ERROR` / `WEB_SEARCH_UNAVAILABLE` text.**
`WEB_SEARCH_UNAVAILABLE` means no Tavily key was supplied (paste one in the
sidebar). `WEB_SEARCH_ERROR` means a key was supplied but Tavily rejected it or
the network failed. In both cases the graph does not crash — the message is fed
to the LLM as text so generation continues.

**Modal deploy shows `DeprecationError` for `container_idle_timeout` /
`allow_concurrent_inputs`.** Expected; see the note in the Modal backend section.

## Extending: adding a new workflow

1. Copy the closest module in `scripting/workflows/` and rename it.
2. Update its `NAME` / `KIND` / `HAS_FEEDBACK` / `HAS_TOOLS` / `INPUT_KIND` /
   `DESCRIPTION` constants and its `build_graph` wiring.
3. Register it in `scripting/workflows/__init__.py` (import + append to `MODULES`).
4. Reuse `nodes.py` factories and `prompts.py` templates where possible; add new
   node factories to `nodes.py` if needed.
5. Add an entry to `deployment/catalog.py` (same metadata).
6. Copy the module to `deployment/workflows/` (keep the two trees in sync).
7. Add a test in `tests/test_workflows.py`.

```bash
python -m scripting list       # your workflow should appear
python -m pytest -q            # all green
```

---

## Credits and references

- Built with [LangGraph](https://langchain-ai.github.io/langgraph/),
  [LangChain](https://python.langchain.com/), Google Gemini, and Tavily.
- Deployment on [Modal](https://modal.com/); UI via
  [Streamlit](https://streamlit.io/); tracing on
  [LangSmith](https://docs.smith.langchain.com/).
- PII redaction uses [Microsoft Presidio](https://microsoft.github.io/presidio/).