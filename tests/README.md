# Tests

pytest suite for the workflows engine, CLI, deployment, PII, BYOK, and tools.

## Structure
- akes.py - Deterministic fake LLMs and model factories for offline testing.
- ake_streamlit.py - Minimal Streamlit test double for frontend tests.
- 	est_*.py - Unit/integration tests covering workflows, CLI, convert, registry, schemas, deployment (frontend+modal), pii, tools, byok_gate.

## Running
`ash
python -m pytest -q
python -m pytest tests/test_deployment.py -q
`

Tests are network-free; Modal, external APIs, and LangSmith are mocked where needed. Secrets are never required.
