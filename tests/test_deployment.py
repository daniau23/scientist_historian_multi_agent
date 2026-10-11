"""Tests for the deployment layer (service, redaction, frontend, Modal app).

Covers the service execution wrapper, redaction helper, frontend HTTP call,
Modal app metadata, and critically the contract between the dependency-free
frontend catalog and the engine registry (they must remain in sync).  Also
verifies JSON-serialisation of Pydantic models.
"""

from __future__ import annotations

import sys
from unittest.mock import Mock

from deployment import frontend
from deployment import modal_app  # noqa: F401  (exercises the module import)
from deployment import registry as engine_registry
from deployment.catalog import WORKFLOWS as CATALOG
from deployment.fakes import make_fake_models
from deployment.frontend import call_endpoint
from deployment.redaction import redact_payload
from deployment.service import build_payload, execute, json_safe
from deployment.pii import Redactor
from deployment.schemas import Section
from deployment.registry import get_workflow
from tests.fake_streamlit import FakeStreamlit


def test_catalog_matches_engine_registry():
    """The dependency-free frontend catalog (no LangGraph/LangChain imports)
    must stay in sync with the engine registry.  If these drift, the Streamlit
    UI will advertise workflows that the service cannot run or vice versa."""
    catalog_names = [meta.name for meta in CATALOG]
    engine_names = [info.name for info in engine_registry.list_workflows()]
    assert catalog_names == engine_names
    for meta in CATALOG:
        info = engine_registry.get_workflow(meta.name)
        assert (
            meta.kind,
            meta.has_feedback,
            meta.has_tools,
            meta.input_kind,
            meta.description,
        ) == (
            info.kind,
            info.has_feedback,
            info.has_tools,
            info.input_kind,
            info.description,
        )


def test_execute_returns_jsonable_payload():
    """``execute`` wraps a workflow run with metadata (name and input_kind) and
    the raw ``result`` dict; all values must be JSON-serialisable."""
    result = execute(
        "prompt_chaining",
        {"scientist": "Isaac Newton"},
        models=make_fake_models(),
    )
    assert result["workflow"] == "prompt_chaining"
    assert result["input_kind"] == "scientist"
    assert isinstance(result["result"]["refined_article"], str)


def test_build_payload_respects_input_kind():
    """The service's ``build_payload`` ignores the wrong input field depending
    on whether the workflow expects ``scientist`` or ``input``."""
    scientist_info = get_workflow("prompt_chaining")
    assert build_payload(scientist_info, {"scientist": "Ada", "input": "x"}) == {
        "scientist": "Ada"
    }
    routing_info = get_workflow("routing")
    assert build_payload(routing_info, {"input": "article please"}) == {
        "input": "article please"
    }


def test_json_safe_converts_pydantic_models():
    """``json_safe`` recursively converts Pydantic ``BaseModel`` instances into
    plain dicts so the result can be dumped to JSON."""
    section = Section(title="History", description="Early life.")
    assert json_safe(section) == {"title": "History", "description": "Early life."}
    assert json_safe({"sections": [section]}) == {
        "sections": [{"title": "History", "description": "Early life."}]
    }


def test_redact_payload_copies_and_redacts():
    """``redact_payload`` produces a scrubbed *copy* without mutating the
    original payload (important if it is logged or reused)."""
    redactor = Redactor(use_presidio=False)
    payload = {"messages": [{"content": "ping a@b.com"}]}
    safe = redact_payload(redactor, payload)
    assert "[EMAIL]" in safe["messages"][0]["content"]
    assert payload["messages"][0]["content"] == "ping a@b.com"


def test_call_endpoint_posts_json(monkeypatch):
    """The frontend helper posts JSON and returns the parsed response."""
    fake_response = Mock()
    fake_response.json.return_value = {"workflow": "routing", "result": {}}
    monkeypatch.setattr(
        "deployment.frontend.requests.post",
        lambda url, json, timeout: fake_response,
    )
    result = call_endpoint("https://example.run", {"workflow": "routing"})
    assert result == {"workflow": "routing", "result": {}}
    fake_response.raise_for_status.assert_called_once()


def test_modal_app_module_metadata():
    """The Modal app exposes its name in two common places (module constant and
    the ``modal.App`` instance)."""
    assert modal_app.APP_NAME == "scientist-workflows"
    assert modal_app.app.name == "scientist-workflows"


USER_KEY = "AIzaFAKEKEY0000000000000000000000000000"


def _install_fake_streamlit(monkeypatch, **kwargs) -> FakeStreamlit:
    """Inject a fake ``streamlit`` module for the next ``frontend.main()``."""
    fake_st = FakeStreamlit(**kwargs)
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    return fake_st


def test_frontend_requires_key_before_showing_workflow(monkeypatch):
    """Without a key the frontend must render the BYOK gate and never POST."""
    fake_st = _install_fake_streamlit(monkeypatch, buttons={"Run workflow": True})

    calls = {"n": 0}

    def _fail(*args, **kwargs):  # pragma: no cover - should never be called
        calls["n"] += 1
        raise AssertionError("endpoint must not be called without a key")

    monkeypatch.setattr(frontend, "call_endpoint", _fail)

    frontend.main()

    assert calls["n"] == 0
    assert fake_st.subheaders  # the gate form was shown
    # The workflow UI (dropdown) is not rendered while gated.
    assert fake_st.titles  # title is always shown


def test_frontend_forwards_user_key_in_payload(monkeypatch):
    """With a key set, the frontend runs the workflow and includes that key
    in the POST body so the backend can bill the user's own quota."""
    fake_st = _install_fake_streamlit(
        monkeypatch,
        session={"google_api_key": USER_KEY},
        buttons={"Run workflow": True},
        inputs={"Scientist name": "Ada Lovelace", "Request": "about Ada"},
    )

    captured = {}

    def _capture(endpoint, payload, timeout=600):
        captured["endpoint"] = endpoint
        captured["payload"] = payload
        return {"workflow": payload["workflow"], "result": {"refined_article": "x"}}

    monkeypatch.setattr(frontend, "call_endpoint", _capture)
    monkeypatch.setenv("MODAL_URL", "https://example.run")

    frontend.main()

    assert captured["payload"]["google_api_key"] == USER_KEY
    assert captured["endpoint"] == "https://example.run"


def _tools_workflow_label() -> str:
    """Label string for the first tools-enabled workflow in the catalog."""
    meta = next(meta for meta in CATALOG if meta.has_tools)
    return f"{meta.label} ({meta.name})"


def test_public_request_strips_both_secret_keys():
    """``_public_request`` removes every secret field before tracing/logging."""
    safe = modal_app._public_request(
        {
            "workflow": "x",
            "scientist": "Ada",
            "google_api_key": "AIza-secret",
            "tavily_api_key": "tvly-secret",
        }
    )
    assert safe == {"workflow": "x", "scientist": "Ada"}


def test_make_anonymizer_scrubs_trace_payload_secrets():
    """The LangSmith anonymiser strips keys from a models-like trace payload
    (the ``google_api_key`` inside the ``Models`` bundle, plus any raw key in
    the request) and leaves everything else intact."""
    from deployment.pii import Redactor

    anonymizer = modal_app._make_anonymizer(Redactor(use_presidio=False))
    # LangSmith serialises run inputs first, so feed JSON-native data.
    payload = {
        "workflow": "prompt_chaining",
        "request": {
            "workflow": "prompt_chaining",
            "scientist": "Ada",
            "google_api_key": "AIzaFAKEKEY0000000000000000000000000000",
            "tavily_api_key": "tvly-dev-FAKEKEY",
        },
        "models": {
            "google_llm": {
                "model": "gemini-2.5-flash",
                "google_api_key": "**********",
            }
        },
    }
    cleaned = anonymizer(payload)
    dumped = str(cleaned)
    assert "AIza" not in dumped
    assert "tvly-" not in dumped
    assert cleaned["request"]["scientist"] == "Ada"
    assert cleaned["models"]["google_llm"]["model"] == "gemini-2.5-flash"
    assert cleaned["models"]["google_llm"]["google_api_key"] == "[REDACTED]"


def test_frontend_warns_and_forwards_empty_tavily_key_for_tools(monkeypatch):
    """Selecting a tools workflow prompts for a Tavily key; without one the app
    warns and still sends an empty key (so the backend disables search)."""
    fake_st = _install_fake_streamlit(
        monkeypatch,
        session={"google_api_key": USER_KEY},
        buttons={"Run workflow": True},
        inputs={"Scientist name": "Ada", "Request": "about Ada"},
        selectbox=_tools_workflow_label(),
    )

    captured = {}
    monkeypatch.setattr(
        frontend,
        "call_endpoint",
        lambda endpoint, payload, timeout=600: captured.update(payload)
        or {"result": {}},
    )
    monkeypatch.setenv("MODAL_URL", "https://example.run")

    frontend.main()

    assert captured["tavily_api_key"] == ""
    assert any("Tavily" in w for w in fake_st.warnings)


def test_frontend_forwards_provided_tavily_key(monkeypatch):
    """When the user pastes a Tavily key it is forwarded with the request."""
    fake_st = _install_fake_streamlit(
        monkeypatch,
        session={"google_api_key": USER_KEY},
        buttons={"Run workflow": True},
        inputs={
            "Scientist name": "Ada",
            "Request": "about Ada",
            "tavily_api_key": "tvly-user-key",
        },
        selectbox=_tools_workflow_label(),
    )

    captured = {}
    monkeypatch.setattr(
        frontend,
        "call_endpoint",
        lambda endpoint, payload, timeout=600: captured.update(payload)
        or {"result": {}},
    )
    monkeypatch.setenv("MODAL_URL", "https://example.run")

    frontend.main()

    assert captured["tavily_api_key"] == "tvly-user-key"
    assert fake_st.warnings == []