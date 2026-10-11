"""Tests for the standalone ``byok_gate`` package.

These exercise the gate without Streamlit or the network: a tiny fake ``st``
object records what the gate rendered, and ``verify_key_online`` receives an
injected HTTP getter.
"""

from __future__ import annotations

import pytest

from byok_gate import (
    SESSION_KEY,
    KeyCheckResult,
    clear_api_key,
    get_api_key,
    require_google_api_key,
    validate_key_format,
    verify_key_online,
)
from byok_gate import gate as gate_module
from tests.fake_streamlit import FakeStreamlit

VALID_KEY = "AIza" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"  # ~43 chars


# ---------------------------------------------------------------------------
# validate_key_format
# ---------------------------------------------------------------------------


def test_validate_key_format_accepts_prefix_and_length():
    assert validate_key_format(VALID_KEY) is True


def test_validate_key_format_rejects_wrong_prefix():
    assert validate_key_format("sk-" + "A" * 40) is False


def test_validate_key_format_rejects_too_short():
    assert validate_key_format("AIza123") is False


def test_validate_key_format_rejects_non_string():
    assert validate_key_format(None) is False  # type: ignore[arg-type]
    assert validate_key_format(12345) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# verify_key_online
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def test_verify_key_online_ok():
    result = verify_key_online(VALID_KEY, http_get=lambda *a, **k: _FakeResponse(200))
    assert result.ok is True
    assert result.reason == "ok"


@pytest.mark.parametrize("status", [400, 401, 403])
def test_verify_key_online_invalid_statuses(status):
    result = verify_key_online(VALID_KEY, http_get=lambda *a, **k: _FakeResponse(status))
    assert result.ok is False
    assert result.reason == "invalid"


def test_verify_key_online_server_error_is_retryable():
    result = verify_key_online(VALID_KEY, http_get=lambda *a, **k: _FakeResponse(500))
    assert result.ok is False
    assert result.reason == "unverified"


def test_verify_key_online_network_error_is_retryable():
    def _boom(*a, **k):
        raise ConnectionError("no route to host")

    result = verify_key_online(VALID_KEY, http_get=_boom)
    assert result.ok is False
    assert result.reason == "unverified"
    assert "Could not reach Google" in result.message


def test_verify_key_online_empty_key_is_invalid():
    result = verify_key_online("", http_get=lambda *a, **k: _FakeResponse(200))
    assert result.ok is False
    assert result.reason == "invalid"


# ---------------------------------------------------------------------------
# session-state helpers
# ---------------------------------------------------------------------------


def test_get_and_clear_api_key():
    st = FakeStreamlit(session={SESSION_KEY: "  " + VALID_KEY + "  "})
    assert get_api_key(st) == VALID_KEY
    clear_api_key(st)
    assert get_api_key(st) is None


def test_get_api_key_none_when_absent():
    assert get_api_key(FakeStreamlit()) is None


# ---------------------------------------------------------------------------
# require_google_api_key
# ---------------------------------------------------------------------------


def test_gate_returns_existing_key_without_form():
    st = FakeStreamlit(session={SESSION_KEY: VALID_KEY})
    assert require_google_api_key(st) == VALID_KEY
    # Active-key path shows a status line, never the entry form.
    assert st.subheaders == []
    assert any("your Google API key" in s for s in st.successes)


def test_gate_blocks_and_renders_form_when_no_key():
    st = FakeStreamlit()
    assert require_google_api_key(st) is None
    assert st.subheaders  # form heading rendered
    assert st.errors == []


def test_gate_saves_valid_key_without_online_check():
    st = FakeStreamlit(inputs={"byok_input": VALID_KEY}, buttons={"Save key": True})
    assert require_google_api_key(st, verify_online=False) is None
    assert st.session_state[SESSION_KEY] == VALID_KEY
    assert st.rerun_called == 1
    assert st.errors == []


def test_gate_rejects_bad_format():
    st = FakeStreamlit(inputs={"byok_input": "not-a-key"}, buttons={"Save key": True})
    assert require_google_api_key(st, verify_online=False) is None
    assert SESSION_KEY not in st.session_state
    assert st.rerun_called == 0
    assert any("doesn't look like" in e for e in st.errors)


def test_gate_rejects_key_failing_online_verification(monkeypatch):
    monkeypatch.setattr(
        gate_module,
        "verify_key_online",
        lambda *a, **k: KeyCheckResult(False, "invalid", "Google rejected this key."),
    )
    st = FakeStreamlit(inputs={"byok_input": VALID_KEY}, buttons={"Save key": True})
    assert require_google_api_key(st, verify_online=True) is None
    assert SESSION_KEY not in st.session_state
    assert "Google rejected this key." in st.errors


def test_gate_accepts_key_passing_online_verification(monkeypatch):
    monkeypatch.setattr(
        gate_module,
        "verify_key_online",
        lambda *a, **k: KeyCheckResult(True, "ok", "API key verified with Google."),
    )
    st = FakeStreamlit(inputs={"byok_input": VALID_KEY}, buttons={"Save key": True})
    assert require_google_api_key(st, verify_online=True) is None
    assert st.session_state[SESSION_KEY] == VALID_KEY
    assert st.rerun_called == 1


def test_change_key_button_clears_session():
    st = FakeStreamlit(session={SESSION_KEY: VALID_KEY}, buttons={"Change key": True})
    require_google_api_key(st)
    assert SESSION_KEY not in st.session_state
    assert st.rerun_called == 1
