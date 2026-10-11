"""Tests for the web-search tooling (``deployment/tools.py``).

These run fully offline: the disabled (no-key) path never touches Tavily, and
the keyed path is exercised by monkeypatching ``TavilySearch`` so no network
call is made.
"""

from __future__ import annotations

from deployment.tools import MISSING_TAVILY_KEY_MESSAGE, build_tools


def test_build_tools_without_key_is_disabled():
    """An empty key yields a ``web_search`` tool that explains how to enable it
    instead of searching (and never uses a host key)."""
    tools, by_name = build_tools(tavily_api_key="")
    assert [t.name for t in tools] == ["web_search"]
    result = by_name["web_search"].invoke({"query": "anything"})
    assert result == MISSING_TAVILY_KEY_MESSAGE


def test_build_tools_with_user_key_uses_that_key(monkeypatch):
    """A non-empty key is handed to ``TavilySearch`` (verified via a fake that
    raises, which the tool turns into a graceful error string)."""
    import langchain_tavily

    captured = {}

    def fake_tavily(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("boom")

    monkeypatch.setattr(langchain_tavily, "TavilySearch", fake_tavily)

    _, by_name = build_tools(tavily_api_key="tvly-user-key")
    result = by_name["web_search"].invoke({"query": "q"})

    assert captured.get("tavily_api_key") == "tvly-user-key"
    assert result.startswith("WEB_SEARCH_ERROR:")


def test_build_tools_none_omits_key_for_env_fallback(monkeypatch):
    """``None`` (CLI/local) does not pass a key, leaving Tavily's environment
    fallback in place."""
    import langchain_tavily

    captured = {}

    def fake_tavily(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("boom")

    monkeypatch.setattr(langchain_tavily, "TavilySearch", fake_tavily)

    _, by_name = build_tools(tavily_api_key=None)
    by_name["web_search"].invoke({"query": "q"})

    assert "tavily_api_key" not in captured
