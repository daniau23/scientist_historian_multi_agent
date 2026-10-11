"""Tests for PII redaction (regex layer, no presidio).

The redactor scrubs potentially sensitive identifiers out of anything sent to
an LLM provider.  These tests exercise the dependency-free regex fallback
(``use_presidio=False``) so they run without the heavy presidio/spaCy stack.
"""

from __future__ import annotations

from scripting.pii import Redactor, redact_secrets


def test_regex_anonymize_common_entities():
    """Each common entity type (email, phone, card, SSN, PPSN, zip, passport)
    is replaced by its placeholder tag.  The email case also covers a
    multi-label domain (``.co.uk``) and ``+``/hyphen characters in a phone."""
    redactor = Redactor(use_presidio=False)

    cases = [
        ("reach me at a.b-c@example.co.uk", "[EMAIL]"),
        ("call +1 555 123 4567", "[PHONE]"),
        ("card 4111111111111111", "[CREDIT_CARD]"),
        ("ssn 123-45-6789", "[SSN]"),
        ("ppsn 1234567AB", "[PPSN]"),
        ("zip D02 X285", "[ZIPCODE]"),
        ("passport PA1234567", "[PASSPORT]"),
    ]
    for text, tag in cases:
        assert tag in redactor.hybrid_anonymize_text(text)


def test_hybrid_anonymize_does_not_mutate_input():
    """The returned payload is scrubbed but the caller's original dict is left
    untouched (redaction must not have side effects on shared state)."""
    redactor = Redactor(use_presidio=False)
    payload = {"messages": [{"content": "email me at a.b-c@example.co.uk now"}]}
    scrubbed = redactor.hybrid_anonymize_data(payload)
    assert "[EMAIL]" in scrubbed["messages"][0]["content"]
    assert payload["messages"][0]["content"] == (
        "email me at a.b-c@example.co.uk now"
    )


def test_hybrid_anonymize_text_passthrough():
    """Non-string inputs (e.g. ints, None) pass through unchanged rather than
    raising, since arbitrary nested JSON values may be redacted."""
    redactor = Redactor(use_presidio=False)
    assert redactor.hybrid_anonymize_text(42) == 42
    assert redactor.hybrid_anonymize_text(None) is None


def test_regex_anonymize_reports_change():
    """``regex_anonymize`` returns a ``(text, changed)`` pair: the flag must be
    true only when a substitution actually happened, false otherwise."""
    redactor = Redactor(use_presidio=False)
    changed, did_change = redactor.regex_anonymize("a.b-c@example.co.uk")
    assert did_change is True
    assert changed == "[EMAIL]"
    unchanged, did_change = redactor.regex_anonymize("no secrets here")
    assert did_change is False
    assert unchanged == "no secrets here"


def test_redact_secrets_masks_credential_named_fields():
    """Fields whose key looks like a credential are masked outright, including
    the already-masked SecretStr form LangSmith emits for models."""
    data = {
        "google_api_key": "AIzaFAKEKEY0000000000000000000000000000",
        "tavily_api_key": "tvly-dev-FAKEKEY",
        "model": "gemini-2.5-flash",
    }
    cleaned = redact_secrets(data)
    assert cleaned["google_api_key"] == "[REDACTED]"
    assert cleaned["tavily_api_key"] == "[REDACTED]"
    assert cleaned["model"] == "gemini-2.5-flash"


def test_redact_secrets_scrubs_embedded_key_strings():
    """A credential shape is scrubbed even when it is embedded in free text
    (not hidden behind a credential-named key)."""
    cleaned = redact_secrets(
        {"content": "my key is AIzaFAKEKEY0000000000000000000000000000 ok"}
    )
    assert "AIza" not in cleaned["content"]
    assert "[REDACTED]" in cleaned["content"]


def test_redact_secrets_handles_nested_and_secretstr():
    """Nested structures are walked and ``SecretStr``-like objects are masked
    without calling ``get_secret_value``."""
    class FakeSecret:
        def get_secret_value(self):  # pragma: no cover - must never be called
            raise AssertionError("raw secret must never be read")

    cleaned = redact_secrets({"a": [{"api_key": "x"}], "b": FakeSecret()})
    assert cleaned["a"][0]["api_key"] == "[REDACTED]"
    assert cleaned["b"] == "[REDACTED]"


def test_redact_secrets_does_not_mutate_input():
    """The scrubber works on a copy: the caller's original data is untouched."""
    original = {"google_api_key": "AIzaFAKEKEY0000000000000000000000000000"}
    redact_secrets(original)
    assert original["google_api_key"] == "AIzaFAKEKEY0000000000000000000000000000"
