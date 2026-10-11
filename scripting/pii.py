"""PII redaction helpers (regex + optional Presidio).

Ported from ``server.py``.  Presidio is optional at runtime: if it is not
installed the redactor falls back to the regex layer only, which keeps the
package usable (and testable) in minimal environments.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Tuple

# ---------------------------------------------------------------------------
# Regex layer: cheap, deterministic detection of the common PII families.
# Ported verbatim from ``server.py`` so behaviour matches the reference.
# ---------------------------------------------------------------------------
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(
    r"\b(?:\+?\d{1,3})?[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}\b"
)
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
PPSN_PATTERN = re.compile(r"\b\d{7}[a-z]{1,2}\b", re.IGNORECASE)
ZIP_EIR_PATTERN = re.compile(r"\b[A-Z][0-9][0-9W][ ]?[0-9A-Z]{4}\b", re.IGNORECASE)
PASSPORT_PATTERN = re.compile(r"\b[PL][A-Z]\d{7}\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Secret layer: API keys / tokens must never reach LangSmith, even if a caller
# accidentally passes a raw credential inside a traced payload.  This is a
# structural + textual scrub that runs over already-JSON-serialisable data.
# ---------------------------------------------------------------------------
REDACTED = "[REDACTED]"

# Dict keys whose value is a credential (case-insensitive substring match).
SECRET_KEY_PATTERN = re.compile(
    r"(api[_-]?key|apikey|access[_-]?token|auth[_-]?token|bearer|"
    r"client[_-]?secret|secret|password|passwd|credential)",
    re.IGNORECASE,
)

# Known credential shapes, scrubbed wherever they appear inside a string.
SECRET_VALUE_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_\-]{10,}"),  # Google AI Studio API key
    re.compile(r"tvly-[0-9A-Za-z_\-]{8,}"),  # Tavily API key
    re.compile(r"sk-[0-9A-Za-z_\-]{16,}"),  # OpenAI-style key
    re.compile(r"lsv2_[0-9A-Za-z_\-]{16,}"),  # LangSmith API key
    re.compile(r"ls__[0-9A-Za-z_\-]{16,}"),  # LangSmith API key (legacy)
    re.compile(r"gh[pousr]_[0-9A-Za-z]{16,}"),  # GitHub token
)


def _scrub_secret_strings(value: str) -> str:
    """Replace any credential-looking substring with ``[REDACTED]``."""
    for pattern in SECRET_VALUE_PATTERNS:
        value = pattern.sub(REDACTED, value)
    return value


def redact_secrets(data: Any) -> Any:
    """Recursively remove API keys/tokens from JSON-ready ``data``.

    Used as (part of) the LangSmith ``anonymizer`` callback so credentials never
    land in a trace.  Returns a new structure and never mutates the caller's
    data.  Handles dicts, lists, tuples, strings and ``SecretStr``-like objects
    (anything exposing ``get_secret_value``).
    """
    if isinstance(data, dict):
        cleaned: dict = {}
        for key, value in data.items():
            # A credential-named string field is masked outright.
            if (
                isinstance(key, str)
                and isinstance(value, str)
                and SECRET_KEY_PATTERN.search(key)
            ):
                cleaned[key] = REDACTED
            else:
                cleaned[key] = redact_secrets(value)
        return cleaned
    if isinstance(data, list):
        return [redact_secrets(item) for item in data]
    if isinstance(data, tuple):
        return [redact_secrets(item) for item in data]
    if isinstance(data, str):
        return _scrub_secret_strings(data)
    if hasattr(data, "get_secret_value") and callable(data.get_secret_value):
        # pydantic ``SecretStr`` and friends: never expose the raw secret.
        return REDACTED
    return data


class Redactor:
    """Hybrid regex + NLP anonymiser for LangSmith trace safety."""

    def __init__(self, use_presidio: bool = True) -> None:
        # Both engines start ``None``; Presidio is filled in only if available.
        self.analyzer = None
        self.anonymizer = None
        if use_presidio:
            self._try_load_presidio()

    def _try_load_presidio(self) -> None:
        # Presidio pulls in spaCy models, so it is an optional install: any
        # failure here just leaves the engines ``None`` and we degrade to the
        # regex layer.
        try:  # pragma: no cover - depends on optional install
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
        except Exception:
            self.analyzer = None
            self.anonymizer = None

    def regex_anonymize(self, text: str) -> Tuple[str, bool]:
        """Replace regex-detected PII with placeholder tokens.

        Returns ``(redacted_text, changed)`` so callers know if anything matched.
        """
        original_text = text
        # Order matters: run fine-grained patterns (email) before greedier ones
        # (phone / credit card) to avoid mis-tagging.
        text = EMAIL_PATTERN.sub("[EMAIL]", text)
        text = PHONE_PATTERN.sub("[PHONE]", text)
        text = CREDIT_CARD_PATTERN.sub("[CREDIT_CARD]", text)
        text = SSN_PATTERN.sub("[SSN]", text)
        text = PPSN_PATTERN.sub("[PPSN]", text)
        text = ZIP_EIR_PATTERN.sub("[ZIPCODE]", text)
        text = PASSPORT_PATTERN.sub("[PASSPORT]", text)
        return text, text != original_text

    def presidio_anonymize(self, text: str) -> str:
        """Run Presidio over the text when the engine is available."""
        if self.analyzer is None or self.anonymizer is None:
            return text
        # Detect the entity types in the text (only if the engine is loaded).
        results = self.analyzer.analyze(
            text=text,
            entities=[
                "PERSON",
                "LOCATION",
                "ORGANIZATION",
                "PASSPORT",
                "PPSN",
                "ADDRESS",
                "ZIPCODE",
                "EIRCODE",
            ],
            language="en",
        )
        if not results:
            return text
        return self.anonymizer.anonymize(
            text=text, analyzer_results=results
        ).text

    def hybrid_anonymize_text(self, text: str) -> str:
        """Regex first, then Presidio when available."""
        if not isinstance(text, str):
            return text
        text, _ = self.regex_anonymize(text)
        text = self.presidio_anonymize(text)
        return text

    def hybrid_anonymize_data(self, data: Any) -> Any:
        """Anonymise ``{"messages": [...]}`` style payloads without mutating.

        Used directly as the LangSmith ``hide_inputs`` / ``hide_outputs``
        callbacks, so it must tolerate the shapes LangSmith passes in.
        """
        # Deep-copy first: never mutate the caller's (real) data.
        scrubbed_data = copy.deepcopy(data)
        try:
            # Handle the two payload shapes LangSmith produces:
            # 1) ``{"messages": [...]}``, 2) ``{"choices": [{"message": ...}]}``.
            if isinstance(scrubbed_data, dict):
                message_list = scrubbed_data.get("messages") or [
                    scrubbed_data.get("choices", [{}])[0].get("message")
                ]
            elif isinstance(scrubbed_data, list):
                message_list = scrubbed_data
            else:
                return scrubbed_data

            if not message_list:
                return scrubbed_data
            for message in message_list:
                if isinstance(message, dict) and "content" in message:
                    content = message.get("content", "")
                    if content:
                        message["content"] = self.hybrid_anonymize_text(content)
        except Exception as exc:  # pragma: no cover - defensive
            # Anonymisation must never take down a request; log and return as-is.
            print(f"Anonymization error: {exc}")
        return scrubbed_data

    def anonymize_messages(self, messages: list) -> list:
        """In-place variant for message lists (kept for parity with server.py)."""
        for msg in messages:
            if isinstance(msg, dict) and "content" in msg:
                msg["content"] = self.hybrid_anonymize_text(msg["content"])
        return messages


default_redactor = Redactor()
