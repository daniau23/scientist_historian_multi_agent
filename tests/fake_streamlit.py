"""A minimal, dependency-free Streamlit test double.

Streamlit is imported lazily inside the app functions, so tests can inject a
fake module (or pass a fake object directly, as the ``byok_gate`` gate does).
Only the handful of attributes the gate and the frontend actually touch are
implemented; every widget call is recorded so tests can assert on what was
rendered.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class _FakeContainer:
    """Context-manager stand-in for ``st.sidebar`` / ``st.spinner`` / expander."""

    def __init__(self, parent: "FakeStreamlit", *, kind: str = "box") -> None:
        self._parent = parent
        self._kind = kind

    def __enter__(self) -> "_FakeContainer":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def success(self, message: str, *args: object, **kwargs: object) -> None:
        target = (
            self._parent.sidebar_successes
            if self._kind == "sidebar"
            else self._parent.successes
        )
        target.append(message)

    def caption(self, message: str, *args: object, **kwargs: object) -> None:
        target = (
            self._parent.sidebar_captions
            if self._kind == "sidebar"
            else self._parent.captions
        )
        target.append(message)

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        self._parent.warnings.append(message)

    def button(self, label: str, *args: object, **kwargs: object) -> bool:
        return bool(self._parent.buttons.get(label, False))

    def markdown(self, text: str, *args: object, **kwargs: object) -> None:
        self._parent.markdowns.append(text)

    def json(self, data: Any, *args: object, **kwargs: object) -> None:
        self._parent.jsons.append(data)


class FakeStreamlit:
    """Just enough Streamlit for ``byok_gate`` and ``deployment.frontend``."""

    def __init__(
        self,
        *,
        session: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        buttons: Optional[Dict[str, bool]] = None,
        selectbox: Any = None,
    ) -> None:
        # ``session_state`` mirrors Streamlit's mutable per-session dict.
        self.session_state: Dict[str, Any] = dict(session or {})
        # widget key/label -> value the widget yields (and writes to session).
        self.inputs = dict(inputs or {})
        # button label -> whether it was "clicked" this run.
        self.buttons = dict(buttons or {})
        # value to return from ``selectbox`` (defaults to the first option).
        self.selectbox_value = selectbox

        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.successes: List[str] = []
        self.subheaders: List[str] = []
        self.captions: List[str] = []
        self.titles: List[str] = []
        self.markdowns: List[Any] = []
        self.jsons: List[Any] = []
        self.sidebar_successes: List[str] = []
        self.sidebar_captions: List[str] = []
        self.rerun_called = 0

        self.sidebar = _FakeContainer(self, kind="sidebar")

    # layout / text -------------------------------------------------------
    def set_page_config(self, *args: object, **kwargs: object) -> None:
        pass

    def title(self, text: str, *args: object, **kwargs: object) -> None:
        self.titles.append(text)

    def subheader(self, text: str, *args: object, **kwargs: object) -> None:
        self.subheaders.append(text)

    def caption(self, text: str, *args: object, **kwargs: object) -> None:
        self.captions.append(text)

    def markdown(self, text: str, *args: object, **kwargs: object) -> None:
        self.markdowns.append(text)

    def json(self, data: Any, *args: object, **kwargs: object) -> None:
        self.jsons.append(data)

    def error(self, text: str, *args: object, **kwargs: object) -> None:
        self.errors.append(text)

    def warning(self, text: str, *args: object, **kwargs: object) -> None:
        self.warnings.append(text)

    def success(self, text: str, *args: object, **kwargs: object) -> None:
        self.successes.append(text)

    # inputs --------------------------------------------------------------
    def text_input(
        self, label: str, *args: object, key: Optional[str] = None, **kwargs: object
    ) -> str:
        return self._value(label, key)

    def text_area(
        self, label: str, *args: object, key: Optional[str] = None, **kwargs: object
    ) -> str:
        return self._value(label, key)

    def _value(self, label: str, key: Optional[str]) -> str:
        # Real Streamlit persists widget values in session_state under ``key``.
        if key is not None:
            if key in self.inputs:
                self.session_state[key] = self.inputs[key]
            return self.session_state.get(key, "")
        return self.inputs.get(label, "")

    def selectbox(
        self, label: str, options: Any, *args: object, **kwargs: object
    ) -> Any:
        if self.selectbox_value is not None:
            return self.selectbox_value
        options = list(options)
        return options[0] if options else None

    def button(self, label: str, *args: object, **kwargs: object) -> bool:
        return bool(self.buttons.get(label, False))

    def spinner(self, *args: object, **kwargs: object) -> _FakeContainer:
        return _FakeContainer(self)

    def expander(self, *args: object, **kwargs: object) -> _FakeContainer:
        return _FakeContainer(self)

    def rerun(self) -> None:
        self.rerun_called += 1


__all__ = ["FakeStreamlit"]
