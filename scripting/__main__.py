"""Allow ``python -m scripting`` to invoke the CLI."""

from .cli import main

if __name__ == "__main__":
    # Exit code propagates so shell scripts can react to failures.
    raise SystemExit(main())