from __future__ import annotations

import nox

nox.options.sessions = ["tests"]
nox.options.reuse_existing_virtualenvs = False


@nox.session
def tests(session: nox.Session) -> None:
    """Run the pytest suite. Extra args are forwarded: `nox -s tests -- -k bls`."""
    session.install("-e", ".[test]")
    session.run("pytest", *session.posargs)
