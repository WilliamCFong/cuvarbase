from __future__ import annotations

import nox

nox.options.sessions = ["tests"]
nox.options.reuse_existing_virtualenvs = False


@nox.session
def tests(session: nox.Session) -> None:
    """Run the pytest suite. Extra args are forwarded: `nox -s tests -- -k bls`.

    Wheel-first install (supported Linux distros only). pycuda has no wheels on
    PyPI (all 100+ releases are sdist), so it is exempted via `--no-binary` and
    built from source against the local CUDA toolchain.
    """
    session.install(
        "--only-binary", ":all:",
        "--no-binary", "pycuda",
        "-e", ".[test]",
    )
    session.run("pytest", *session.posargs)
