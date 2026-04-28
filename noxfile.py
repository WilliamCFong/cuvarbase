from __future__ import annotations

import nox

nox.options.sessions = ["tests"]
nox.options.reuse_existing_virtualenvs = False


@nox.session
def tests(session: nox.Session) -> None:
    """Run the pytest suite. Extra args are forwarded: `nox -s tests -- -k bls`.

    Wheel-first install (supported Linux distros only). pycuda and nfft are
    sdist-only on PyPI, so they are exempted via `--no-binary` and built from
    source (pycuda against the local CUDA toolchain; nfft is pure Python +
    Cython but distributed as sdist).
    """
    session.install(
        "--only-binary", ":all:",
        "--no-binary", "pycuda,nfft",
        "-e", ".[test]",
    )
    session.run("pytest", *session.posargs)
