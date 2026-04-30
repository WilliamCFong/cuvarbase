# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`cuvarbase` is a PyCUDA-backed library of GPU period-finding / time-series tools used in astronomy: generalized Lomb-Scargle, Box-Least-Squares (BLS), Conditional Entropy (CE), Phase Dispersion Minimization (PDM2), and an adjoint NFFT used by GLS. Each algorithm is implemented as a CUDA kernel (`src/cuvarbase/kernels/*.cu`) plus a Python driver (`src/cuvarbase/<name>.py`). Tests live at the top-level `tests/` directory and import the package as `cuvarbase.*` (requires `pip install -e .`).

Active development is on the `v1.0` branch (per README). `master` is the historical line.

## Common commands

The project uses `pyproject.toml` (PEP 621, setuptools backend) and `nox` for task running.

```bash
# Install for development with test extras
pip install -e ".[test]"

# Run the full test suite via nox (creates an isolated venv, installs ".[test]", runs pytest)
nox            # alias for `nox -s tests`
nox -s tests -- -k bls -x       # forward args to pytest

# Or run pytest directly if the venv is already set up
pytest                           # picks up testpaths=tests from pyproject
pytest tests/test_bls.py
pytest tests/test_lombscargle.py::<name>

# Choose which GPU to run on (script must wrap work in `initialize_gpu`)
CUDA_DEVICE=1 python script.py

# Build Sphinx docs locally
cd docs && make html   # output: docs/build/html/index.html
```

Tests require a working CUDA toolchain plus `pytest`, `nfft`, `astropy`, and `matplotlib` — they actually launch CUDA kernels, so they will fail without a GPU.

## Architecture

### The `GPUAsyncProcess` pattern (`src/cuvarbase/core.py`)

Every algorithm subclasses `GPUAsyncProcess`. The base class owns CUDA streams, compiled kernel handles (`prepared_functions`), and a list-of-light-curves `batched_run` that chunks data to avoid OOM. Subclasses must implement `_compile_and_prepare_functions` (load `.cu` from `kernels/`, compile via `pycuda.compiler.SourceModule`, prepare argument signatures) and `run`.

The user-facing class for each algorithm is `<Name>AsyncProcess` (`LombScargleAsyncProcess`, `BLSAsyncProcess`-style entry points like `eebls_gpu`/`eebls_gpu_fast`/`eebls_transit_gpu`, `ConditionalEntropyAsyncProcess`, `PDMAsyncProcess`, `NFFTAsyncProcess`). Most also expose a `precompute(...)` path that allocates and uploads once when many light curves share a frequency grid, plus a `large_run`/`batched_run_constant_nfreq` path for grids that exceed device memory.

### Memory containers

Each module defines a `<Name>Memory` class (`LombScargleMemory`, `BLSMemory`, `ConditionalEntropyMemory`, `NFFTMemory`) holding all per-light-curve GPU buffers, the CUDA stream, and dtype settings (`use_double` toggles `float32`/`complex64` vs `float64`/`complex128`). The driver function (`lomb_scargle_async`, `nfft_adjoint_async`, etc.) operates on a `Memory` instance plus the dict of prepared kernels — separating allocation from execution is what enables stream pipelining and `precompute`.

### Kernel compilation

`utils._module_reader` reads a `.cu` file and substitutes a `//{CPP_DEFS}` marker with `#define`s, so block size, dtype, and feature flags are baked in at compile time. `utils.find_kernel(name)` resolves `src/cuvarbase/kernels/<name>.cu` via `importlib.resources` (works from an installed wheel). When changing kernel constants (block size, double precision, harmonic count), the calling Python must recompile — different settings produce different `SourceModule` objects.

### NFFT → Lomb-Scargle dependency

`lombscargle.py` builds on `cunfft.py`: GLS computes adjoint NFFTs of `y*w` and `w`, then combines them. `LombScargleAsyncProcess` instantiates an internal `NFFTAsyncProcess`. If `use_fft=False` the GLS path falls back to direct trig sums and skips NFFT GPU allocation. The `k0` machinery (`get_k0`, `check_k0`) requires the frequency grid to be a uniform multiple of `df = freqs[1] - freqs[0]` starting at `k0 * df`.

### Context initialization

GPU initialization is explicit via the `initialize_gpu(device_id)` context manager in `src/cuvarbase/gpu.py`. It calls `pycuda.driver.Device.retain_primary_context()` (sharing the primary context with nvmath/cuFFT, CuPy, Torch, etc.), pushes it for the duration of the block, and pops + detaches on exit. The active handle is exposed via a contextvar; AsyncProcess and Memory constructors read it through `current_gpu()` and register themselves via `gpu.track(self)` so their `close()` runs on context exit (releasing streams, cuFFT plans, and SourceModule references). Importing the package no longer initializes a context — instantiating a GPU class outside an `initialize_gpu` block raises `RuntimeError`. `CUDA_DEVICE` is honored only as the default `device_id` when `initialize_gpu()` is called with no argument.

Don't reintroduce `pycuda.autoprimaryctx` or `pycuda.autoinit` at module scope — the previous reliance on import-time side effects is what this refactor removed (and `autoinit` triggers the `cuFuncSetBlockShape` error alongside scikit-cuda's FFT runtime; see CHANGELOG 0.2.5).

## Notes for changes

- `pycuda 2024.1.2` is excluded in `pyproject.toml` (`pycuda>=2017.1.1,!=2024.1.2`) — keep the pin if touching deps.
- Version is dynamic: setuptools AST-parses `__version__` from `src/cuvarbase/__init__.py`. Keep `__version__ = "..."` as the first line of that file (a plain string literal) so the parser doesn't fall back to *importing* the package — importing now pulls in `pycuda.driver` via `gpu.py`, which would still break version reads on machines without CUDA.
- The `future`/`builtins` imports throughout are stale Py2/Py3 compat. The package targets Py 3.9+ (`importlib.resources.files`, PEP 604 generics) and the future shims can be removed.
- `publish_docs.sh` rebuilds the `gh-pages` branch from `master`; it's author-specific (johnh2o2) and shouldn't be run casually.
