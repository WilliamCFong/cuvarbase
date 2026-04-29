__version__ = "0.3.0"

# Ensure -allow-unsupported-compiler reaches every nvcc invocation, including
# PyCUDA's own elementwise / reduction kernels (gpuarray.zeros().fill() etc.)
# that we can't reach via SourceModule options. NVCC_APPEND_FLAGS is read by
# nvcc itself, so it covers callers we don't control. Idempotent and
# preserves any user-set flags.
import os as _os  # noqa: E402

_nvcc_append = _os.environ.get("NVCC_APPEND_FLAGS", "")
if "-allow-unsupported-compiler" not in _nvcc_append:
    _os.environ["NVCC_APPEND_FLAGS"] = (
        _nvcc_append + " -allow-unsupported-compiler"
    ).strip()
del _nvcc_append, _os

from .gpu import initialize_gpu, current_gpu  # noqa: E402, F401
