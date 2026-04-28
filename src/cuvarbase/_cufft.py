"""Thin cuFFT wrapper replacing the scikit-cuda surface used by cuvarbase.

cuvarbase consumed exactly two skcuda.fft APIs from the adjoint NFFT:
``Plan(n, in_dtype, out_dtype, stream=...)`` and
``ifft(in_arr, out_arr, plan)``. This module reproduces just those two,
backed by the official NVIDIA bindings in ``nvmath.bindings.cufft``.

The shapes here intentionally mirror the old skcuda call sites so that
``src/cuvarbase/cunfft.py`` and ``tests/test_nfft.py`` only need to swap
the import line. PyCUDA gpuarrays are read via their ``.ptr`` attribute
(integer device pointer) and PyCUDA streams via ``.handle``.
"""

from __future__ import annotations

import numpy as np

from nvmath.bindings import cufft as _cufft

CUFFT_FORWARD = -1
CUFFT_INVERSE = 1

_TYPE_MAP = {
    (np.complex64, np.complex64): _cufft.Type.C2C,
    (np.complex128, np.complex128): _cufft.Type.Z2Z,
}


class Plan:
    """1D cuFFT plan handle.

    Mirrors the subset of ``skcuda.fft.Plan`` that cuvarbase actually used:
    a scalar ``shape``, a complex in/out dtype pair, and an optional PyCUDA
    stream. Multi-dim and batched plans are out of scope here.
    """

    def __init__(self, shape, in_dtype, out_dtype, batch=1, stream=None):
        n = shape if isinstance(shape, (int, np.integer)) else int(np.prod(shape))
        in_t = np.dtype(in_dtype).type
        out_t = np.dtype(out_dtype).type
        try:
            cufft_type = _TYPE_MAP[(in_t, out_t)]
        except KeyError as e:
            raise NotImplementedError(
                f"unsupported cuFFT type pair: {in_t} -> {out_t}"
            ) from e

        self._type = int(cufft_type)
        self._handle = _cufft.create()
        try:
            _cufft.plan1d(self._handle, int(n), self._type, int(batch))
            if stream is not None:
                _cufft.set_stream(self._handle, int(stream.handle))
        except Exception:
            _cufft.destroy(self._handle)
            self._handle = None
            raise

    def __del__(self):
        h = getattr(self, "_handle", None)
        if h is not None:
            try:
                _cufft.destroy(h)
            except Exception:
                # Best-effort cleanup; suppress at interpreter shutdown.
                pass


def ifft(in_arr, out_arr, plan):
    """Inverse FFT. Operates in place when ``in_arr is out_arr``."""
    if plan._type == int(_cufft.Type.C2C):
        _cufft.exec_c2c(plan._handle, in_arr.ptr, out_arr.ptr, CUFFT_INVERSE)
    elif plan._type == int(_cufft.Type.Z2Z):
        _cufft.exec_z2z(plan._handle, in_arr.ptr, out_arr.ptr, CUFFT_INVERSE)
    else:
        raise NotImplementedError(f"unsupported plan type: {plan._type}")
