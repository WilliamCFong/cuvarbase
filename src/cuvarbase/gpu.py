"""
Context-managed GPU initialization for cuvarbase.

Use ``initialize_gpu(device_id)`` as a context manager to bind a CUDA
primary context to a block of code:

    >>> import cuvarbase
    >>> from cuvarbase.lombscargle import LombScargleAsyncProcess
    >>> with cuvarbase.initialize_gpu(0) as gpu:
    ...     proc = LombScargleAsyncProcess()
    ...     results = proc.run([(t, y, dy)])

On exit, the active context is synchronized, every tracked
``AsyncProcess``/``Memory`` instance has its ``close()`` called (releasing
streams, cuFFT plans, and SourceModule references), and the primary context
is popped and detached.

The active handle is exposed via the ``cuvarbase_current_gpu`` context
variable; AsyncProcess and Memory constructors read it via
``current_gpu()``. There is no implicit fallback: instantiating a GPU
class outside an active ``initialize_gpu`` block raises ``RuntimeError``.

If ``device_id`` is omitted, ``int(os.environ['CUDA_DEVICE'])`` is used,
defaulting to 0.

Notes
-----
- ``Device.retain_primary_context()`` is used (not ``make_context()``) so
  cuvarbase shares the primary context with cuFFT/nvmath, CuPy, Torch,
  etc. Plans and allocations belong to the same context across libraries.
- ``SourceModule`` has no explicit free; ``close()`` drops the Python
  reference so its CUmodule unloads when the GC runs. Holding external
  references to prepared functions past the ``with`` block is unsafe.
- Multi-device works via nested ``initialize_gpu`` blocks, but streams /
  modules created in an outer block must not be invoked while a nested
  block on a different device is active.
"""
import contextvars
import functools
import os
import warnings
import weakref
from contextlib import contextmanager
from typing import Optional

import pycuda.driver as cuda


_current_gpu: "contextvars.ContextVar[Optional[GpuHandle]]" = (
    contextvars.ContextVar("cuvarbase_current_gpu", default=None)
)


class GpuHandle:
    """Active GPU binding yielded by :func:`initialize_gpu`.

    Attributes
    ----------
    device_id : int
        Ordinal of the bound device.
    device : pycuda.driver.Device
        The bound device.
    context : pycuda.driver.Context
        The retained primary context (currently pushed).
    """

    def __init__(self, device_id, device, context):
        self.device_id = device_id
        self.device = device
        self.context = context
        self._tracked = weakref.WeakSet()
        self._closed = False

    def track(self, obj):
        """Register an object for cleanup on context-manager exit.

        Objects with a ``close()`` method will have it called when the
        enclosing ``initialize_gpu`` block exits.
        """
        self._tracked.add(obj)
        return obj

    def new_stream(self):
        return cuda.Stream()


def current_gpu():
    """Return the active :class:`GpuHandle`.

    Raises
    ------
    RuntimeError
        If called outside an ``initialize_gpu`` block.
    """
    h = _current_gpu.get()
    if h is None:
        raise RuntimeError(
            "No active cuvarbase GPU context. Wrap your code in "
            "`with cuvarbase.initialize_gpu(device_id) as gpu: ...`."
        )
    return h


@contextmanager
def ensure_gpu(device_id=None):
    """Ensure a cuvarbase GPU context is active for the duration.

    - ``device_id=None`` and a context is active: yield it unchanged.
    - ``device_id=None`` and no context is active: open one on
      ``int(os.environ.get('CUDA_DEVICE', 0))``.
    - ``device_id`` is given: always open a fresh context on that
      device, nesting under any existing context.

    Use this when you want a function to "just work" whether the caller
    has already wrapped it in ``initialize_gpu`` or not.
    """
    if device_id is None and _current_gpu.get() is not None:
        yield _current_gpu.get()
        return
    with initialize_gpu(device_id) as h:
        yield h


def with_active_gpu(func):
    """Decorator: run ``func`` inside :func:`ensure_gpu`.

    The wrapped function gains an optional ``device=`` keyword argument.
    If omitted (or ``None``) and a context is already active, the body
    runs in that context. Otherwise a new context is opened, defaulting
    to ``CUDA_DEVICE`` (or 0).
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        device = kwargs.pop('device', None)
        with ensure_gpu(device):
            return func(*args, **kwargs)
    return wrapper


@contextmanager
def initialize_gpu(device_id=None):
    """Context manager binding a CUDA primary context to a block of code.

    Parameters
    ----------
    device_id : int, optional
        GPU ordinal. Defaults to ``int(os.environ['CUDA_DEVICE'])``,
        falling back to 0.

    Yields
    ------
    GpuHandle
        Active GPU binding.
    """
    cuda.init()
    if device_id is None:
        device_id = int(os.environ.get("CUDA_DEVICE", 0))
    dev = cuda.Device(device_id)
    ctx = dev.retain_primary_context()
    ctx.push()
    handle = GpuHandle(device_id, dev, ctx)
    token = _current_gpu.set(handle)
    try:
        yield handle
    finally:
        try:
            ctx.synchronize()
        except cuda.Error:
            pass
        for obj in list(handle._tracked):
            close = getattr(obj, "close", None)
            if close is None:
                continue
            try:
                close()
            except Exception as exc:
                warnings.warn(
                    "cuvarbase: error closing %r on context exit: %s"
                    % (obj, exc)
                )
        handle._closed = True
        _current_gpu.reset(token)
        ctx.pop()
        ctx.detach()
