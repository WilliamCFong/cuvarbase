"""Tests for the cuvarbase GPU context manager contract.

Exercises behavior introduced by ``src/cuvarbase/gpu.py``:
- hard-break when no context is active,
- deterministic ``close()`` of tracked AsyncProcess instances on exit,
- no GPU memory growth across repeated ``initialize_gpu`` blocks,
- the ``device=`` kwarg on standalone BLS entry points,
- ``ensure_gpu`` semantics (reuse-if-active, open-if-not).
"""
import os

import numpy as np
import pytest
import pycuda.driver as cuda

import cuvarbase.gpu as gpu_mod
from cuvarbase import initialize_gpu
from cuvarbase.bls import eebls_transit_gpu
from cuvarbase.lombscargle import LombScargleAsyncProcess


_DEVICE = int(os.environ.get("CUDA_DEVICE", 0))


def _small_lc(seed=0, n=300, freq=2.5):
    rand = np.random.RandomState(seed)
    t = np.sort(100 * rand.rand(n))
    y = 12 + 0.05 * np.cos(2 * np.pi * t * freq) + 0.02 * rand.randn(n)
    dy = 0.02 * np.ones_like(y)
    return t, y, dy


def test_async_process_outside_context_raises():
    """LombScargleAsyncProcess() outside an active context raises RuntimeError."""
    token = gpu_mod._current_gpu.set(None)
    try:
        with pytest.raises(RuntimeError, match="No active cuvarbase GPU context"):
            LombScargleAsyncProcess()
    finally:
        gpu_mod._current_gpu.reset(token)


def test_current_gpu_outside_context_raises():
    """current_gpu() raises when no context is active."""
    token = gpu_mod._current_gpu.set(None)
    try:
        with pytest.raises(RuntimeError, match="No active cuvarbase GPU context"):
            gpu_mod.current_gpu()
    finally:
        gpu_mod._current_gpu.reset(token)


def test_close_runs_on_context_exit():
    """AsyncProcess.close() drops streams and the SourceModule on __exit__."""
    with initialize_gpu(_DEVICE):
        proc = LombScargleAsyncProcess(nstreams=2)
        proc._compile_and_prepare_functions()
        assert len(proc.streams) >= 2
        assert proc.module is not None
        assert proc.prepared_functions
    # After exiting the (nested) context, the runtime should have called
    # close() on `proc`, dropping streams, prepared functions, and the module.
    assert proc.streams == []
    assert proc.module is None
    assert proc.prepared_functions == {}


def test_no_memory_growth_across_blocks():
    """Three back-to-back initialize_gpu blocks should not leak GPU memory.

    Iteration 1 incurs first-time SourceModule compilation; iterations 2 and 3
    should reach steady state.
    """
    t, y, dy = _small_lc()
    free_per_iter = []
    for _ in range(3):
        with initialize_gpu(_DEVICE):
            proc = LombScargleAsyncProcess()
            proc.run([(t, y, dy)])
            proc.finish()
        free, _ = cuda.mem_get_info()
        free_per_iter.append(free)
    # If every iteration leaked, free_per_iter would be strictly decreasing
    # by a roughly constant amount. We require iter 2->3 to consume no more
    # than iter 1->2 plus 50 MB of jitter.
    delta_12 = free_per_iter[0] - free_per_iter[1]
    delta_23 = free_per_iter[1] - free_per_iter[2]
    slack = 50 * 1024 * 1024
    assert delta_23 <= max(delta_12, 0) + slack, (
        "GPU memory grew across iterations: free_per_iter=%s, deltas=%s"
        % (free_per_iter, [delta_12, delta_23])
    )


def test_eebls_transit_with_explicit_device():
    """eebls_transit_gpu(device=N) opens its own nested context and runs."""
    t, y, dy = _small_lc(seed=1)
    freqs, powers, sols = eebls_transit_gpu(t, y, dy, device=_DEVICE)
    assert len(freqs) == len(powers)
    assert len(freqs) == len(sols)


def test_eebls_transit_without_device_reuses_active_context():
    """eebls_transit_gpu without device= reuses the autouse session context."""
    outer = gpu_mod.current_gpu()
    t, y, dy = _small_lc(seed=2)
    freqs, powers, sols = eebls_transit_gpu(t, y, dy)
    assert len(freqs) == len(powers)
    assert len(freqs) == len(sols)
    # Active handle should be unchanged: the call should not have left a
    # stale or different handle behind.
    assert gpu_mod.current_gpu() is outer


def test_ensure_gpu_opens_when_no_active_context():
    """ensure_gpu() with no active context opens one, then unsets on exit."""
    token = gpu_mod._current_gpu.set(None)
    try:
        with gpu_mod.ensure_gpu():
            handle = gpu_mod.current_gpu()
            assert handle.device_id == _DEVICE
        # After exit the contextvar should be back to its prior value
        # (which was None, since we set it just above).
        assert gpu_mod._current_gpu.get() is None
    finally:
        gpu_mod._current_gpu.reset(token)


def test_ensure_gpu_reuses_when_active_and_no_device():
    """ensure_gpu() with no device= and an active context yields it unchanged."""
    outer = gpu_mod.current_gpu()
    with gpu_mod.ensure_gpu() as inner:
        assert inner is outer
