cuvarbase
=========

.. image:: https://badge.fury.io/py/cuvarbase.svg
    :target: https://badge.fury.io/py/cuvarbase


Active development happening! see `v1.0` branch
-----------------------------------------------

John Hoffman
(c) 2017

``cuvarbase`` is a Python library that uses `PyCUDA <https://mathema.tician.de/software/pycuda/>`_ to implement several time series tools used in astronomy on GPUs.

See the `documentation <https://johnh2o2.github.io/cuvarbase/>`_.

This project is under active development, and currently includes implementations of

- Generalized `Lomb Scargle <https://arxiv.org/abs/0901.2573>`_ periodogram
- Box-least squares (`BLS <http://adsabs.harvard.edu/abs/2002A%26A...391..369K>`_ )
- Non-equispaced fast Fourier transform (adjoint operation) (`NFFT paper <http://epubs.siam.org/doi/abs/10.1137/0914081>`_)
- Conditional entropy period finder (`CE <http://adsabs.harvard.edu/abs/2013MNRAS.434.2629G>`_)
- Phase dispersion minimization (`PDM2 <http://www.stellingwerf.com/rfs-bin/index.cgi?action=PageView&id=29>`_)
	- Currently operational but minimal unit testing or documentation (yet)

Hopefully future developments will have

- (Weighted) wavelet transforms
- Spectrograms (for PDM and GLS)
- Multiharmonic extensions for GLS


Dependencies
------------

- `PyCUDA <https://mathema.tician.de/software/pycuda/>`_ **<-essential**
- `nvmath-python <https://docs.nvidia.com/cuda/nvmath-python/>`_ **<-also essential**
	- NVIDIA's official Python bindings for the CUDA math libraries; we use ``nvmath.bindings.cufft`` for the FFT inside the adjoint NFFT
- `matplotlib <https://matplotlib.org/>`_ (for plotting utilities)
- `nfft <https://github.com/jakevdp/nfft>`_ (for unit testing)
- `astropy <http://www.astropy.org/>`_ (for unit testing)


Using multiple GPUs
-------------------

GPU initialization is explicit. Wrap any code that uses
``cuvarbase`` in an ``initialize_gpu`` context manager:

.. code:: python

    import cuvarbase
    from cuvarbase.lombscargle import LombScargleAsyncProcess

    with cuvarbase.initialize_gpu(0) as gpu:
        proc = LombScargleAsyncProcess()
        results = proc.run([(t, y, dy)])

On exit, cuvarbase synchronizes the active context and releases
streams, cuFFT plans, and compiled modules owned by AsyncProcess /
Memory instances created in the block.

If ``initialize_gpu()`` is called with no argument, it reads
``int(os.environ['CUDA_DEVICE'])`` (defaulting to 0) so existing
``CUDA_DEVICE=1 python script.py`` invocations keep working as long
as the script is wrapped in the context manager.

Instantiating a GPU class outside an ``initialize_gpu`` block raises
``RuntimeError`` -- there is no implicit fallback.

The standalone BLS entry points (``eebls_gpu``, ``eebls_gpu_fast``,
``eebls_gpu_custom``, ``eebls_transit_gpu``) accept an optional
``device=`` keyword and open the context themselves, so a one-shot
call can skip the ``with`` block:

.. code:: python

    from cuvarbase.bls import eebls_transit_gpu
    freqs, powers, sols = eebls_transit_gpu(t, y, dy, device=0)

If ``device`` is omitted and a context is already active, the call
reuses it. Otherwise it opens one on ``int(os.environ['CUDA_DEVICE'])``
(defaulting to 0).

If anyone is interested in implementing a multi-device load-balancing
solution, they are encouraged to do so. Nested ``initialize_gpu``
blocks on different devices work, but state created in an outer block
must not be invoked while a nested block on a different device is
active.
