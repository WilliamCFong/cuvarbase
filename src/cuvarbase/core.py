from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
from .utils import gaussian_window, tophat_window, get_autofreqs
from .gpu import current_gpu
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


class GPUAsyncProcess(object):
    def __init__(self, *args, **kwargs):
        self.reader = kwargs.get('reader', None)
        self.nstreams = kwargs.get('nstreams', None)
        self.function_kwargs = kwargs.get('function_kwargs', {})
        self.gpu = kwargs.get('gpu') or current_gpu()
        self.device = self.gpu.device_id
        self.streams = []
        self.gpu_data = []
        self.results = []
        self._adjust_nstreams = self.nstreams is None
        if self.nstreams is not None:
                self._create_streams(self.nstreams)
        self.prepared_functions = {}
        self.module = None
        self.gpu.track(self)

    def _create_streams(self, n):
        for i in range(n):
            self.streams.append(cuda.Stream())

    def close(self):
        """Release GPU resources owned by this process.

        Called automatically when the enclosing ``initialize_gpu`` block
        exits. Synchronizes all streams, drops references to compiled
        modules and prepared functions.
        """
        for stream in self.streams:
            try:
                stream.synchronize()
            except cuda.Error:
                pass
        self.streams = []
        self.prepared_functions = {}
        self.module = None
        self.gpu_data = []
        self.results = []

    def _compile_and_prepare_functions(self):
        raise NotImplementedError()

    def run(self, *args, **kwargs):
        raise NotImplementedError()

    def finish(self):
        """ synchronize all active streams """
        for i, stream in enumerate(self.streams):
            stream.synchronize()

    def batched_run(self, data, batch_size=10, **kwargs):
        """ Run your data in batches (avoids memory problems) """
        nsubmit = 0
        results = []
        while nsubmit < len(data):
            batch = []
            while len(batch) < batch_size and nsubmit < len(data):
                batch.append(data[nsubmit])
                nsubmit += 1

            res = self.run(batch, **kwargs)
            self.finish()
            results.extend(res)

        return results
