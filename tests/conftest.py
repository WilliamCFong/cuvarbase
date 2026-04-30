import os

import pytest

import cuvarbase


@pytest.fixture(scope="session", autouse=True)
def _gpu_session():
    device_id = int(os.environ.get("CUDA_DEVICE", 0))
    with cuvarbase.initialize_gpu(device_id) as gpu:
        yield gpu
