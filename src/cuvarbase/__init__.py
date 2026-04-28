__version__ = "0.3.0"

# import pycuda.autoinit causes problems when running e.g. FFT
import pycuda.autoprimaryctx  # noqa: E402, F401
