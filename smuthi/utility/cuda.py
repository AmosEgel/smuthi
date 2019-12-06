import numpy as np
import warnings
import sys

use_gpu = False

default_blocksize = 128


def enable_gpu(enable=True):
    """Sets the use_gpu flag to enable/disable the use of CUDA kernels.

    Args:
        enable (bool): Set use_gpu flag to this value (default=True).
    """
    # todo: do gpuarrays require cuda toolkit? otherwise distinguish if only gpuarrays work but no kernel compilation
    global use_gpu

    if use_gpu == enable:
        return

    if not enable:
        use_gpu = False
        sys.stdout.write('Disabling GPU usage.\n')
        sys.stdout.flush()
        return

    try:
        import pycuda.autoinit
        import pycuda.driver as drv
        from pycuda import gpuarray
        from pycuda.compiler import SourceModule
        import pycuda.cumath
        pycuda_available = True

        current_module = sys.modules[__name__]
        current_module.gpuarray = gpuarray
        current_module.SourceModule = SourceModule
    except ImportError:
        pycuda_available = False

    if not pycuda_available:
        warnings.warn("Unable to import PyCuda - fall back to CPU mode")
        use_gpu = False
        sys.stdout.write('Disabling GPU usage.\n')
        sys.stdout.flush()
    else:
        try:
            test_fun = SourceModule("""
                __global__ void test_kernel(float *x)
                {
                    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
                    x[i] = 2 * i;
                }""").get_function("test_kernel")
            x = gpuarray.to_gpu(np.zeros(128, dtype=np.float32))
            test_fun(x, block=(128,1,1), grid=(1,1))
            assert np.allclose(x.get(), 2 * np.arange(128))
            use_gpu = True
            sys.stdout.write('Enabling GPU usage.\n')
            sys.stdout.flush()
            return

        except Exception as e:
            warnings.warn("CUDA test kernel failed - fall back to CPU mode")
            print(e)
            use_gpu = False
            sys.stdout.write('Disabling GPU usage.\n')
            sys.stdout.flush()
