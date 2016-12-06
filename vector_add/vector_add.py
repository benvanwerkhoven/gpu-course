#!/usr/bin/env python
from __future__ import print_function

import numpy
import pycuda.driver as drv
from pycuda.compiler import SourceModule

def vector_add_example(context):

    #create input data
    n = numpy.int32(5e7)
    a = numpy.random.randn(n).astype(numpy.float32)
    b = numpy.random.randn(n).astype(numpy.float32)
    c = numpy.zeros_like(b)

    #move data to the GPU
    args = [c, a, b]
    gpu_args = []
    for arg in args:
        gpu_args.append(drv.mem_alloc(arg.nbytes))
        drv.memcpy_htod(gpu_args[-1], arg)
    gpu_args.append(n)

    #read kernel into string
    with open('vector_add.cu', 'r') as f:
        kernel_string = f.read()

    #get compute capability for compiling CUDA kernels
    devprops = { str(k): v for (k, v) in context.get_device().get_attributes().items() }
    cc = str(devprops['COMPUTE_CAPABILITY_MAJOR']) + str(devprops['COMPUTE_CAPABILITY_MINOR'])

    #compile the kernel
    vector_add = SourceModule(kernel_string, arch='compute_' + cc, code='sm_' + cc,
                    cache_dir=False, no_extern_c=True).get_function("vec_add_kernel")

    #launch the kernel
    threads = (1024, 1, 1)
    grid = (int(numpy.ceil(n/float(threads[0]))), 1, 1)
    vector_add(*gpu_args, block=threads, grid=grid, stream=None, shared=0)

    #copy output data back from GPU
    drv.memcpy_dtoh(c, gpu_args[0])

    #compare output with reference
    correct = numpy.allclose(c, a+b, atol=1e-6)
    if not correct:
        print("TEST FAILED!")
        print(c)
        print(a+b)
    else:
        print("TEST PASSED!")


if __name__ == "__main__":

    #init pycuda
    drv.init()
    context = drv.Device(0).make_context()
    try:
        vector_add_example(context)
    finally:
        context.pop()
