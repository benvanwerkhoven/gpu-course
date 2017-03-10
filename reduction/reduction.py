#!/usr/bin/env python
from __future__ import print_function

from timeit import default_timer as timer
import numpy
import pycuda.driver as drv
from pycuda.compiler import SourceModule

def reduction_exercise(context):

    block_size_x = 256
    num_blocks = 1024

    #create input data
    n = numpy.int32(5e7)
    in_array = numpy.random.randn(n).astype(numpy.float32) + 1.0
    out_array = numpy.zeros(num_blocks).astype(numpy.float32)

    #measure CPU time
    start = timer()
    npsum = numpy.sum(in_array)
    end = timer()
    print("numpy.sum took", (end-start)*1000.0, "ms")

    #move data to the GPU
    args = [out_array, in_array, n]
    gpu_args = []
    for arg in args:
        gpu_args.append(drv.mem_alloc(arg.nbytes))
        drv.memcpy_htod(gpu_args[-1], arg)
    gpu_args.append(n)

    #read kernel into string
    with open('reduction.cu', 'r') as f:
        kernel_string = f.read()

    #get compute capability for compiling CUDA kernels
    devprops = { str(k): v for (k, v) in context.get_device().get_attributes().items() }
    cc = str(devprops['COMPUTE_CAPABILITY_MAJOR']) + str(devprops['COMPUTE_CAPABILITY_MINOR'])

    #compile the kernel
    reduce_kernel = SourceModule(kernel_string, arch='compute_' + cc, code='sm_' + cc,
                    cache_dir=False, no_extern_c=True).get_function("reduce_kernel")

    #launch the kernel
    threads = (block_size_x, 1, 1)
    grid = (num_blocks, 1, 1)

    context.synchronize()
    start = drv.Event()
    end = drv.Event()
    start.record()
    reduce_kernel(*gpu_args, block=threads, grid=grid, stream=None, shared=0)
    end.record()
    context.synchronize()

    print("reduction_kernel took", end.time_since(start), "ms.")

    #copy output data back from GPU
    gpu_sum = numpy.zeros(1).astype(numpy.float32)
    drv.memcpy_dtoh(gpu_sum, gpu_args[0])

    #compare output with reference
    correct = numpy.absolute(npsum - gpu_sum) < 10.0
    if not correct:
        print("TEST FAILED!")
        print(npsum)
        print(gpu_sum)
    else:
        print("TEST PASSED!")


if __name__ == "__main__":

    #init pycuda
    drv.init()
    context = drv.Device(0).make_context()
    try:
        reduction_exercise(context)
    finally:
        context.pop()
