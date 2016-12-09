#!/usr/bin/env python
from __future__ import print_function

import numpy
import pycuda.driver as drv
from pycuda.compiler import SourceModule

def convolution_example(context):

    #create input data
    image_width = 1024
    image_height = 1024
    filter_width = 17
    filter_height = 17
    input_width = image_width + 2*(filter_width//2)
    input_height = image_height + 2*(filter_height//2)

    input_image = numpy.random.randn(input_width, input_height).astype(numpy.float32)
    filter = numpy.random.randn(filter_width, filter_height).astype(numpy.float32)
    output_image = numpy.zeros((image_width, image_height), dtype=numpy.float32)

    #move data to the GPU
    args = [output_image, input_image, filter]
    gpu_args = []
    for arg in args:
        gpu_args.append(drv.mem_alloc(arg.nbytes))
        drv.memcpy_htod(gpu_args[-1], arg)

    #read kernel into string
    with open('convolution.cu', 'r') as f:
        kernel_string = f.read()

    #get compute capability for compiling CUDA kernels
    devprops = { str(k): v for (k, v) in context.get_device().get_attributes().items() }
    cc = str(devprops['COMPUTE_CAPABILITY_MAJOR']) + str(devprops['COMPUTE_CAPABILITY_MINOR'])

    #compile the kernels
    module = SourceModule(kernel_string, arch='compute_' + cc, code='sm_' + cc,
                    cache_dir=False, no_extern_c=True)
    convolution = module.get_function("convolution_kernel")
    convolution_naive = module.get_function("convolution_kernel_naive")

    #setup thread block sizes
    threads = (32, 16, 1)
    grid = (int(numpy.ceil(image_width/float(threads[0]))), int(numpy.ceil(image_height/float(threads[1]))), 1)

    #compute reference using naive kernel
    reference = numpy.zeros_like(output_image)
    start = drv.Event()
    end = drv.Event()
    context.synchronize()
    start.record()
    convolution_naive(*gpu_args, block=threads, grid=grid, stream=None, shared=0)
    end.record()
    context.synchronize()
    print("convolution_kernel_naive took", end.time_since(start), "ms.")
    drv.memcpy_dtoh(reference, gpu_args[0])
    drv.memcpy_htod(gpu_args[0], output_image)

    #launch the kernel
    context.synchronize()
    start.record()
    convolution(*gpu_args, block=threads, grid=grid, stream=None, shared=0)
    end.record()
    context.synchronize()
    print("convolution_kernel took", end.time_since(start), "ms.")

    #copy output data back from GPU
    drv.memcpy_dtoh(output_image, gpu_args[0])

    #compare output with reference
    correct = numpy.allclose(output_image, reference, atol=1e-6)
    if not correct:
        print("TEST FAILED!")
    else:
        print("TEST PASSED!")


if __name__ == "__main__":

    #init pycuda
    drv.init()
    context = drv.Device(0).make_context()
    try:
        convolution_example(context)
    finally:
        context.pop()
