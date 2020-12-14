#!/usr/bin/env python
import numpy

import pycuda.driver as drv
from pycuda.compiler import SourceModule

def allocate(n, dtype=numpy.float32):
    """ allocate context-portable device mapped host memory """
    return drv.pagelocked_zeros(int(n), dtype, order='C', mem_flags=drv.host_alloc_flags.PORTABLE|drv.host_alloc_flags.DEVICEMAP)

def run_pnpoly(context, cc):

    #read kernel into string
    with open('pnpoly.cu', 'r') as f:
        kernel_string = f.read()

    #compile the kernels
    module = SourceModule(kernel_string, arch='compute_' + cc, code='sm_' + cc,
                    cache_dir=False, no_extern_c=True)
    pnpoly_kernel = module.get_function("cn_pnpoly")

    #set the number of points and the number of vertices
    size = numpy.int32(2e7)
    vertices = 600

    #allocate page-locked device-mapped host memory
    points = allocate(2*size, numpy.float32)
    bitmap = allocate(size, numpy.int32)
    vertices = allocate(2*vertices, numpy.float32)
    # HINT: need to reference constant memory
    #
    d_bitmap = numpy.intp(bitmap.base.get_device_pointer())
    d_points = numpy.intp(points.base.get_device_pointer())

    #generate/read input data
    numpy.copyto(points, numpy.random.randn(2*size).astype(numpy.float32))
    numpy.copyto(vertices, numpy.fromfile("vertices.dat", dtype=numpy.float32))

    #allocate gpu device memory for storing the vertices
    d_vertices = drv.mem_alloc(vertices.nbytes)

    #copy from host memory to GPU device memory
    drv.memcpy_htod(d_vertices, vertices)
    # HINT: need to also copy memory to constant array
    #
    #kernel arguments
    gpu_args = [d_bitmap, d_points, d_vertices, size]

    #setup thread block sizes
    threads = (256, 1, 1)
    grid = (int(numpy.ceil(size/float(threads[0]))), 1)

    #create events for time measurement
    start = drv.Event()
    end = drv.Event()

    #warm up the device a bit before measurement
    context.synchronize()
    for i in range(5):
        pnpoly_kernel(*gpu_args, block=threads, grid=grid)
    context.synchronize()

    #run the kernel and measure time using events
    start.record()
    pnpoly_kernel(*gpu_args, block=threads, grid=grid)
    end.record()
    context.synchronize()
    print("cn_pnpoly took", end.time_since(start), "ms.")

    #compute the reference answer using the reference kernel
    reference = allocate(size, numpy.int32)
    d_reference = numpy.intp(reference.base.get_device_pointer())
    reference_kernel = module.get_function("cn_pnpoly_reference_kernel")
    ref_args = [d_reference, d_points, d_vertices, size]
    context.synchronize()
    start.record()
    reference_kernel(*ref_args, block=threads, grid=grid)
    end.record()
    context.synchronize()
    print("reference kernel took", end.time_since(start), "ms.")


    #check if the result is the same
    test = numpy.sum(numpy.absolute(bitmap - reference)) == 0
    if test != True:
        print("answer:")
        print(bitmap)
        print("reference:")
        print(reference)
    else:
        print("ok!")



if __name__ == "__main__":
    drv.init()
    context = drv.Device(0).make_context()
    try:
        #get compute capability for compiling CUDA kernels
        devprops = { str(k): v for (k, v) in context.get_device().get_attributes().items() }
        cc = str(devprops['COMPUTE_CAPABILITY_MAJOR']) + str(devprops['COMPUTE_CAPABILITY_MINOR'])

        run_pnpoly(context, cc)
    finally:
        context.pop()
