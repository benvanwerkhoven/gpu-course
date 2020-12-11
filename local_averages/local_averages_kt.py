from collections import OrderedDict
import numpy as np
import kernel_tuner

# Size of the problem
N = np.int32(10e6)
# Input array
A = np.random.randn(N).astype(np.float32)
# Reference output array
B1 = np.zeros(N//4).astype(np.float32)
# Output array for the GPU
B2 = np.zeros_like(B1)


# Naive version of the algorithm in Python
def local_averages(A, B, N):
    for i in range(0, np.int32(N/4)):
        temp = 0.0
        for j in range(0, 4):
            temp = temp + A[(i * 4) + j]
        B[i] = temp / 4.0


# Running the Python sequential version and storing the result in B1
local_averages(A, B1, N)

# We will specify the tunable parameters of this kernel using a dictionary
tune_params = OrderedDict()

# Using the special name "block_size_x" we can specify what values 
# Kernel Tuner should use for the number of threads per block in the x-dimension
tune_params["block_size_x"] = [1024]

# We can also specify how Kernel Tuner should compute performance metrics
metrics = OrderedDict(GFLOPs=lambda p: (N/4*5/1e9)/(p["time"]/1e3))

res, env = kernel_tuner.tune_kernel("local_averages_kernel", # The name of the kernel
                                    "local_averages.cu",     # Kernel source file
                                    N//4,                    # Problem size
                                    [A, B2, N],              # Kernel argument list
                                    tune_params,             # Tunable parameters
                                    answer=[None, B1, None], # Reference answer
                                    metrics=metrics)         # Performance metric
