#include <stdio.h>

#define block_size_x 256
#define num_blocks 1024


extern "C" {
    void start_timer();
    void stop_timer(float *time);
    __global__ void reduce_kernel(float *out_array, float *in_array, int n);
}

//a naive summation in C
float sum_floats(float *in_array, int n) {
    float sum = 0.0;
    for (int i=0; i<n; i++) {
        sum += in_array[i];
    }
    return sum;
}

//Kahan summation to avoid floating-point precision errors
float sum_floats_kahan(float *in_array, int n) {
    float sum = 0.0;
    float c = 0.0;
    for (int i=0; i<n; i++) {
        float v = in_array[i] - c;
        float t = sum + v;
        c = (t - sum) - v;
        sum = t;
    }
    return sum;
}

//CUDA kernel for parallel reduction
__global__ void reduce_kernel(float *out_array, float *in_array, int n) {

    int ti = threadIdx.x;
    int x = blockIdx.x * block_size_x + threadIdx.x;
    int step_size = gridDim.x * block_size_x;
    float sum = 0.0f;

    //cooperatively (with all threads in all thread blocks) iterate over input array
    for (int i=x; i<n; i+=step_size) {
        sum += in_array[i];
    }

    //at the point we have reduced the number of values to be summed from n to
    //the total number of threads in all thread blocks combined

    //the goal is now to reduce the values within each thread block to a single
    //value per thread block for this we will need shared memory

    //declare shared memory array, how much shared memory do we need?
    //__shared__ float ...;
    __shared__ float thread_sum[block_size_x];

    //make every thread store its thread-local sum to the array in shared memory
    //... = sum;
    thread_sum[ti] = sum;
    
    //now let's call syncthreads() to make sure all threads have finished
    //storing their local sums to shared memory
    __syncthreads();

    //now this interesting looking loop will do the following:
    //it iterates over the block_size_x with the following values for s:
    //if block_size_x is 256, 's' will be powers of 2 from 128, 64, 32, down to 1.
    //these decreasing offsets can be used to reduce the number
    //of values within the thread block in only a few steps.
    #pragma unroll
    for (unsigned int s=block_size_x/2; s>0; s/=2) {

        //you are to write the code inside this loop such that
        //threads will add the sums of other threads that are 's' away
        //do this iteratively such that together the threads compute the
        //sum of all thread-local sums 

        //use shared memory to access the values of other threads
        //and store the new value in shared memory to be used in the next round
        //be careful that values that should be read are
        //not overwritten before they are read
        //make sure to call __syncthreads() when needed
        if (ti < s) {
            thread_sum[ti] += thread_sum[ti+s];
        }
        __syncthreads();
    }

    //write back one value per thread block
    if (ti == 0) {
        //out_array[blockIdx.x] = ;  //store the per-thread block reduced value to global memory
        out_array[blockIdx.x] = thread_sum[ti];
    }
}


int main() {

    int n = (int)5e7; //problem size
    float time;
    cudaError_t err;

    //allocate arrays and fill them
    float *in_array = (float *) malloc(n * sizeof(float));
    float *out_array = (float *) malloc(num_blocks * sizeof(float));
    for (int i=0; i<n; i++) {
        in_array[i] = (rand() % 10000) / 100000.0;
    }
    memset(out_array, 0, num_blocks * sizeof(float));

    //measure the CPU function
    start_timer();
    float sum = sum_floats(in_array, n);
    stop_timer(&time);
    printf("sum_floats took %.3f ms\n", time);

    //allocate GPU memory
    float *d_in; float *d_out;
    err = cudaMalloc((void **)&d_in, n*sizeof(float));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMalloc: %s\n", cudaGetErrorString( err ));
    err = cudaMalloc((void **)&d_out, num_blocks*sizeof(float));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMalloc: %s\n", cudaGetErrorString( err ));

    //copy the input data to the GPU
    err = cudaMemcpy(d_in, in_array, n*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemcpy host to device: %s\n", cudaGetErrorString( err ));

    //zero the output array
    err = cudaMemset(d_out, 0, num_blocks*sizeof(float));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemset: %s\n", cudaGetErrorString( err ));

    //setup the grid and thread blocks
    dim3 grid(num_blocks, 1);
    dim3 grid2(1, 1);
    dim3 threads(block_size_x, 1, 1);

    //measure the GPU function
    cudaDeviceSynchronize();
    start_timer();
    reduce_kernel<<<grid, threads>>>(d_out, d_in, n);
    reduce_kernel<<<grid2, threads>>>(d_out, d_out, num_blocks); //call the kernel again with only 1 thread block
    cudaDeviceSynchronize();
    stop_timer(&time);
    printf("reduce_kernel took %.3f ms\n", time);

    //check to see if all went well
    err = cudaGetLastError();
    if (err != cudaSuccess) fprintf(stderr, "Error during kernel launch: %s\n", cudaGetErrorString( err ));

    //copy the result back to host memory
    err = cudaMemcpy(out_array, d_out, 1*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemcpy device to host: %s\n", cudaGetErrorString( err ));

    //compute a reliable reference answer on the host
    float sum2 = sum_floats_kahan(in_array, n);

    //check the result
    float diff = abs(*out_array - sum2);
    printf("cpu: %f, corrected: %f\n", sum, sum2);
    printf("gpu: %f\n", *out_array);

    if (diff < 1.0) {
        printf("TEST PASSED!\n");
    } else {
        printf("TEST FAILED!\n");
    }

    //clean up
    cudaFree(d_in);
    cudaFree(d_out);
    free(in_array);
    free(out_array);

    return 0;
}

