/*
 * Copyright 2014 Netherlands eScience Center
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


/**
 * This program is part of a training in GPU Programming to practice with
 * overlapping CPU-GPU communication and GPU computation of a
 * 2D Convolution kernel.
 *
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 */
#include <stdio.h>
#include <stdlib.h>

#define image_height 1024
#define image_width 1024
#define filter_height 17
#define filter_width 17

#define block_size_x 16
#define block_size_y 16
#define tile_size_x 4

#define border_height ((filter_height/2)*2)
#define border_width ((filter_width/2)*2)
#define input_height (image_height + border_height)
#define input_width (image_width + border_width)

#define ITERATIONS 5

#ifdef USE_READ_ONLY_CACHE
#define LDG(x) __ldg(x)
#else
#define LDG(x) *(x)
#endif

extern "C" {
  void convolution2d_explicit(float *image, float *input, float *filter);
  void convolution2d_streams(float *image, float *input, float *filter);

  void start_timer();
  void stop_timer(float *);

  int compare(float *a, float *b, int N);

  void convolution2d(float *image, float *input, float *filter);
  __global__ void convolution2d_kernel_naive(float *image, float *input);
  __global__ void convolution2d_kernel(float *__restrict__ image, const float *__restrict__ input);
}

//host memory pointers
float *h_filter;
float *h_image;
float *h_imageref;
float *h_input;

//device memory 
__constant__ float d_filter[filter_width*filter_height];
float *d_image;
float *d_input;

int main() {
    cudaError_t err;

    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaSetDevice(0);
    cudaDeviceSynchronize();

    //allocate host memory
    h_imageref = (float *) malloc(image_width * image_height * sizeof(float));
    if (!h_imageref) {
        fprintf(stderr, "Error in malloc: h_imageref\n");
    }

    err = cudaHostAlloc(&h_filter, filter_width * filter_height * sizeof(float), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        fprintf (stderr, "Error in cudaHostAlloc h_filter: %s\n", cudaGetErrorString(err));
    }

    err = cudaHostAlloc(&h_image, image_width * image_height * sizeof(float), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        fprintf (stderr, "Error in cudaHostAlloc h_image: %s\n", cudaGetErrorString(err));
    }

    err = cudaHostAlloc(&h_input, input_width * input_height * sizeof(float), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        fprintf (stderr, "Error in cudaHostAlloc h_input: %s\n", cudaGetErrorString(err));
    }

    //fill the input image including border
    for (int y = 0; y < input_height; y++) {
        for (int x = 0; x < input_width; x++) {
            int r = rand();
            h_input[y * input_width + x] = 1.0 + r % 254;
        }
    }

    //fill the filter
    for (int y = 0; y < filter_height; y++) {
        for (int x = 0; x < filter_width; x++) {
            int r = rand();
            float w = 0.001 + (r % 999) / 1000.0;
            h_filter[y * filter_width + x] = w;
        }
    }

    memset(h_image, 0, image_width * image_height * sizeof (float));
    memset(h_imageref, 0, image_width * image_height * sizeof (float));

    //allocate device memory
    err = cudaMalloc ((void **) &d_image, image_width * image_height * sizeof(float));
    if (err != cudaSuccess) {
        fprintf (stderr, "Error in cudaMalloc d_image: %s\n", cudaGetErrorString(err));
    }

    err = cudaMemset (d_image, 0, image_width * image_height * sizeof(float));
    if (err != cudaSuccess) {
        fprintf (stderr, "Error in cudaMemset d_image: %s\n", cudaGetErrorString(err));
    }

    err = cudaMalloc ((void **) &d_input, input_width * input_height * sizeof(float));
    if (err != cudaSuccess) {
        fprintf (stderr, "Error in cudaMalloc d_input: %s\n", cudaGetErrorString(err));
    }

    //error checking
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf (stderr, "Error after memory setup: %s\n", cudaGetErrorString(err));
    }

    //compute reference answer using naive kernel and mapped memory
    dim3 threads(block_size_x, block_size_y);
    dim3 grid((int)ceilf((float)image_width / (float)(block_size_x)) , (int)ceilf((float)image_height / (float)(block_size_y))); 
    err = cudaMemcpyToSymbol(d_filter, h_filter, filter_width * filter_height * sizeof(float), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf (stderr, "Error in cudaMemcpyToSymbol filter: %s\n", cudaGetErrorString(err));
    }
    convolution2d_kernel_naive<<<grid, threads>>>(h_image, h_input);
    cudaDeviceSynchronize();
    memcpy(h_imageref, h_image, image_width * image_height * sizeof(float));

    //warm up the device
    for (int i = 0; i < 4*ITERATIONS; i++) {
        convolution2d_explicit(h_image, h_input, h_filter);
    }

    //measure execution time of the explicit implementation
    float time;
    cudaDeviceSynchronize();
    start_timer();
    for (int i = 0; i < ITERATIONS; i++) {
        convolution2d_explicit(h_image, h_input, h_filter);
    }
    cudaDeviceSynchronize();
    stop_timer(&time);
    printf("Explicit average time: %.6f ms\n", time/ITERATIONS);
    compare(h_imageref, h_image, image_width * image_height);

    //measure execution time of the streams implementation
    cudaDeviceSynchronize();
    start_timer();
    for (int i = 0; i < ITERATIONS; i++) {
        convolution2d_streams(h_image, h_input, h_filter);
    }
    cudaDeviceSynchronize();
    stop_timer(&time);
    printf("Streams average time: %.6f ms\n", time/ITERATIONS);
    compare(h_imageref, h_image, image_width*image_height);

    //cleanup
    cudaFree(d_image);
    cudaFree(d_input);
    cudaFreeHost(h_image);
    cudaFreeHost(h_input);
    cudaFreeHost(h_filter);
    free(h_imageref);

    return 0;
}





/*
 * Host code that invokes the 2D Convolution kernel
 *
 * The explicit implementation uses explicit memory copy
 * statements to move all data to the GPU, executes the
 * GPU kernel, and uses memory copies to copy the output
 * data back to host memory. This implementation achieves
 * no overlap between transfers and/or computation.
 *
 */
void convolution2d_explicit(float *h_image, float *h_input, float *h_filter) {
    cudaError_t err;

    dim3 threads (block_size_x, block_size_y);
    // for naive kernel use this grid
    //  dim3 grid ((int) ceilf ((float) image_width / (float) (block_size_x)), (int) ceilf ((float) image_height / (float) (block_size_y)));
    // for tiled kernel use this grid
    dim3 grid ((int) ceilf ((float) image_width / (float) (tile_size_x * block_size_x)), (int) ceilf ((float) image_height / (float) (block_size_y)));

    //copy h_filter to constant memory symbol d_filter (host to device)
    err = cudaMemcpyToSymbol(d_filter, h_filter, filter_width * filter_height * sizeof(float), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf (stderr, "Error in cudaMemcpyToSymbol filter: %s\n", cudaGetErrorString(err));
    }

    //TODO: copy host memory array h_input to device memory array d_input 
    //note the size of h_input in bytes is input_width * input_height * sizeof(float)
    //err = cudaMemcpy( ... );

    //TODO: change this kernel launch to use the device memory arrays d_image and d_image instead of device-mapped host memory
    convolution2d_kernel<<<grid, threads>>>(h_image, h_input);

    //TODO: copy the device memory array d_image to host memory array h_image
    //note the size of d_image is image_width * image_height * sizeof(float)
    //err = cudaMemcpy( ... );

    //do not remove
    cudaDeviceSynchronize();
}


/*
 * Host code that invokes the 2D Convolution kernel using multiple streams
 *
 * The streams implementation should use CUDA streams combined
 * with explicit memory copy statements. This way transfers
 * in one stream may overlap with computation and transfers
 * in other streams. 
 *
 */
void convolution2d_streams(float *h_image, float *h_input, float *h_filter) {

    // thread block size - no need to change
    dim3 threads (block_size_x, block_size_y);
    dim3 grid ((int) ceilf ((float) image_width / (float) (tile_size_x * block_size_x)), (int) ceilf ((float) image_height / (float) (block_size_y)));

    /* Host to Device transfer of input data
     *
     * You should rewrite this code to allow Host to Device transfers
     * to happen asynchronously with respect to the host.
     * Additionally you should use streams to enable overlap between
     * these transfers and kernel execution and transfers from device to host.
     */

    /* Calling the kernel
     *
     * This statement calls the kernel with a given amount of thread blocks - specified in the variable grid.
     * To specify that the kernel should be executed in a particular stream you can use the following syntax:
     * kernel<<<grid, threads, 0, stream>>>, where stream is the cudaStream_t identifying the stream.
     * Be sure to use offsets to direct the kernel in a stream to the correct part of the input and output
     * data.
     */
    convolution2d_kernel<<<grid, threads>>>(h_image, h_input);

    /* Device to Host transfers of output data
     *
     * You can use streams to overlap device to host transfers with operations in other streams. This way
     * you can start copying the output data back to main memory directly after the kernel in a stream has
     * finished its computations.
     */

    //do not remove - important for measuring time correctly
    cudaDeviceSynchronize();
}
















/*
 * 2D Convolution reference CPU implementation
 */
void convolution2d (float *image, float *input, float *filter) {
    int x, y;
    int i, j;
    float sum = 0.0;

    for (y = 0; y < image_height; y++) {
        for (x = 0; x < image_width; x++) {
            sum = 0.0;

            for (j = 0; j < filter_height; j++) {
	            for (i = 0; i < filter_width; i++) {
	                sum += input[(y + j) * input_width + (x + i)] * filter[j * filter_width + i];
	            }
            }

            image[y * image_width + x] = sum;
        }
    }

}

/*
 * The following 2D Convolution kernel is a naive implementation
 * used for correctness checks
 */
__global__ void convolution2d_kernel_naive(float *image, float *input) {
    int x = blockIdx.x * block_size_x + threadIdx.x;
    int y = blockIdx.y * block_size_y + threadIdx.y;
    int i, j;
    float sum = 0.0;

    if (y < image_height && x < image_width) {
        for (j = 0; j < filter_height; j++) {
            for (i = 0; i < filter_width; i++) {
	            sum += input[(y + j) * input_width + (x + i)] * d_filter[j * filter_width + i];
            }
        }

        image[y * image_width + x] = sum;
    }
}

#define SHWIDTH (tile_size_x*block_size_x+filter_width-1)
#define SHMEMSIZE (SHWIDTH*(block_size_y+filter_height-1))

__shared__ float shared_scratch[SHMEMSIZE];

/*
 * The following 2D Convolution kernel is adapted from a kernel
 * specifically generated for filter size 17x17 and thread block
 * size 16x16 using 1x4 tiling.
 *
 * For more information on how this kernel was generated see:
 *  Optimizing convolution operations on GPUs using adaptive tiling
 *  B. van Werkhoven, J. Maassen, F.J. Seinstra, H.E Bal
 *  Future Generation Computer Systems, Volume 30, 2014
 */
__global__ void convolution2d_kernel(float *__restrict__ image, const float *__restrict__ input) {
    float sum0 = 0;
    float sum1 = 0;
    float sum2 = 0;
    float sum3 = 0;

    int sindex = 0;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    //set input to point to start of input for this block
    input += (ty + blockIdx.y * block_size_y) * input_width + tile_size_x * blockIdx.x * block_size_x + tx;
    image += (ty + blockIdx.y * block_size_y) * image_width + tile_size_x * blockIdx.x * block_size_x + tx;

    //coalsced global memory loads
    sindex = ty * SHWIDTH + tx;

    shared_scratch[sindex] = LDG(input);
    shared_scratch[sindex + 1 * block_size_x] = LDG(input + 1 * block_size_x);
    shared_scratch[sindex + 2 * block_size_x] = LDG(input + 2 * block_size_x);
    shared_scratch[sindex + 3 * block_size_x] = LDG(input + 3 * block_size_x);
    shared_scratch[sindex + 4 * block_size_x] = LDG(input + 4 * block_size_x);

    sindex += block_size_y * SHWIDTH;
    input += block_size_y * input_width;
    shared_scratch[sindex] = LDG(input);
    shared_scratch[sindex + 1 * block_size_x] = LDG(input + 1 * block_size_x);
    shared_scratch[sindex + 2 * block_size_x] = LDG(input + 2 * block_size_x);
    shared_scratch[sindex + 3 * block_size_x] = LDG(input + 3 * block_size_x);
    shared_scratch[sindex + 4 * block_size_x] = LDG(input + 4 * block_size_x);

    __syncthreads();
    sindex = ty * SHWIDTH + tx;

    //kernel computation
    int kindex = 0;
    int i = 0;
    int j = 0;
#pragma unroll
    for (j = 0; j < filter_height; j++) {
#pragma unroll
        for (i = 0; i < filter_width; i++) {
            sum0 += shared_scratch[sindex] * d_filter[kindex];
            sum1 += shared_scratch[sindex + 1 * block_size_x] * d_filter[kindex];
            sum2 += shared_scratch[sindex + 2 * block_size_x] * d_filter[kindex];
            sum3 += shared_scratch[sindex + 3 * block_size_x] * d_filter[kindex];
            sindex++;
            kindex++;
        }
        sindex = sindex - filter_width + SHWIDTH;
    }

    //global memory store
    *image = sum0;
    image += block_size_x;
    *image = sum1;
    image += block_size_x;
    *image = sum2;
    image += block_size_x;
    *image = sum3;
}



/*
 * Compare function that compares two arrays of length N for similarity
 * 
 * This function performs a number of different tests, for example the number of
 * values at an epsilon from 0.0 should be similar in both arrays and may not
 * be greater than 1/4th of the array. Additionally NaN values are treated as
 * errors.
 *
 * The value of eps should be adjusted to something reasonable given the
 * fact that CPU and GPU do not produce exactly the same numerical results. 
 */
int compare(float *a1, float *a2, int N) {
    int i = 0, res = 0;
    int print = 0;
    int zero_one = 0;
    int zero_two = 0;
    float eps = 1e-6f;

    for (i = 0; i < N; i++) {

        if (a1[i] < eps && a1[i] > -eps) {
            zero_one++;
        }
        if (a2[i] < eps && a2[i] > -eps) {
            zero_two++;
        }

        if (isnan (a1[i]) || isnan (a2[i])) {
            res++;
            if (print < 10) {
	            print++;
	            fprintf (stderr, "Error detected at i=%d,\t a1= %10.7e \t a2= \t %10.7e\n", i, a1[i], a2[i]);
            }
        }

        float diff = a1[i] - a2[i];
        if (diff > eps || diff < -eps) {
            res++;
            if (print < 10) {
	            print++;
	            fprintf(stderr, "Error detected at i=%d,\t a1= \t %10.7e \t a2= \t %10.7e\n", i, a1[i], a2[i]);
            }
        }

    }

    if (zero_one > (N / 4)) {
        fprintf(stderr, "Warning: array1 contains %d zeros\n", zero_one);
    }
    if (zero_two > (N / 4)) {
        fprintf(stderr, "Warning: array2 contains %d zeros\n", zero_two);
    }

    if (zero_one != zero_two) {
        fprintf(stderr, "Warning: number of zeros in arrays dont correspond zero1=%d, zero2=%d\n", zero_one, zero_two);
    }

    if (res > 0) {
        fprintf(stdout, "Number of errors in GPU result: %d\n", res);
    }

    return res;
}


