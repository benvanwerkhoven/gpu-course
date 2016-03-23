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

#define WIDTH 6144
#define HEIGHT 6144

#define FW 17
#define FH 17

#define BLOCK_X 16
#define BLOCK_Y 16

#define TILE_X 4

#define ITERATIONS 5

#ifdef USE_READ_ONLY_CACHE
#define LDG(x) __ldg(x)
#else
#define LDG(x) *(x)
#endif

extern "C" {
  void convolvution2d(float *image, float *scratch, float *filter, int scratchWidth, int scratchHeight, float filterWeight);
  void convolvution2d_explicit(float *image, float *scratch, int scratchWidth, int scratchHeight, float filterWeight);
  void convolvution2d_streams(float *image, float *scratch, int scratchWidth, int scratchHeight, float filterWeight);

  void start_timer();
  void stop_timer(float *);

  int compare(float *a, float *b, int N);

  __global__ void convolvution2d_kernel_naive(float *image, float *scratch, int scratchWidth, int scratchHeight, float filterWeight);
  __global__ void convolvution2d_kernel(float *__restrict__ iPtr, const float *__restrict__ sPtr, int totalWidth, int scratchHeight, float divisor);
}

float *h_filter;
float *h_image;
float *h_imageref;
float *h_scratch;
float filterWeight = 0.0;

__constant__ float d_filter[FW * FH];
float *d_image;
float *d_scratch;

int main() {
    cudaError_t err;

    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaSetDevice(0);
    cudaDeviceSynchronize();

    //allocate host memory
    h_imageref = (float *) malloc(WIDTH * HEIGHT * sizeof(float));
    if (!h_imageref) {
        fprintf(stderr, "Error in malloc: h_imageref\n");
    }

    h_filter = (float *) malloc(FW * FH * sizeof(float));
    if (!h_filter) {
        fprintf(stderr, "Error in malloc: h_filter\n");
    }

    h_image = (float *) malloc(WIDTH * HEIGHT * sizeof(float));
    if (!h_image) {
        fprintf(stderr, "Error in malloc: h_image\n");
    }

    h_scratch = (float *) malloc((WIDTH + FW - 1) * (HEIGHT + FH - 1) * sizeof(float));
    if (!h_scratch) {
        fprintf(stderr, "Error in malloc: h_scratch\n");
    }

    //fill host memory
    for (int y = 0; y < HEIGHT + FH - 1; y++) {
        for (int x = 0; x < WIDTH + FW - 1; x++) {
            int r = rand ();
            h_scratch[y * (WIDTH + FW - 1) + x] = 1.0 + r % 254;
        }
    }

    for (int y = 0; y < FH; y++) {
        for (int x = 0; x < FW; x++) {
            int r = rand ();
            float w = 0.001 + (r % 999) / 1000.0;
            h_filter[y * FW + x] = w;
            filterWeight += w;
        }
    }

    memset (h_image, 0, WIDTH * HEIGHT * sizeof (float));

    //allocate device memory
    err = cudaMalloc ((void **) &d_image, WIDTH * HEIGHT * sizeof(float));
    if (err != cudaSuccess) {
        fprintf (stderr, "Error in cudaMalloc d_image: %s\n", cudaGetErrorString(err));
    }

    err = cudaMemset (d_image, 0, WIDTH * HEIGHT * sizeof(float));
    if (err != cudaSuccess) {
        fprintf (stderr, "Error in cudaMemset d_image: %s\n", cudaGetErrorString(err));
    }

    err = cudaMalloc ((void **) &d_scratch, (WIDTH + FW - 1 ) * ( HEIGHT + FH - 1) * sizeof(float));
    if (err != cudaSuccess) {
        fprintf (stderr, "Error in cudaMalloc d_scratch: %s\n", cudaGetErrorString(err));
    }

    //error checking
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf (stderr, "Error after memory setup: %s\n", cudaGetErrorString(err));
    }

    //warm up the device
    for (int i = 0; i < 4*ITERATIONS; i++) {
        convolvution2d_explicit(h_image, h_scratch, WIDTH + FW - 1, HEIGHT + FH - 1, filterWeight);
    }
    memcpy(h_imageref, h_image, WIDTH * HEIGHT * sizeof(float));

    //measure execution time of the explicit implementation
    float time;
    cudaDeviceSynchronize();
    start_timer();
    for (int i = 0; i < ITERATIONS; i++) {
        convolvution2d_explicit(h_image, h_scratch, WIDTH + FW - 1, HEIGHT + FH - 1, filterWeight);
    }
    cudaDeviceSynchronize();
    stop_timer(&time);
    printf("Explicit average time: %.6f ms\n", time/ITERATIONS);
    compare(h_imageref, h_image, WIDTH * HEIGHT);

    //measure execution time of the streams implementation
    cudaDeviceSynchronize();
    start_timer();
    for (int i = 0; i < ITERATIONS; i++) {
        convolvution2d_streams(h_image, h_scratch, WIDTH + FW - 1, HEIGHT + FH - 1, filterWeight);
    }
    cudaDeviceSynchronize();
    stop_timer(&time);
    printf("Streams average time: %.6f ms\n", time/ITERATIONS);
    compare(h_imageref, h_image, WIDTH * HEIGHT);

    //cleanup
    cudaFree(d_image);
    cudaFree(d_scratch);
    free(h_image);
    free(h_imageref);
    free(h_scratch);
    free(h_filter);

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
void convolvution2d_explicit(float *image, float *scratch, int scratchWidth, int scratchHeight, float filterWeight) {
    cudaError_t err;

    dim3 threads (BLOCK_X, BLOCK_Y);
    // for naive kernel use this grid
    //  dim3 grid( (int)ceilf((float)WIDTH / (float)(BLOCK_X)) , (int)ceilf((float)HEIGHT / (float)(BLOCK_Y))); 
    // for tiled kernel use this grid
    dim3 grid ((int) ceilf ((float) WIDTH / (float) (TILE_X * BLOCK_X)), (int) ceilf ((float) HEIGHT / (float) (BLOCK_Y)));

    //copy h_filter to constant memory symbol d_filter (host to device)
    // err = cudaMemcpyToSymbol(...);

    //copy scratch in host memory to device memory d_scratch (host to device)

    //call the CUDA kernel instead of this CPU function
    convolvution2d(h_image, h_scratch, h_filter, scratchWidth, scratchHeight, filterWeight);

    //copy output data in d_image back to host memory array image (device to host)

    //force the host to wait for all device operations to be completed - do not remove
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
void convolvution2d_streams(float *image, float *scratch, int scratchWidth, int scratchHeight, float filterWeight) {

    //do not remove - important for measuring time correctly
    cudaDeviceSynchronize();
}
















/*
 * 2D Convolution reference CPU implementation
 */
void convolvution2d (float *image, float *scratch, float *filter, int scratchWidth, int scratchHeight, float filterWeight) {
    int x, y;
    int i, j;
    float sum = 0.0;

    for (y = 0; y < HEIGHT; y++) {
        for (x = 0; x < WIDTH; x++) {
            sum = 0.0;

            for (j = 0; j < FH; j++) {
	            for (i = 0; i < FW; i++) {
	                sum += scratch[(y + j) * scratchWidth + (x + i)] * filter[j * FH + i];
	            }
            }

            image[y * WIDTH + x] = sum / filterWeight;
        }
    }

}

/*
 * The following 2D Convolution kernel is a naive implementation
 * used for correctness checks
 */
__global__ void convolvution2d_kernel_naive(float *image, float *scratch, int scratchWidth, int scratchHeight, float filterWeight) {
    int x = blockIdx.x * BLOCK_X + threadIdx.x;
    int y = blockIdx.y * BLOCK_Y + threadIdx.y;
    int i, j;
    float sum = 0.0;

    if (y * x < HEIGHT * WIDTH) {
        for (j = 0; j < FH; j++) {
            for (i = 0; i < FW; i++) {
	            sum += scratch[(y + j) * scratchWidth + (x + i)] * d_filter[j * FH + i];
            }
        }

        image[y * WIDTH + x] = sum / filterWeight;
    }
}

#define SHWIDTH (4*BLOCK_X+FW-1)
#define SHMEMSIZE (SHWIDTH*(BLOCK_Y+FH-1))

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
__global__ void convolvution2d_kernel(float *__restrict__ iPtr, const float *__restrict__ sPtr, int totalWidth, int scratchHeight, float divisor) {
    float sum0 = 0;
    float sum1 = 0;
    float sum2 = 0;
    float sum3 = 0;

    int sindex = 0;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    //set scratch to point to start of scratch for this block
    sPtr += (ty + blockIdx.y * BLOCK_Y) * totalWidth + 4 * blockIdx.x * BLOCK_X + tx;
    iPtr += (ty + blockIdx.y * BLOCK_Y) * WIDTH + 4 * blockIdx.x * BLOCK_X + tx;

    //coalsced global memory loads
    //since there are more elements than threads there is some branching here
    sindex = ty * SHWIDTH + tx;

    shared_scratch[sindex] = LDG(sPtr);
    shared_scratch[sindex + 1 * BLOCK_X] = LDG(sPtr + 1 * BLOCK_X);
    shared_scratch[sindex + 2 * BLOCK_X] = LDG(sPtr + 2 * BLOCK_X);
    shared_scratch[sindex + 3 * BLOCK_X] = LDG(sPtr + 3 * BLOCK_X);
    shared_scratch[sindex + 4 * BLOCK_X] = LDG(sPtr + 4 * BLOCK_X);

    sindex += BLOCK_Y * SHWIDTH;
    sPtr += BLOCK_Y * totalWidth;
    shared_scratch[sindex] = LDG(sPtr);
    shared_scratch[sindex + 1 * BLOCK_X] = LDG(sPtr + 1 * BLOCK_X);
    shared_scratch[sindex + 2 * BLOCK_X] = LDG(sPtr + 2 * BLOCK_X);
    shared_scratch[sindex + 3 * BLOCK_X] = LDG(sPtr + 3 * BLOCK_X);
    shared_scratch[sindex + 4 * BLOCK_X] = LDG(sPtr + 4 * BLOCK_X);

    __syncthreads();
    sindex = ty * SHWIDTH + tx;

    //kernel computation
    int kindex = 0;
    int i = 0;
    int j = 0;
#pragma unroll
    for (j = 0; j < FH; j++) {
#pragma unroll
        for (i = 0; i < FW; i++) {
            sum0 += shared_scratch[sindex] * d_filter[kindex];
            sum1 += shared_scratch[sindex + 1 * BLOCK_X] * d_filter[kindex];
            sum2 += shared_scratch[sindex + 2 * BLOCK_X] * d_filter[kindex];
            sum3 += shared_scratch[sindex + 3 * BLOCK_X] * d_filter[kindex];
            sindex++;
            kindex++;
        }
        sindex = sindex - FW + SHWIDTH;
    }

    //global memory store
    *iPtr = sum0 / divisor;
    iPtr += BLOCK_X;
    *iPtr = sum1 / divisor;
    iPtr += BLOCK_X;
    *iPtr = sum2 / divisor;
    iPtr += BLOCK_X;
    *iPtr = sum3 / divisor;
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


