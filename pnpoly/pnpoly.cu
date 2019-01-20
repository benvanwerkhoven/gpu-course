#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define VERTICES 600

__constant__ float2 c_vertices[VERTICES];

extern "C" {
    __global__ void cn_pnpoly(int *bitmap, float2 *points, float2 *vertices, int n);
    __global__ void cn_pnpoly_reference_kernel(int *bitmap, float2 *points, float2 *vertices, int n);
}

/*
 * This file contains the implementation of a CUDA Kernel for the
 * point-in-polygon problem using the crossing number algorithm
 *
 * Simplified for use in the NLeSC GPU Course
 *
 * The algorithm used here is adapted from: 
 *     'Inclusion of a Point in a Polygon', Dan Sunday, 2001
 *     (http://geomalgorithms.com/a03-_inclusion.html)
 *
 * Author: Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 */
__global__ void cn_pnpoly(int *bitmap, float2 *points, float2 *vertices, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        int c = 0;
        float2 p = points[i];

        int k = VERTICES-1;

        for (int j=0; j<VERTICES; k = j++) {    // edge from vk to vj
            float2 vj = c_vertices[j]; 
            float2 vk = c_vertices[k]; 

            float slope = (vk.x-vj.x) / (vk.y-vj.y);

            if ( (  (vj.y>p.y) != (vk.y>p.y)) &&            //if p is between vj and vk vertically
                    (p.x < slope * (p.y-vj.y) + vj.x) ) {   //if p.x crosses the line vk-vj when moved in positive x-direction
                c = !c;
            }
        }

        bitmap[i] = c; // 0 if even (out), and 1 if odd (in)
    }

}






int main() {

    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaSetDevice(0);
    cudaDeviceSynchronize();

    cudaError_t err;
    int num_points = (int)2e7;

    float2 *h_vertices;
    float2 *d_vertices;
    float2 *h_points;
    int *h_bitmap;
    int *h_reference;

    //Allocate pinned and aligned host memory and copy input data
    err = cudaHostAlloc((void **)&h_vertices, VERTICES*sizeof(float2), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaHostAlloc: %s\n", cudaGetErrorString(err));
    }
    err = cudaHostAlloc((void **)&h_points, num_points *sizeof(float2), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaHostAlloc: %s\n", cudaGetErrorString(err));
    }
    err = cudaHostAlloc((void **)&h_bitmap, num_points *sizeof(int), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaHostAlloc: %s\n", cudaGetErrorString(err));
    }
    err = cudaHostAlloc((void **)&h_reference, num_points *sizeof(int), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaHostAlloc: %s\n", cudaGetErrorString(err));
    }

    // generate random input
    for (int i=0; i< num_points; i++) {
        h_points[i].x = 50.0 / (rand() % 1000);
        h_points[i].y = 50.0 / (rand() % 1000);
    }
    // read vertices from disk
    FILE *file = fopen("vertices.dat", "rb");
    fread(h_vertices, sizeof(float), 2*VERTICES, file);

    // allocate constant memory for storing the vertices
    /*err = cudaMalloc((void **)&c_vertices, VERTICES*sizeof(float2));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMalloc: %s\n", cudaGetErrorString( err ));
    }*/

    // transfer vertices to c_vertices
    err = cudaMemcpyToSymbol(c_vertices, h_vertices, VERTICES*sizeof(float2));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMemcpyToSymbol: %s\n", cudaGetErrorString( err ));
    }

    // allocate device memory for storing the vertices
    err = cudaMalloc((void **)&d_vertices, VERTICES*sizeof(float2));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMalloc: %s\n", cudaGetErrorString( err ));
    }

    // transfer vertices to d_vertices
    err = cudaMemcpy(d_vertices, h_vertices, VERTICES*sizeof(float2), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMemcpy: %s\n", cudaGetErrorString(err));
    }

    // create CUDA streams and events
    cudaStream_t stream[1];
    err = cudaStreamCreate(&stream[0]);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaStreamCreate: %s\n", cudaGetErrorString(err));
    }
    cudaEvent_t start;
    err = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaEventCreate: %s\n", cudaGetErrorString(err));
    }

    cudaEvent_t stop;
    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaEventCreate: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error after memory setup: %s\n", cudaGetErrorString(err));
    }

    //kernel parameters
    dim3 threads(256, 1, 1);
    dim3 grid((int)ceil(num_points / (float)threads.x), 1);

    //run the kernel a few times to warmup the device
    for (int i=0; i<5; i++) {
        cn_pnpoly_reference_kernel<<<grid, threads, 0, stream[0]>>>(h_reference, h_points, d_vertices, num_points);
    }
    memset(h_bitmap, 0, num_points*sizeof(int));

    //start measuring time
    cudaDeviceSynchronize();
    cudaEventRecord(start, stream[0]);

    //call the kernel
    cn_pnpoly<<<grid, threads, 0, stream[0]>>>(h_bitmap, h_points, d_vertices, num_points);

    //stop time measurement
    cudaEventRecord(stop, stream[0]);
    cudaDeviceSynchronize();
    float time = 0.0;
    cudaEventElapsedTime(&time, start, stop);
    printf("cn_pnpoly kernel took: %f (ms)\n", time);



    //compute reference answer and measure time
    cudaDeviceSynchronize();
    cudaEventRecord(start, stream[0]);
    cn_pnpoly_reference_kernel<<<grid, threads, 0, stream[0]>>>(h_reference, h_points, d_vertices, num_points);
    cudaEventRecord(stop, stream[0]);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&time, start, stop);
    printf("reference kernel took: %f (ms)\n", time);

    //cleanup
    cudaStreamDestroy(stream[0]);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_vertices);
    cudaFreeHost(h_vertices);
    cudaFreeHost(h_points);

    //final check for errors
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error after CUDA kernel: %s\n", cudaGetErrorString(err));
        exit(1);
    } else {
        int errors = 0;
        int print = 0;
        for (int i=0; i<num_points; i++) {
            if (h_bitmap[i] != h_reference[i]) {
                errors++;
                if (print++ < 10) {
                    fprintf(stderr, "error at %d, reference=%d, answer=%d\n", i, h_reference[i], h_bitmap[i]);
                }
            }
        }
        if (errors == 0) {
            printf("ok!\n");
        } else {
            printf("there were %d errors\n", errors);
        }
    }

    cudaFreeHost(h_bitmap);
    cudaFreeHost(h_reference);

    return 0;
}






















/*
 * Reference kernel
 *
 * This kernel is kept for checking the output of the above kernel, DO NOT MODIFY THIS KERNEL
 */
__global__ void cn_pnpoly_reference_kernel(int *bitmap, float2 *points, float2 *vertices, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int c = 0;
        float2 p = points[i];   // DO NOT MODIFY THIS KERNEL
        int k = VERTICES-1;
        for (int j=0; j<VERTICES; k = j++) {
            float2 vj = vertices[j];    // DO NOT MODIFY THIS KERNEL
            float2 vk = vertices[k]; 
            float slope = (vk.x-vj.x) / (vk.y-vj.y);
            if ( (  (vj.y>p.y) != (vk.y>p.y)) && (p.x < slope * (p.y-vj.y) + vj.x) ) {
                c = !c;
            }
        }
        bitmap[i] = c; // DO NOT MODIFY THIS KERNEL
    }
}

