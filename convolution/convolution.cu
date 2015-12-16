#include <stdio.h>

extern "C" {
    void start_timer();
    void stop_timer(float *time);
}

int compare_arrays(float *c, float *d, int n);


#define image_height 1024
#define image_width 1024
#define filter_height 17
#define filter_width 17

#define block_size_x 32
#define block_size_y 16

#define border_height ((filter_height/2)*2)
#define border_width ((filter_width/2)*2)
#define input_height (image_height + border_height)
#define input_width (image_width + border_width)

void convolve(float *output, float *input, float *filter) {
    //for each pixel in the output image
    for (int y=0; y < image_height; y++) {
        for (int x=0; x < image_width; x++) { 

            //for each filter weight
            for (int i=0; i < filter_height; i++) {
                for (int j=0; j < filter_width; j++) {
                    output[y*image_width+x] += input[(y+i)*input_width+x+j] * filter[i*filter_width+j];
                }
            }

        }
    }

}


__global__ void convolution_kernel(float *output, float *input, float *filter) {
    //for each pixel in the output image
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    //thread-local register to hold local sum
    float sum = 0.0f;

    //for each filter weight
    for (int i=0; i < filter_height; i++) {
        for (int j=0; j < filter_width; j++) {
            sum += input[(y+i)*input_width+x+j] * filter[i*filter_width+j];
        }
    }

    //store result to global memory
    output[y*image_width+x] = sum;

}


__global__ void convolution_kernel_shared_mem(float *output, float *input, float *filter) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int by = blockIdx.y * block_size_y;
    int bx = blockIdx.x * block_size_x;
    int y = by+ty;
    int x = bx+tx;

    //declare shared memory for this thread block
    //the area reserved is equal to the thread block size plus
    //the size of the border needed for the computation
    __shared__ float sh_input[block_size_y+border_height][block_size_x+border_width];

    //Write a for loop that loads all values needed by this thread block
    //from global memory (input) and stores it into shared memory (sh_input)
    //that is local to this thread block
    //for ( ... ) {
        //for ( ... ) {
            //...
        //}
    //}

    //synchronize to make all writes visible to all threads within the thread block
    __syncthreads();
    
    //thread-local register to hold local sum
    float sum = 0.0f;

    //for each filter weight
    for (int i=0; i < filter_height; i++) {
        for (int j=0; j < filter_width; j++) {
            // Oops! I forgot to actually use sh_input instead of input! Please fix it!
            sum += input[(y+i)*input_width+x+j] * filter[i*filter_width+j];
        }
    }

    //store result to global memory
    output[y*image_width+x] = sum;

}






int main() {

    float time;
    cudaError_t err;
    int errors = 0;

    //allocate arrays and fill them
    float *input = (float *) malloc(input_height * input_width * sizeof(float));
    float *output1 = (float *) malloc(image_height * image_width * sizeof(float));
    float *output2 = (float *) malloc(image_height * image_width * sizeof(float));
    float *filter = (float *) malloc(filter_height * filter_width * sizeof(float));
    for (int i=0; i< input_height * input_width; i++) {
        input[i] = 1.0 / rand();
    }
    for (int i=0; i< filter_height * filter_width; i++) {
        filter[i] = 1.0 / rand();
    }    
    memset(output1, 0, image_height * image_width * sizeof(float));
    memset(output2, 0, image_height * image_width * sizeof(float));

    //measure the CPU function
    start_timer();
    convolve(output1, input, filter);
    stop_timer(&time);
    printf("convolve took %.3f ms\n", time);

    //allocate GPU memory
    float *d_input; float *d_output; float *d_filter;
    err = cudaMalloc((void **)&d_input, input_height*input_width*sizeof(float));
    if (err != cudaSuccess) { fprintf(stderr, "Error in cudaMalloc d_input: %s\n", cudaGetErrorString( err )); errors++; }
    err = cudaMalloc((void **)&d_output, image_height*image_width*sizeof(float));
    if (err != cudaSuccess) { fprintf(stderr, "Error in cudaMalloc d_output: %s\n", cudaGetErrorString( err )); errors++; }
    err = cudaMalloc((void **)&d_filter, filter_height*filter_width*sizeof(float));
    if (err != cudaSuccess) { fprintf(stderr, "Error in cudaMalloc d_filter: %s\n", cudaGetErrorString( err )); errors++; }

    //copy the input data to the GPU
    err = cudaMemcpy(d_input, input, input_height*input_width*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "Error in cudaMemcpy host to device input: %s\n", cudaGetErrorString( err )); errors++; }
    err = cudaMemcpy(d_filter, filter, filter_height*filter_width*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "Error in cudaMemcpy host to device filter: %s\n", cudaGetErrorString( err )); errors++; }

    //zero the output array
    err = cudaMemset(d_output, 0, image_height*image_width*sizeof(float));
    if (err != cudaSuccess) { fprintf(stderr, "Error in cudaMemset output: %s\n", cudaGetErrorString( err )); errors++; }

    //setup the grid and thread blocks
    //thread block size
    dim3 threads(block_size_x, block_size_y);
    //problem size divided by thread block size rounded up
    dim3 grid(int(ceilf(image_width/(float)threads.x)), int(ceilf(image_height/(float)threads.y)) );

    //measure the GPU function
    cudaDeviceSynchronize();
    start_timer();
    convolution_kernel<<<grid, threads>>>(d_output, d_input, d_filter);
    cudaDeviceSynchronize();
    stop_timer(&time);
    printf("convolution_kernel took %.3f ms\n", time);

    //check to see if all went well
    err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "Error during kernel launch convolution_kernel: %s\n", cudaGetErrorString( err )); errors++; }

    //copy the result back to host memory
    err = cudaMemcpy(output2, d_output, image_height*image_width*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "Error in cudaMemcpy device to host output: %s\n", cudaGetErrorString( err )); errors++; }

    //check the result
    errors += compare_arrays(output1, output2, image_height*image_width);
    if (errors > 0) {
        printf("TEST FAILED! %d errors!\n", errors);
    } else {
        printf("TEST PASSED!\n");
    }

    //zero the output arrays
    errors = 0;
    memset(output2, 0, image_height*image_width*sizeof(float));
    err = cudaMemset(d_output, 0, image_height*image_width*sizeof(float));
    if (err != cudaSuccess) { fprintf(stderr, "Error in cudaMemset output: %s\n", cudaGetErrorString( err )); errors++; }

    //measure the GPU function
    start_timer();
    convolution_kernel_shared_mem<<<grid, threads>>>(d_output, d_input, d_filter);
    cudaDeviceSynchronize();
    stop_timer(&time);
    printf("convolution_kernel_shared_mem took %.3f ms\n", time);

    //check to see if all went well
    err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "Error during kernel launch convolution_kernel_shared_mem: %s\n", cudaGetErrorString( err )); errors++; }

    //copy the result back to host memory
    err = cudaMemcpy(output2, d_output, image_height*image_width*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "Error in cudaMemcpy device to host output: %s\n", cudaGetErrorString( err )); errors++; }

    //check the result
    errors += compare_arrays(output1, output2, image_height*image_width);
    if (errors > 0) {
        printf("TEST FAILED! %d errors!\n", errors);
    } else {
        printf("TEST PASSED!\n");
    }

    //clean up
    cudaFree(d_output);
    cudaFree(d_input);
    cudaFree(d_filter);
    free(filter);
    free(input);
    free(output1);
    free(output2);

    return 0;
}



int compare_arrays(float *a1, float *a2, int n) {
    int errors = 0;
    int print = 0;

    for (int i=0; i<n; i++) {

        if (isnan(a1[i]) || isnan(a2[i])) {
            errors++;
            if (print < 10) {
                print++;
                fprintf(stderr, "Error NaN detected at i=%d,\t a1= %10.7e \t a2= \t %10.7e\n",i,a1[i],a2[i]);
            }
        }

        float diff = (a1[i]-a2[i])/a1[i];
        if (diff > 1e-6f) {
            errors++;
            if (print < 10) {
                print++;
                fprintf(stderr, "Error detected at i=%d, \t a1= \t %10.7e \t a2= \t %10.7e \t rel_error=\t %10.7e\n",i,a1[i],a2[i],diff);
            }
        }

    }

    return errors;
}
