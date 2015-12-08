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

#define input_height (image_height + (filter_height/2)*2)
#define input_width (image_width + (filter_width/2)*2)


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






int main() {

    float time;
    cudaError_t err;

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
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMalloc d_input: %s\n", cudaGetErrorString( err ));
    err = cudaMalloc((void **)&d_output, image_height*image_width*sizeof(float));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMalloc d_output: %s\n", cudaGetErrorString( err ));
    err = cudaMalloc((void **)&d_filter, filter_height*filter_width*sizeof(float));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMalloc d_filter: %s\n", cudaGetErrorString( err ));

    //copy the input data to the GPU
    err = cudaMemcpy(d_input, input, input_height*input_width*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemcpy host to device input: %s\n", cudaGetErrorString( err ));
    err = cudaMemcpy(d_filter, filter, filter_height*filter_width*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemcpy host to device filter: %s\n", cudaGetErrorString( err ));

    //zero the output array
    err = cudaMemset(d_output, 0, image_height*image_width*sizeof(float));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemset output: %s\n", cudaGetErrorString( err ));

    //setup the grid and thread blocks
    //thread block size
    dim3 threads(32, 32);
    //problem size divided by thread block size rounded up
    dim3 grid(int(ceilf(image_width/(float)threads.x)), int(ceilf(image_height/(float)threads.y)) );

    //measure the GPU function
    start_timer();
    convolution_kernel<<<grid, threads>>>(d_output, d_input, d_filter);
    cudaDeviceSynchronize();
    stop_timer(&time);
    printf("convolution_kernel took %.3f ms\n", time);

    //check to see if all went well
    err = cudaGetLastError();
    if (err != cudaSuccess) fprintf(stderr, "Error during kernel launch convolution_kernel: %s\n", cudaGetErrorString( err ));

    //copy the result back to host memory
    err = cudaMemcpy(output2, d_output, image_height*image_width*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemcpy device to host output: %s\n", cudaGetErrorString( err ));

    //check the result
    int errors = compare_arrays(output1, output2, image_height*image_width);
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

        /*
        unsigned int int_a1 = *(unsigned int *)(a1+i);
        unsigned int int_a2 = *(unsigned int *)(a2+i);
        unsigned int dist = (unsigned int)0;
        if (int_a1 > int_a2) {
            dist = int_a1 - int_a2;
        } else {
            dist = int_a2 - int_a1;
        }
        if (dist > 0) {
            errors++;
            if (print < 10) {
                print++;
                fprintf(stderr, "Error detected at i=%d, \t a1= \t %10.7e \t a2= \t %10.7e \t ulp_dist=\t %u\n",i,a1[i],a2[i],dist);
            }
        }
        */

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
