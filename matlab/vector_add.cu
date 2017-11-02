__global__ void vector_add(float *c, const float *a, const float *b, const int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n) {
        c[i] = a[i] + b[i];
    }
}
