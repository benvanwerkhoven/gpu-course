__global__ void vector_add(float *c, const float *a, const float *b, const int n) {
    int i = threadIdx.x;  /* <--- Oops! something is not right here! */ 
    if (i<n) {
        c[i] = a[i] + b[i];
    }
}
