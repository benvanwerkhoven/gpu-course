#if vecsize == 1
#define my_type float
#elif vecsize == 4
#define my_type float4
#endif


__global__ void local_averages_kernel(my_type * A, float * B, int size_B)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    if ( index < size_B )
    {
        float temp = 0.0;

        #if vecsize == 1        
        for ( int j = 0; j < 4; j++ )
        {
            temp = temp + A[(index * 4) + j];
        }
        #elif vecsize == 4
        temp = A[index].x + A[index].y + A[index].z + A[index].w;
        #endif

        #if use_division == 1
        B[index] = temp / 4.0;
        #else
        B[index] = temp * 0.25;
        #endif
    }
}
