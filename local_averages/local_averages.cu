__global__ void local_averages_kernel(float * A, float * B, int size_B)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    if ( index < size_B )
    {
        float temp = 0.0;
        
        for ( int j = 0; j < 4; j++ )
        {
            temp = temp + A[(index * 4) + j];
        }
        B[index] = temp / 4.0;
    }
}