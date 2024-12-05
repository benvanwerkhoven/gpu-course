/*
 * Reference kernel
 *
 * This kernel is kept for checking the output of the above kernel, PLEASE DO NOT MODIFY THIS KERNEL
 */
extern "C"
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
