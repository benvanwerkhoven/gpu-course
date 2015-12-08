#include <stdio.h>
#include <stdlib.h>

extern "C" {
    void start_timer();
    void stop_timer(float* time);
}

#define N 1000

int main() {

    float time;
    int i,j,k;
    float *A = (float*)malloc(N*N*sizeof(float));
    float *B = (float*)malloc(N*N*sizeof(float));
    float *C = (float*)malloc(N*N*sizeof(float));

    //ijk
    start_timer();
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            for (k=0; k<N; k++) {
                C[i*N+j] += A[i*N+k] * B[k*N+j];
            }
        }
    }
    stop_timer(&time);
    printf("ijk loop order took %.6f ms\n", time);

    //ikj
    start_timer();
    for (i=0; i<N; i++) {
        for (k=0; k<N; k++) {
            for (j=0; j<N; j++) {
                C[i*N+j] += A[i*N+k] * B[k*N+j];
            }
        }
    }
    stop_timer(&time);
    printf("ikj loop order took %.6f ms\n", time);

    //jik
    start_timer();
    for (j=0; j<N; j++) {
        for (i=0; i<N; i++) {
            for (k=0; k<N; k++) {
                C[i*N+j] += A[i*N+k] * B[k*N+j];
            }
        }
    }
    stop_timer(&time);
    printf("jik loop order took %.6f ms\n", time);

    //jki
    start_timer();
    for (j=0; j<N; j++) {
        for (k=0; k<N; k++) {
            for (i=0; i<N; i++) {
                C[i*N+j] += A[i*N+k] * B[k*N+j];
            }
        }
    }
    stop_timer(&time);
    printf("jki loop order took %.6f ms\n", time);

    //kij
    start_timer();
    for (k=0; k<N; k++) {
        for (i=0; i<N; i++) {
            for (j=0; j<N; j++) {
                C[i*N+j] += A[i*N+k] * B[k*N+j];
            }
        }
    }
    stop_timer(&time);
    printf("kij loop order took %.6f ms\n", time);

    //kji
    start_timer();
    for (k=0; k<N; k++) {
        for (j=0; j<N; j++) {
            for (i=0; i<N; i++) {
                C[i*N+j] += A[i*N+k] * B[k*N+j];
            }
        }
    }
    stop_timer(&time);
    printf("kji loop order took %.6f ms\n", time);

    //tiled
    int BS=40;
    start_timer();
    for (i=0; i<N/BS; i++) {
        for (k=0; k<N/BS; k++) {
            for (j=0; j<N/BS; j++) {

                for (int ib=0; ib<BS; ib++) {
                    for (int kb=0; kb<BS; kb++) {
                        for (int jb=0; jb<BS; jb++) {
                            C[(i*BS+ib)*N+j*BS+jb] += A[(i*BS+ib)*N+k*BS+kb] * B[(k*BS+kb)*N+j*BS+jb];
                        }
                    }
                }

            }
        }
    }
    stop_timer(&time);
    printf("tiled loop order took %.6f ms\n", time);



    return 0;
}
