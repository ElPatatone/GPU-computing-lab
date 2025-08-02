#include <stdio.h>
#include <time.h>

void vectorAddCPU(float* x, float* y, float* z, int N) {
    for (int i = 0; i < N; i++) {
        z[i] = x[i] + y[i];
    }
}

void vectorAddGPU(float* x, float* y, float* z, int N) {
    // 1. allocate GPU memory
    float *x_d, *y_d, *z_d;

    cudaMalloc((void**)&x_d, sizeof(float)*N);
    cudaMalloc((void**)&y_d, sizeof(float)*N);
    cudaMalloc((void**)&z_d, sizeof(float)*N);

    // 2. copy data from CPU memory to GPU memory
    cudaMemcpy(x_d, x, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, sizeof(float)*N, cudaMemcpyHostToDevice);
    
    // 3. perform computation on GPU
    
    // 4. copy data from GPU memory to CPU memory
    cudaMemcpy(z_d, z, sizeof(float)*N, cudaMemcpyDeviceToHost);

    // 5. deallocate GPU memory
    cudaFree(&x_d);
    cudaFree(&y_d);
    cudaFree(&z_d);

}


int main (int argc, char *argv[]) {

    int N = (argc > 1) ? atoi(argv[1]) : (1 << 25);

    float* x = (float*) malloc(sizeof(float)*N);
    float* y = (float*) malloc(sizeof(float)*N);
    float* z = (float*) malloc(sizeof(float)*N);

    for (int i = 0; i < 1; i++) {
        x[i] = rand();
        y[i] = rand();
    }

    vectorAddCPU(x, y, z, N);

    return 0;
}
