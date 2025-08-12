#include <stdio.h>

#define TOTAL_THREADS 262144

__global__ void kernel(float *data) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float x = 0;
    // arithmetic operation that will not get optimised by the compiler
    for (int i = 0; i < 1000000; i++) {
        x = x * 2 + 0.5;    
    }
    data[idx] = x;
}

float run_test(int numThreadsPerBlock) {
    // size of data
    int N = 128;
    int numBlocks = TOTAL_THREADS/numThreadsPerBlock;

    // 1. allocate mem
    float *data = (float*) malloc(N*sizeof(float));
    float *data_d;
    cudaMalloc((void**)&data_d, N*sizeof(float));
    
    // 2. copy mem cpu to gpu
    cudaMemcpy(data_d, data, N*sizeof(float), cudaMemcpyHostToDevice);
    // time computation speed
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warmup run
    for (int i = 0; i < 100; i++) {
        kernel<<<1, 1>>>(data_d);
    }

    cudaEventRecord(start);
    kernel<<<numBlocks, numThreadsPerBlock>>>(data_d);
    cudaEventRecord(stop);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Error launching the kernel: %s\n", cudaGetErrorString(error));
    }
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);

    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 4. copy mem gpu to cpu
    cudaMemcpy(data, data_d, N*sizeof(float), cudaMemcpyDeviceToHost);
    // 5. del mem
    cudaFree(data_d);
    free(data);

    return elapsed;
}

int main(int argc, char *argv[]) {

    printf("Total number of threads: %d\n\n", TOTAL_THREADS);
    printf("Threads per block\tTime (ms)\n");
    // for(int numThreadsPerBlock = 1; numThreadsPerBlock <= 1024; numThreadsPerBlock*=2) {
    //     float elapsed = run_test(numThreadsPerBlock);
    //     printf("%d\t\t\t%0.2f\n", numThreadsPerBlock, elapsed);
    // }

    // printf("\n");

    // 1020 is max number of threads per block on RTX 2060
    int numThreadsPerBlock = 1025;
    float elapsed = run_test(numThreadsPerBlock);
    printf("%d\t\t\t%0.2f\n", numThreadsPerBlock, elapsed);

    printf("\n");


    //  for (int i = 0; i < N; i++) {
    //     if (i % 16 == 0 && i != 0) {
    //        printf("\n");
    //     }
    //     printf("%3d ", data[i]);
    // }
    // printf("\n");   
    return 0;
}
