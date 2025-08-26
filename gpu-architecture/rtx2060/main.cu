#include <stdio.h>

#define TOTAL_THREADS 262144

__global__ void kernel(float *data, int N) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N) {
        float x = 0;
        // arithmetic operation that will not get optimised by the compiler
        for (int j = 0; j < 1000000; j++) {
            x = x * 2 + 0.5;    
        }
        data[idx] = x;
    }
}

float run_test(int numThreadsPerBlock) {
    // size of data
    int N = TOTAL_THREADS;
    int numBlocks = TOTAL_THREADS/numThreadsPerBlock;
    // int numBlocks = 20;

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
        kernel<<<1, 1>>>(data_d, N);
    }

    cudaEventRecord(start);
    kernel<<<numBlocks, numThreadsPerBlock>>>(data_d, N);
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
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Device %d: %s\n", device, prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Warp size: %d\n", prop.warpSize);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Number of SMs: %d\n", prop.multiProcessorCount);
    printf("\n");


    printf("Total number of threads: %d\n\n", TOTAL_THREADS);
    printf("Threads per block\tTime (ms)\n");
    for(int numThreadsPerBlock = 1; numThreadsPerBlock <= 1024; numThreadsPerBlock*=2) {
        float elapsed = run_test(numThreadsPerBlock);
        printf("%d\t\t\t%0.2f\n", numThreadsPerBlock, elapsed);
    }

    printf("\n");

    // 1020 is max number of threads per block on RTX 2060
    int numThreadsPerBlock = 1024;
    float elapsed = run_test(numThreadsPerBlock);
    printf("%d\t\t\t%0.2f\n", numThreadsPerBlock, elapsed);

    printf("\n");
    
    int threadsPerBlock = 32;
    size_t dynamicMem = 0;

    int numBlocksPerSM;
    cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM,
        kernel,
        threadsPerBlock,
        dynamicMem
    );

    printf("\n");
    printf("Blocks per SM: %d\n", numBlocksPerSM);
    printf("Total active blocks on GPU: %d\n", 
           numBlocksPerSM * prop.multiProcessorCount);
    
    //  for (int i = 0; i < N; i++) {
    //     if (i % 16 == 0 && i != 0) {
    //        printf("\n");
    //     }
    //     printf("%3d ", data[i]);
    // }
    // printf("\n");   
    return 0;
}
