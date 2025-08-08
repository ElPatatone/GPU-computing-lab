#include <time.h>
#include <stdio.h>

__global__ void vectorAddKernel(int *z, int N) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; 
    if (i < N) {
        z[i] = threadIdx.x;
    }
};

void display(int *z, int N, const unsigned int numBlocks, const unsigned int numThreadsPerBlock) {
    int *z_d;
    cudaMalloc((void**)&z_d, sizeof(int)*N);

    // const unsigned int numThreadsPerBlock = 128;
    // const unsigned int numBlocks = (N + numThreadsPerBlock - 1) /numThreadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    vectorAddKernel<<<numBlocks, numThreadsPerBlock>>>(z_d, N);
    cudaEventRecord(stop, 0);
    cudaError_t err = cudaGetLastError();  // Check launch errors immediately
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();  // Wait for kernel to finish
    
    cudaError_t error = cudaEventSynchronize(stop);
    if (error != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(err));
    }

    float timeTaken = 0;
    cudaEventElapsedTime(&timeTaken, start, stop);
    printf("Kernel computation took: %.10f seconds\n", timeTaken/1000);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(z, z_d, sizeof(int)*N, cudaMemcpyDeviceToHost);
    cudaFree(z_d);
}


int main (int argc, char *argv[]) {

    // int N = (argc > 1) ? atoi(argv[1]) : (1 << 25);
    int N = 128;

    int* z = (int*) malloc(sizeof(int)*N);

    const unsigned int numBlocks = 4;
    const unsigned int numThreadsPerBlock = 32;

    display(z, N, numBlocks, numThreadsPerBlock);


    printf("Total number of threads: %d\n", numBlocks*numThreadsPerBlock);
    printf("Number of blocks: %d\n", numBlocks);
    printf("Number of threads in the blocks: %d\n", numThreadsPerBlock);

    for (int i = 0; i < N; i++) {
        if (i % 16 == 0 && i != 0) {
           printf("\n");
        }
        printf("%3d ", z[i]);
    }
    printf("\n");

    free(z);

    return 0;
}
