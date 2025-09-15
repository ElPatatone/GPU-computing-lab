#include <stdio.h>

// macro for handling errors more easily in cuda
#define HANDLE_ERROR(error) (handleError(error, __FILE__, __LINE__))

void handleError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("Error: %s at line %d in file %s\n", cudaGetErrorString(error), line, file);
        exit(EXIT_FAILURE);
    }
}

void returnDeviceProperties() {
    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
    printf("Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Clock rate: %d\n", prop.clockRate);

    printf("Memory information\n");
    printf("Total global memory: %ld\n", prop.totalGlobalMem);
    printf("Total constant memory: %ld\n", prop.totalConstMem);

    printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
    printf("Shared memory per MP: %ld\n", prop.sharedMemPerMultiprocessor);
    printf("Registers per MP: %d\n", prop.regsPerMultiprocessor);
    printf("Threads in warp: %d\n", prop.warpSize);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
}

__global__ void add(int a, int b, int* c) {
    *c = a + b;
    printf("2+7 = %d\n", *c);
}

int main(void) {
    int c; 
    int* c_d;

    HANDLE_ERROR(cudaMalloc((void**)&c_d, sizeof(int)));
    add<<<1,1>>> (2, 7, c_d);
    HANDLE_ERROR(cudaMemcpy(&c, c_d, sizeof(int), cudaMemcpyDeviceToHost));

    printf("2+7 = %d\n", c);
    cudaFree(c_d);

    returnDeviceProperties();
    return 0;
}
