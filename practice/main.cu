#include <stdio.h>

__global__ void add(int a, int b, int *c) {
    *c = a + b;
}

void handleError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("Error: %s at line %d in file %s\n", cudaGetErrorString(error), line, file);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(error) (handleError(error, __FILE__, __LINE__))

int main(void) {
    int c;
    int *c_d;
    HANDLE_ERROR(cudaMalloc((void**)&c_d, sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(&c, c_d, sizeof(int), cudaMemcpyDeviceToHost));
    printf("hello world\n");

    return 0;
}
