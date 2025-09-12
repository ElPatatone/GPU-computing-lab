#include <stdio.h>

__global__ void add(int a, int b, int *c) {
    *c = a + b;
}

int main(void) {
    int c;
    int *c_d;
    cudaMalloc((void**)&c_d, sizeof(int));
    cudaMemcpy(&c, c_d, sizeof(int), cudaMemcpyDeviceToHost);
    printf("hello world\n");
    return 0;
}
