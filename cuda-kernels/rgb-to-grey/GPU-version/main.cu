#include <time.h>
#include <stdio.h>

#define WIDTH 8192
#define HEIGHT 8192
#define IMAGE_SIZE (WIDTH * HEIGHT)

__global__ void rgbTogrey_kernel(unsigned char *r, unsigned char *g, 
                                 unsigned char *b, unsigned char *grey) {

    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    // gets the individual pixel in the image
    unsigned int i = row*WIDTH + col;
    // boundary checks
    if (col < WIDTH && row < HEIGHT) {
        grey[i] = (unsigned char)(r[i]*0.299 + g[i]*0.587 + b[i]*0.114);
    }
}

void rgbTogrey_cpu(unsigned char *r, unsigned char *g,
                   unsigned char *b, unsigned char *grey) {

    // planar RGB processing
    for (int i = 0; i < IMAGE_SIZE; i++) {
        grey[i] = (unsigned char)(r[i]*0.299 + g[i]*0.587 + b[i]*0.114);
    }
}


void rgbTogrey_gpu(unsigned char *r, unsigned char *g,
                   unsigned char *b, unsigned char *grey) {

    // 1. allocate gpu memory
    unsigned char *r_d, *g_d, *b_d, *grey_d;
    
    cudaMalloc((void**)&r_d, sizeof(unsigned char)*IMAGE_SIZE);
    cudaMalloc((void**)&g_d, sizeof(unsigned char)*IMAGE_SIZE);
    cudaMalloc((void**)&b_d, sizeof(unsigned char)*IMAGE_SIZE);
    cudaMalloc((void**)&grey_d, sizeof(unsigned char)*IMAGE_SIZE);

    // 2. copy data from cpu to gpu
    cudaMemcpy(r_d, r, sizeof(unsigned char)*IMAGE_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(g_d, g, sizeof(unsigned char)*IMAGE_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, sizeof(unsigned char)*IMAGE_SIZE, cudaMemcpyHostToDevice);

    // 3. run the kernel
    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((WIDTH + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x, 
                   (HEIGHT + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    rgbTogrey_kernel<<<numBlocks, numThreadsPerBlock>>>(r_d, g_d, b_d, grey_d);
    cudaEventRecord(stop, 0);

    cudaError_t err = cudaGetLastError();  // Check launch errors immediately
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();  // Wait for kernel to finish
    
    cudaError_t error = cudaEventSynchronize(stop);
    if (error != cudaSuccess) {
        printf("[Error] Kernel failed to launch: %s", cudaGetErrorString(error));
    }

    float timeTaken = 0;
    cudaEventElapsedTime(&timeTaken, start, stop);
    printf("GPU computation took: %.10f seconds\n", timeTaken/1000);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 4. copy data from cpu to gpu
    cudaMemcpy(grey, grey_d, sizeof(unsigned char)*IMAGE_SIZE, cudaMemcpyDeviceToHost);

    // 5. free the gpu memory
    cudaFree(r_d);
    cudaFree(g_d);
    cudaFree(b_d);
    cudaFree(grey_d);
}

int main(int argc, char *argv[]) {

    // unsigned char rgb[IMAGE_SIZE*3];
    unsigned char *r = (unsigned char*) malloc(IMAGE_SIZE); 
    unsigned char *g = (unsigned char*) malloc(IMAGE_SIZE); 
    unsigned char *b = (unsigned char*) malloc(IMAGE_SIZE); 

    for (int i = 0; i < IMAGE_SIZE; i++) {
        r[i] = i % 256;
        g[i] = (i*2) % 256;
        b[i] = (i*3) % 256;
    }

    // unsigned char r[IMAGE_SIZE] = {255, 0, 0, 255};
    // unsigned char g[IMAGE_SIZE] = {0, 255, 0, 255};
    // unsigned char b[IMAGE_SIZE] ={0, 0, 255, 255};
    unsigned char *grey_cpu = (unsigned char*) malloc(IMAGE_SIZE);
    unsigned char *grey_gpu = (unsigned char*) malloc(IMAGE_SIZE);
    
    rgbTogrey_gpu(r, g, b, grey_gpu);

    clock_t start = clock();
    rgbTogrey_cpu(r, g, b, grey_cpu);
    clock_t stop = clock();

    float timeTaken = (float) (stop - start) / CLOCKS_PER_SEC;
    printf("CPU computation took: %.10f seconds\n", timeTaken);

    for (int i = 0; i < IMAGE_SIZE; i++) {
        if (grey_cpu[i] != grey_gpu[i]) {
            printf("Mismatch in the elements at index: %d\n", i);
            printf("grey_cpu: %d\n", grey_cpu[i]);
            printf("grey_gpu: %d\n", grey_gpu[i]);
        }
    }

    for (int i = 0; i < 10; i++) {
        printf("index: %d\n", i);
        printf("grey_gpu: %d\n", grey_gpu[i]);
        printf("grey_cpu: %d\n", grey_cpu[i]);
    }

    free(r);
    free(g);
    free(b);
    free(grey_cpu);
    free(grey_gpu);

    return 0;
}
