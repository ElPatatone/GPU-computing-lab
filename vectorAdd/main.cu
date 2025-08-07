#include <time.h>
#include <stdio.h>

__host__ __device__ float add(float a, float b) {
    return a + b;
}

void vectorAddCPU(float* x, float* y, float* z, int N) {
    clock_t start = clock();

    for (int i = 0; i < N; i++) {
        z[i] = add(x[i],  y[i]);
    }

    clock_t end = clock();

    double timeTaken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("CPU computation took: %.10f seconds\n", timeTaken);
}

__global__ void vectorAddKernel(float* x, float* y, float* z, int N) {

    // this approach to parallel programming is called single program multiple data
    // multiple threads executing the same program operating on a different set of data
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; 
    // if the index of the thread is out of bounds it will not be counted
    if (i < N) {
        z[i] = add(x[i], y[i]);
        // z[i] = i;
    }
};

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
    // call a GPU kernel function (launch a grid of threads)
    
    // N here is the size of the vectors, but it also is the maximum number of threads
    // we would need to have 1 thred per vector element.
    const unsigned int numThreadsPerBlock = 512;
    // we do ceiling divion to cover the case where N is not a multiple of the number.
    // without it the program would create less threads than we need.
    const unsigned int numBlocks = (N + numThreadsPerBlock - 1) /numThreadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    // each thread will execute this function
    vectorAddKernel<<<numBlocks, numThreadsPerBlock>>>(x_d, y_d, z_d, N);
    cudaEventRecord(stop, 0);

    cudaError_t err = cudaGetLastError();  // Check launch errors immediately
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();  // Wait for kernel to finish

    // tells the CPU to wait for the GPU to finish the events specified before 
    // it continues with executing the code.
    // we need to ask the CPU to wait for the GPU events to be done as the CPU has no 
    // access to these events and needs the GPU to tell it when it is done. 
    //
    // what happens is that the GPU commands que at this point only has the following:
    // start event record -> kernel launch -> stop event record 
    // so the CPU will wait until all those things are done before it moves on
    // all asynchronous operations go on the GPU que.
    cudaError_t error = cudaEventSynchronize(stop);
    if (error != cudaSuccess) {
        printf("[Error] Kernel failed to launch: %s", cudaGetErrorString(error));
    }

    float timeTaken = 0;
    cudaEventElapsedTime(&timeTaken, start, stop);

    // printf("GPU computation took: %.10f milliseconds\n", timeTaken);
    printf("GPU computation took: %.10f seconds\n", timeTaken/1000);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // 4. copy data from GPU memory to CPU memory
    cudaMemcpy(z, z_d, sizeof(float)*N, cudaMemcpyDeviceToHost);

    // 5. deallocate GPU memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
}


int main (int argc, char *argv[]) {

    int N = (argc > 1) ? atoi(argv[1]) : (1 << 25);
    // int N = 256;

    float* x = (float*) malloc(sizeof(float)*N);
    float* y = (float*) malloc(sizeof(float)*N);
    float* z = (float*) malloc(sizeof(float)*N);

    for (int i = 0; i < N; i++) {
        x[i] = rand();
        y[i] = rand();
    }

    clock_t CPUstart = clock();
    vectorAddCPU(x, y, z, N);
    clock_t CPUstop = clock();

    float CPUtimeTaken = (float)(CPUstop - CPUstart) / CLOCKS_PER_SEC;
    printf("CPU total time taken: %.10f seconds\n", CPUtimeTaken);


    // timing the full time taken including the allocation and deallocationg of 
    // memory
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);

    vectorAddGPU(x, y, z, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeTaken = 0;
    cudaEventElapsedTime(&timeTaken, start, stop);
    
    printf("GPU total time taken: %.10f seconds \n", timeTaken/1000);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // for (int i = 0; i < 256; i++) {
    //     printf("%f ", z[i]);
    //     if (i % 16 == 0 && i != 0) {
    //        printf("\n");
    //     }
    // }
    // printf("\n");

    free(x);
    free(y);
    free(z);

    return 0;
}
