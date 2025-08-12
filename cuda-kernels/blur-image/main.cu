// good example but I was getting too caught up in the pixel processing 
// and right now the main goal is to understand GPU architecture better
#define WIDTH 256
#define HEIGHT 256
#define IMAGE_SIZE (WIDTH*HEIGHT)

__global__ void blurImage_kernel(unsigned char *image, unsigned char *blurred) {
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x; 
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y; 
    
    unsigned int i = row*WIDTH + col;

    if (col < WIDTH && row < HEIGHT) {
        // blurred[i] = ;
    }

}

void bludImage_gpu(unsigned char *image, unsigned char *blurred) {
    //1. allocate memory
    unsigned char *image_d, *blurred_d;
    cudaMalloc((void**)&image_d, sizeof(unsigned char)*IMAGE_SIZE);
    cudaMalloc((void**)&blurred_d, sizeof(unsigned char)*IMAGE_SIZE);

    // 2. copy cpu to gpu
    cudaMemcpy(image_d, image, sizeof(unsigned char)*IMAGE_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(blurred_d, blurred, sizeof(unsigned char)*IMAGE_SIZE, cudaMemcpyHostToDevice);

    // 3. do stuff with kernel
    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((WIDTH + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x, 
                   (HEIGHT + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y);

    blurImage_kernel<<<numBlocks, numThreadsPerBlock>>>(image, blurred);
    // 4. copy back
    cudaMemcpy(blurred, blurred_d, sizeof(unsigned char)*IMAGE_SIZE, cudaMemcpyDeviceToHost);
    // 5. deallocate
    cudaFree(image_d);
    cudaFree(blurred_d);
} 



int main(int argc, char *argv[]) {

    unsigned char *image = (unsigned char*) malloc(sizeof(unsigned char)*IMAGE_SIZE);
    unsigned char *blurred = (unsigned char*) malloc(sizeof(unsigned char)*IMAGE_SIZE);
    
}
