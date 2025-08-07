#include "stdio.h"

#define WIDTH 2
#define HEIGHT 2
#define IMAGE_SIZE (WIDTH * HEIGHT)

int main(int argc, char *argv[])
{

    unsigned char r[IMAGE_SIZE] = {255, 0, 0, 255};
    unsigned char g[IMAGE_SIZE] = {0, 255, 0, 255};
    unsigned char b[IMAGE_SIZE] = {0, 0, 255, 255};
    unsigned char grey[IMAGE_SIZE];
    
    for (int i = 0; i < IMAGE_SIZE; i++) {
        grey[i] = r[i]*3/10 + g[i]*6/10 + b[i]*2/10;
        printf("Greyscale value for index [%d] =  %d\n", i,grey[i]);
    }

    FILE *fptr;
    fptr = fopen("output.pgm", "w");
    fprintf(fptr, "P2\n");
    fprintf(fptr, "# test image\n");
    fprintf(fptr, "%d %d\n", WIDTH, HEIGHT);
    fprintf(fptr, "255\n");

    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        fprintf(fptr, "%d ", grey[i]);
        if ((i + 1) % WIDTH == 0) fprintf(fptr, "\n");
    }

    fclose(fptr);


    return 0;
}
