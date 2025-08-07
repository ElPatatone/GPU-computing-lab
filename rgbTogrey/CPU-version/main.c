#include "stdio.h"

#define WIDTH 2
#define HEIGHT 2
#define IMAGE_SIZE (WIDTH * HEIGHT)

void createImage(FILE* fptr, unsigned char* output) {
    fprintf(fptr, "P2\n");
    fprintf(fptr, "# test image\n");
    fprintf(fptr, "%d %d\n", WIDTH, HEIGHT);
    fprintf(fptr, "255\n");

    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        fprintf(fptr, "%d ", output[i]);
        if ((i + 1) % WIDTH == 0) fprintf(fptr, "\n");
    }

    fclose(fptr);
}

int main(int argc, char *argv[]) {

    unsigned char rgb[IMAGE_SIZE*3] = { 255, 0, 0,
                                        0, 255, 0,
                                        0, 0, 255,
                                        255, 255, 255 };



    unsigned char r[IMAGE_SIZE] = {255, 0, 0, 255};
    unsigned char g[IMAGE_SIZE] = {0, 255, 0, 255};
    unsigned char b[IMAGE_SIZE] = {0, 0, 255, 255};

    unsigned char greyPlanar[IMAGE_SIZE];
    unsigned char greyInterleaved[IMAGE_SIZE];
    
    // planar RGB
    for (int i = 0; i < IMAGE_SIZE; i++) {
        greyPlanar[i] = (int)(r[i]*0.299 + g[i]*0.587 + b[i]*0.114);
        printf("Greyscale value for index [%d] =  %d\n", i, greyPlanar[i]);
    }

    // interleaved RGB
    for (int i = 0; i < IMAGE_SIZE; i++) {
        int offset = i * 3;
        unsigned char r = rgb[offset];
        unsigned char g = rgb[offset + 1];
        unsigned char b = rgb[offset + 2];

        greyInterleaved[i] = r*0.299 + g*0.587 + b*0.114;
        printf("Greyscale value for index [%d] =  %d\n", i, greyInterleaved[i]);
    }

    FILE *fptr1;
    FILE *fptr2;
    fptr1 = fopen("greyPlanar.pgm", "w");
    fptr2 = fopen("greyInterleaved.pgm", "w");

    createImage(fptr1, greyInterleaved);
    createImage(fptr2, greyPlanar);


    return 0;
}
