#include "stdio.h"

#define WIDTH 32
#define HEIGHT 32
#define IMAGE_SIZE (WIDTH * HEIGHT)

void createImage(FILE* fptr, unsigned char* output) {
    fprintf(fptr, "P2\n");
    fprintf(fptr, "# test image\n");
    fprintf(fptr, "%d %d\n", WIDTH, HEIGHT);
    fprintf(fptr, "255\n");

    for (int i = 0; i < IMAGE_SIZE; i++) {
        fprintf(fptr, "%d ", output[i]);
        if ((i + 1) % WIDTH == 0) fprintf(fptr, "\n");
    }

    fclose(fptr);
}

void writePPM(FILE* fptr, unsigned char* rgb) {
    fprintf(fptr, "P6\n");
    fprintf(fptr, "%d %d\n", WIDTH, HEIGHT);
    fprintf(fptr, "255\n");

    fwrite(rgb, sizeof(unsigned char), WIDTH * HEIGHT * 3, fptr); 
    fclose(fptr);
}

int main(int argc, char *argv[]) {

    // interleaved RGB format unsigned char rgb[IMAGE_SIZE*3] = { 255, 0, 0,
    //                                     0, 255, 0,
    //                                     0, 0, 255,
    //                                     255, 255, 255 };


    // planar RGB format
    unsigned char r[IMAGE_SIZE] = {255, 0, 0, 255};
    unsigned char g[IMAGE_SIZE] = {0, 255, 0, 255};
    unsigned char b[IMAGE_SIZE] = {0, 0, 255, 255};

    unsigned char greyPlanar[IMAGE_SIZE];
    unsigned char greyInterleaved[IMAGE_SIZE];
    
    // planar RGB processing
    for (int i = 0; i < IMAGE_SIZE; i++) {
        greyPlanar[i] = (unsigned char)(r[i]*0.299 + g[i]*0.587 + b[i]*0.114);
        printf("Greyscale value for index [%d] =  %d\n", i, greyPlanar[i]);
    }

    unsigned char rgb[IMAGE_SIZE*3];

    for (int i = 0; i < IMAGE_SIZE; i++) {
        rgb[i*3] = i % 256;
        rgb[i*3 + 1] = (i*2) % 256;
        rgb[i*3 + 2] = (i*3) % 256;
    }
    
    // interleaved RGB processing
    for (int i = 0; i < IMAGE_SIZE; i++) {
        int offset = i * 3;
        unsigned char r = rgb[offset];
        unsigned char g = rgb[offset + 1];
        unsigned char b = rgb[offset + 2];

        greyInterleaved[i] = r*0.299 + g*0.587 + b*0.114;
        printf("Greyscale value for index [%d] =  %d\n", i, greyInterleaved[i]);
    }

    FILE *fptr1 = fopen("greyPlanar.pgm", "w");
    FILE *fptr2 = fopen("greyInterleaved.pgm", "w");
    FILE *fptr3 = fopen("inputColor.ppm", "wb");

    if (!fptr1 || !fptr2 || !fptr3) {
        printf("Failed to open the files");
        return 1;
    }

    createImage(fptr1, greyPlanar);
    createImage(fptr2, greyInterleaved);
    writePPM(fptr3, rgb);

    return 0;
}
