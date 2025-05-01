#include <stdio.h> 
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "include/libsais.h"

int main(int argc, char const* argv[])
{
    FILE* file = fopen(argv[1], "rb");
    if (!file) {
        printf("Error opening file\n");
        return 1;
    }
    fseek(file, 0, SEEK_END);
    size_t size = ftell(file);
    rewind(file);

    uint8_t* t = (uint8_t*)malloc(size);
    if (!t) {
        printf("Error allocating\n");
        return 1;
    }

    uint32_t* sa = (uint32_t*)malloc(sizeof(uint32_t) * size);

    if (!sa) {
        printf("Error allocating sa\n");
        return 1;
    }

    fread(t, 1, size, file);
    clock_t start, end;
    float time;


    start = clock();

    int err = libsais(t, sa, size, 0, NULL);

    end = clock();


    time = ((float)(end - start)) / 1000.f;
    const char* fileName = strrchr(argv[1], '/');
    if (fileName) {
        fileName++;
    }
    else {
        fileName = argv[1];
    }
    switch (err) {
    case 0:
        printf("libsais,%s,%f\n", fileName, time);
        break;
    default:
        printf("Error: %d\n", err);
        return 1;
        break;
    }
    free(sa);
    free(t);
    fclose(file);
    return 0;
}
