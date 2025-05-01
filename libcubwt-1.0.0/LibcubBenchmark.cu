#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdint> 
#include <cstring>
#include <chrono>
#include <string>
#include "libcubwt.cuh"
#include "io.cuh"

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Bad args" << std::endl;
        return 1;
    }
    size_t size = 0;
    uint8_t* buffer = read(argv[1], size);
    if (!buffer) {
        std::cerr << "Error buffer" << std::endl;
        return 1;
    }

    void* deviceStorage;
    int64_t allocError = libcubwt_allocate_device_storage(&deviceStorage, size);

    if (allocError == LIBCUBWT_NO_ERROR)
    {
        uint32_t* sa = new uint32_t[size];


        auto start = std::chrono::high_resolution_clock::now();

        int64_t err = libcubwt_sa(deviceStorage, buffer, sa, size);

        auto stop = std::chrono::high_resolution_clock::now();

        delete[] sa;
        if (err == LIBCUBWT_NO_ERROR)
        {
            //for (int i = 0; i < len; i++)
            //{
                //printf("Suffix: %c ISA: %u\n", text[i], isa[i]);
            //}
        }
        else
        {
            std::cerr << "Error: " << err << std::endl;
            return 1;
        }

        auto duration = (float)(std::chrono::duration_cast<std::chrono::microseconds>(stop - start)).count() / 1000.f;

        write(argv[2], argv[1], duration);
    }
    else {
        std::cerr << "Error during allocation: " << allocError << std::endl;
        return 1;
    }
    libcubwt_free_device_storage(deviceStorage);
    delete[] buffer;
    return 0;
}