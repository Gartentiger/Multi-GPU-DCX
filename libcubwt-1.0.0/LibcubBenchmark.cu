#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdint> 
#include <cstring>
#include <chrono>
#include <string>
#include "libcubwt.cuh"
#include "io.cuh"

int main(int argc, char** args)
{
    if (argc != 3) {
        std::cerr << "Bad args" << std::endl;
        return 1;
    }

    std::ifstream inFile(args[1], std::ios::binary | std::ios::ate);
    if (!inFile.is_open()) {
        std::cerr << "Error opening input file" << std::endl;
        return 1;
    }
    auto size = inFile.tellg();
    inFile.seekg(0, std::ios::beg);
    printf("ff");
    uint8_t* buffer = new uint8_t[size];
    printf("t");
    if (!inFile.read(reinterpret_cast<char*>(buffer), size)) {
        std::cerr << "Error reading input file" << std::endl;
        return 1;
    }
    inFile.close();
    printf("t1");
    void* deviceStorage;
    int64_t allocError = libcubwt_allocate_device_storage(&deviceStorage, size);
    printf("t2");
    if (allocError == LIBCUBWT_NO_ERROR)
    {
        uint32_t* sa = new uint32_t[size];
        printf("t3");

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

        if (write(args[2], duration) != 0) {
            return 1;
        }
    }
    else {
        std::cerr << "Error during allocation: " << allocError << std::endl;
        return 1;
    }
    libcubwt_free_device_storage(deviceStorage);
    delete[] buffer;
    return 0;
}