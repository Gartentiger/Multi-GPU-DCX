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
    uint8_t* buffer2;
    size_t size = 0;
    read(args[1], &buffer2, size);

    std::ifstream inFile(args[1], std::ios::binary | std::ios::ate);
    if (!inFile.is_open()) {
        std::cerr << "Error opening input file" << std::endl;
        return 1;
    }
    auto size = inFile.tellg();
    inFile.seekg(0, std::ios::beg);

    uint8_t* buffer = new uint8_t[size];

    if (!inFile.read(reinterpret_cast<char*>(buffer), size)) {
        std::cerr << "Error reading input file" << std::endl;
        return 1;
    }
    inFile.close();

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

        std::ofstream outFile(args[2], std::ios::app);
        if (!outFile.is_open()) {
            std::cerr << "Error opening output file!" << std::endl;
            return 1;
        }
        auto stringPath = ((std::string)args[2]);
        int pos = stringPath.find_last_of("/\\");
        auto fileName = (pos == std::string::npos) ? args[2] : stringPath.substr(pos + 1);
        outFile << "Libcubwt," << fileName << "," << duration << std::endl;
        printf("libcubwt,%s,%f\n", fileName.c_str(), duration);
        outFile.close();
    }
    else {
        std::cerr << "Error during allocation: " << allocError << std::endl;
        return 1;
    }
    libcubwt_free_device_storage(deviceStorage);
    delete[] buffer;
    return 0;
}