#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <string>

#include <kamping/checking_casts.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/data_buffer.hpp>
#include <kamping/environment.hpp>
#include <kamping/measurements/printer.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/named_parameters.hpp>
#include <kamping/communicator.hpp>

#include "io.cuh"
#include "external/libcubwt/libcubwt.cuh"

int main(int argc, char **argv)
{
    using namespace kamping;
    kamping::Environment e;
    Communicator comm;

    auto &t = kamping::measurements::timer();

    if (argc < 3)
    {
        std::cerr << "Bad args" << std::endl;
        return 1;
    }
    for (int i = 0; i < argc - 2; i++)
    {
        size_t size = 0;
        uint8_t *buffer = read(argv[i + 2], size);
        if (!buffer)
        {
            std::cerr << "Error buffer" << std::endl;
            return 1;
        }

        void *deviceStorage;
        int64_t allocError = libcubwt_allocate_device_storage(&deviceStorage, size);
        if (allocError == LIBCUBWT_NO_ERROR)
        {
            uint32_t *sa = new uint32_t[size];

            auto stringPath = ((std::string)argv[i + 2]);
            int pos = stringPath.find_last_of("/\\");
            auto fileName = (pos == std::string::npos) ? argv[i + 2] : stringPath.substr(pos + 1);
            t.synchronize_and_start(fileName);

            int64_t err = libcubwt_sa(deviceStorage, buffer, sa, size);

            t.stop();

            delete[] sa;
            if (err != LIBCUBWT_NO_ERROR)
            {
                std::cerr << "Error: " << err << std::endl;
                return 1;
            }
        }
        else
        {
            std::cerr << "Error during allocation: " << allocError << std::endl;
            return 1;
        }
        libcubwt_free_device_storage(deviceStorage);
        delete[] buffer;
    }
    std::ofstream outFile(argv[1], std::ios::app);
    t.aggregate_and_print(
        kamping::measurements::SimpleJsonPrinter{outFile, {}});
    std::cout << std::endl;
    t.aggregate_and_print(kamping::measurements::FlatPrinter{});
    std::cout << std::endl;
    return 0;
}