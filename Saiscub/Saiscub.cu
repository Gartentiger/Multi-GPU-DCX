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

#ifdef libcub
#include "external/libcubwt/libcubwt.cuh"
#else
#include <libsais.h>
#endif

int main(int argc, char** argv)
{
    using namespace kamping;
    kamping::Environment e;
    Communicator comm;

    auto& t = kamping::measurements::timer();

    if (argc < 4)
    {
        std::cerr << "Bad args" << std::endl;
        return 1;
    }
    for (int i = 0; i < argc - 3; i++)
    {
        size_t size = 0;
        uint8_t* buffer = read(argv[i + 3], size);
        if (!buffer)
        {
            std::cerr << "Error buffer" << std::endl;
            return 1;
        }

        auto stringPath = ((std::string)argv[i + 3]);
        int pos = stringPath.find_last_of("/\\");
        auto fileName = (pos == std::string::npos) ? argv[i + 3] : stringPath.substr(pos + 1);

#ifdef libcub
        void* deviceStorage;
        int64_t allocError = libcubwt_allocate_device_storage(&deviceStorage, size);
        if (allocError == LIBCUBWT_NO_ERROR)
        {
            uint32_t* sa = new uint32_t[size];
            t.synchronize_and_start(fileName);

            int64_t err = libcubwt_sa(deviceStorage, buffer, sa, size);

            t.stop();


            if (err != LIBCUBWT_NO_ERROR)
            {
                std::cerr << "Error: " << err << std::endl;
                return 1;
            }

            write(argv[2], sa, size);

            delete[] sa;
        }
        else
        {
            std::cerr << "Error during allocation: " << allocError << std::endl;
            return 1;
        }
        libcubwt_free_device_storage(deviceStorage);
        delete[] buffer;
#else
        int32_t* sa = (int32_t*)malloc(sizeof(int32_t) * size);
        if (!sa) {
            std::cerr << "Error malloc" << std::endl;
            return 1;
        }

        t.synchronize_and_start(fileName);

        int32_t err = libsais(buffer, sa, size, 0, NULL);

        t.stop();

        if (err) {
            std::cerr << "Error: " << err << std::endl;
            return 1;
        }

        write(argv[2], sa, size);
        free(sa);
#endif
    }

    std::ofstream outFile(argv[1], std::ios::app);
    t.aggregate_and_print(
        kamping::measurements::SimpleJsonPrinter{ outFile, {} });
    std::cout << std::endl;
    t.aggregate_and_print(kamping::measurements::FlatPrinter{});
    std::cout << std::endl;
    return 0;
}