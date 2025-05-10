#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdint> 
#include <numeric>
#include <random>
#include <thread>
#include <vector>
#include <cstring>
#include <chrono>
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


int main(int argc, char** argv)
{
    using namespace kamping;
    kamping::Environment e;
    Communicator         comm;
    std::vector<int>     input(comm.size());
    std::iota(input.begin(), input.end(), 0);
    std::vector<int> output;

    auto sleep_some_time = [&]() {
        static std::mt19937                gen(static_cast<std::mt19937::result_type>(comm.rank() + 17) * 1001);
        std::uniform_int_distribution<int> distrib(50, 10'000);
        const std::chrono::microseconds    sleep_duration{ distrib(gen) };
        std::this_thread::sleep_for(sleep_duration);
        };
    // Get timer singleton. Alternatively you can also instantiate a new timer.
    auto& t = kamping::measurements::timer();

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

        //auto start = std::chrono::high_resolution_clock::now();
        t.synchronize_and_start("algorithm");

        int64_t err = libcubwt_sa(deviceStorage, buffer, sa, size);
        t.stop();
        //auto stop = std::chrono::high_resolution_clock::now();


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
        std::ofstream outFile(argv[2], std::ios::app);
        t.aggregate_and_print(
            kamping::measurements::SimpleJsonPrinter{ outFile, {std::pair("first_config_key", "first_config_value")} }
        );
        std::cout << std::endl;
        t.aggregate_and_print(kamping::measurements::FlatPrinter{});
        std::cout << std::endl;
        //auto duration = (float)(std::chrono::duration_cast<std::chrono::microseconds>(stop - start)).count() / 1000.f;
        //write(argv[2], argv[1], duration);
    }
    else {
        std::cerr << "Error during allocation: " << allocError << std::endl;
        return 1;
    }
    libcubwt_free_device_storage(deviceStorage);
    delete[] buffer;
    return 0;
}