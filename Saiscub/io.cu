#include "io.cuh"
#include <cstdio>
#include <fstream>
#include <iostream>


uint8_t* read(char* path, size_t& size) {
    std::ifstream inFile(path, std::ios::binary | std::ios::ate);
    if (!inFile.is_open()) {
        std::cerr << "Error opening input file" << std::endl;
        return NULL;
    }
    size = inFile.tellg();
    inFile.seekg(0, std::ios::beg);
    auto buffer = new uint8_t[size];
    if (!inFile.read(reinterpret_cast<char*>(buffer), size)) {
        std::cerr << "Error reading input file" << std::endl;
        return NULL;
    }
    inFile.close();

    return buffer;
}

int write(char* OutPath, uint32_t* sa, size_t size) {
    std::ofstream outFile(OutPath, std::ios::app);
    if (!outFile.is_open()) {
        std::cerr << "Error opening output file!" << std::endl;
        return 1;
    }
    for (size_t i = 0; i < size; i++) {
        outFile << sa[i];
    }
    outFile.close();
    return 0;
}

int write(char* OutPath, int32_t* sa, size_t size) {
    std::ofstream outFile(OutPath, std::ios::app);
    if (!outFile.is_open()) {
        std::cerr << "Error opening output file!" << std::endl;
        return 1;
    }
    for (size_t i = 0; i < size; i++) {
        std::cout << sa[i] << std::endl;
        outFile << sa[i];
        outFile << " \n";
    }
    outFile.close();
    return 0;
}