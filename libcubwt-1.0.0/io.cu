#include "io.cuh"
#include <cstdio>
#include <fstream>
#include <iostream>


int read(char* path, uint8_t** content, size_t& size) {
    std::ifstream inFile(path, std::ios::binary | std::ios::ate);
    if (!inFile.is_open()) {
        std::cerr << "Error opening input file" << std::endl;
        return 1;
    }
    size = inFile.tellg();
    inFile.seekg(0, std::ios::beg);
    *content = new uint8_t[size];
    if (!inFile.read(reinterpret_cast<char*>(content), size)) {
        std::cerr << "Error reading input file" << std::endl;
        return 1;
    }
    inFile.close();

    return 0;
}

int write(char* path, float duration) {
    std::ofstream outFile(path, std::ios::app);
    if (!outFile.is_open()) {
        std::cerr << "Error opening output file!" << std::endl;
        return 1;
    }
    auto stringPath = ((std::string)path);
    int pos = stringPath.find_last_of("/\\");
    auto fileName = (pos == std::string::npos) ? path : stringPath.substr(pos + 1);
    outFile << "Libcubwt," << fileName << "," << duration << std::endl;
    printf("libcubwt,%s,%f\n", fileName.c_str(), duration);
    outFile.close();
    return 0;
}