#include "io.cuh"
#include <cstdio>
#include <fstream>
#include <iostream>


uint8_t* read(char* path, size_t* size) {
    std::ifstream inFile(path, std::ios::binary | std::ios::ate);
    if (!inFile.is_open()) {
        std::cerr << "Error opening input file" << std::endl;
        return NULL;
    }
    *size = inFile.tellg();
    inFile.seekg(0, std::ios::beg);
    auto buffer = new uint8_t[*size];
    if (!inFile.read(reinterpret_cast<char*>(buffer), *size)) {
        std::cerr << "Error reading input file" << std::endl;
        return NULL;
    }
    inFile.close();

    return buffer;
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