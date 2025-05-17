#pragma once
#include <cstddef>
#include <cstdint> 

uint8_t* read(char* path, size_t& size);

int write(char* OutPath, uint32_t* sa, size_t size);
int write(char* OutPath, int32_t* sa, size_t size);