#pragma once
#include <cstddef>
#include <cstdint> 

uint8_t* read(char* path, size_t& size);

int write(char* path, float duration);