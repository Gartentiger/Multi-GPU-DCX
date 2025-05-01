#pragma once
#include <cstddef>
#include <cstdint> 

int read(char* path, const uint8_t** content, size_t& size);

int write(char* path, float duration);