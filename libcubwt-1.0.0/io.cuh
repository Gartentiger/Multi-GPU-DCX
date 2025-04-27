#pragma once
#include <cstddef>

int read();
size_t read_file_into_host_memory(char** contents, const char* path, size_t& real_len,
    size_t padd_to, char padd_c);