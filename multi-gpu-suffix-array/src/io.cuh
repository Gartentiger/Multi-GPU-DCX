#ifndef IO_H_
#define IO_H_
#include <cstddef>

#include "suffix_types.h"

size_t read_file_into_host_memory(char** contents, const char* path, size_t& real_len,
    size_t padd_to, size_t maxLength, uint num_gpus, char padd_c = 0);
void write_array(const char* ofile, const sa_index_t* sa, size_t len);

#endif /* IO_H_ */
