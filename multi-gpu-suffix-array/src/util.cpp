#include "util.h"

#include <cstdarg>
#include <cstdlib>
#include <cstdio>

#define ERROR_STREAM stdout

static const size_t ERROR_MSG_BUFFER_SIZE = 2048;

void error(const char *format, ...)
{
    char buff[ERROR_MSG_BUFFER_SIZE+1];
    va_list arg;
    va_start(arg, format);
    vsnprintf(buff, ERROR_MSG_BUFFER_SIZE, format, arg);
    va_end(arg);
    fprintf(ERROR_STREAM, "ERROR: %s\n", buff);
    exit(EXIT_FAILURE);
}

void _handle_assert(const char *expression, const char *_file_, const int _line_, const char* function)
{
    error("Assertion failed in %s at line %d (%s):\n%s", _file_, _line_, function, expression);
}

void _handle_assert(const char *expression, const char *_file_, const int _line_, const char* function, const char* msg)
{
    error("Assertion failed in %s at line %d (%s):\n%s\n%s", _file_, _line_, function, expression, msg);
}
