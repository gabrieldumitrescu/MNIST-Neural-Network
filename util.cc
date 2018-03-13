/* Conver 4 bytes in MSB order to an LSB int */
#include "util.h"
int MSBtoLSB(const char* msbBytes)
{
    int result;
    char reverser[sizeof(int)];
    for(size_t i = 0; i<sizeof(int); i++)
    {
      reverser[i]=msbBytes[3-i];
    }
    result = *((int*)reverser);
    return result;
}
