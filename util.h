/* Utilitary functions & definitions
*
*/

#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <memory>
#include <vector>

typedef unsigned char ubyte;

/*Can be used as a destructor in an std::unique_ptr to a FILE
 * eg: std::unique_ptr<FILE, FileDeleter> f;
 */
struct FileDeleter{
  void operator()(FILE* f) const {
    //puts("Closing file...\n");
    fclose(f);
  }
};

/* Conver 4 bytes in MSB order to an LSB int */
int MSBtoLSB(const char* msbBytes);

#endif //UTIL_H
