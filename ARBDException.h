/*
 * ARBDException class handles the 
 * run-time exception.
 * Han-Yi Chou
 */

#ifndef ARBDEXCEPTION_H_
#define ARBDEXCEPTION_H_

#include <string>
#include <cstdarg>
#include <exception>
class ARBDException : public std::exception 
{
  private:
    std::string _error;
    std::string sformat(const std::string &fmt, va_list &ap);

  public:
    ARBDException(const std::string& location, const std::string &ss, ...);
    virtual const char* what() const throw();
};
#include "common_macros.h"
#define ARBD_Exception(...) ARBDException(LOCATION, __VA_ARGS__)
//use illegal instruction to abort; used in functions defined both in __host__ and __device__
#if 0
#define ARBD_CudaException(...) \
printf("Run-time exception occurs at %s: ", LOCATION); \
printf(__VA_ARGS__);
//TODO I want to add asm("trap;") but the compiling does not work
asm("trap;");
#endif
#endif
