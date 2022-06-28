/*
 * ARBD exception handler
 * to handle the run-time exception.
 */
#include "ARBDException.h"

std::string ARBDException::sformat(const std::string &fmt, va_list &ap) 
{
    int size = 512;
    std::string str;
    while(1) 
    {
        str.resize(size);
        va_list ap_copy;
        va_copy(ap_copy, ap);
        int n = vsnprintf((char*)str.c_str(), size, fmt.c_str(), ap_copy);
        va_end(ap_copy);
        if (n > -1 && n < size) 
        {
            str.resize(n);
            return str;
        }
        if(n > -1)
            size = n + 1;
        else
            size *= 2;
    }
    return str;
}

ARBDException::ARBDException(const std::string& location, const std::string &ss, ...) 
{
        _error = location;
        _error += ": ";
	va_list ap;
	va_start(ap, ss);
	_error += sformat(ss, ap);
	va_end(ap);
}

const char* ARBDException::what() const throw()
{
    return _error.c_str();
}
