#ifndef COMMON_MACROS_H_
#define COMMON_MACROS_H_
//some common exception and error macros to have unified output
#define ILL_FORMATTED_FILE "Ill formatted file %s"
#define FILE_OPEN_ERROR "Fail in opeing file %s"
#define IFSTREAM_ERROR "Error bit set in IO stream "
#define SIMULATION_TERMINATED "The simulation is terminated by exception %s"

#define S1(x) #x
#define S2(x) S1(x)
#define LOCATION __FILE__ "(" S2(__LINE__)")"
#define FORMAT std::string(" %s ")
//typedef int Backend;
#endif
