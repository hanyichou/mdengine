#ifndef UTILITY_H_
#define UTILITY_H_
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <cstdlib>
namespace Utility
{
    void remove_comment(std::string& line, const std::string& header);
    void trim_both_sides(std::string& line);
    void split(std::vector<std::string>& tokens,const std::string& s, const std::string& delim=std::string(" \t"));

    std::ifstream& get(std::string& buff, std::ifstream& infile, const std::string& comment_headers=std::string(""), const std::string& delim=std::string("\n"));
    bool get_line(std::string& buff, std::ifstream& infile, const std::string& token, const std::string& comment_headers=std::string(""), const std::string& delim=std::string("\n"));

    template<typename T>
    T string_to_number(const std::string& s);
}

#endif
