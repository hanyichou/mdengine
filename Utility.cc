#include "Utility.h"
#include "ARBDException.h"
namespace Utility
{
    void remove_comment(std::string& line, const std::string& header)
    {
        size_t begin = line.find(header);
        if(begin != std::string::npos)
            line.erase(begin);
    }
    void trim_both_sides(std::string& line)
    {
        //auto it = line.begin();
        std::string::iterator it = line.begin();
        for(; it != line.end() && std::isspace(*it); ++it)
        {
                
        }
        line.erase(line.begin(), it);
    
        int count = line.length();
        std::string::reverse_iterator it1 = line.rbegin();
        for(; it1 != line.rend() && std::isspace(*it1); ++it1)
        {
            --count;
        }
        line.erase(count);
    }
    void split(std::vector<std::string>& tokens,const std::string& s, const std::string& delim)
    {
        tokens.clear();
        std::size_t old_pos = 0, new_pos = 0;
        while((old_pos = s.find_first_not_of(delim, old_pos)) != std::string::npos)
        {
            new_pos = s.find_first_of(delim, old_pos);
            std::string sub_string = s.substr(old_pos, new_pos-old_pos);
            trim_both_sides(sub_string);
            tokens.push_back(sub_string);
            old_pos = new_pos;
        }

    }

    std::ifstream& get(std::string& buff, std::ifstream& infile, const std::string& comment_headers, const std::string& delim)
    {
        do
        {
            std::getline(infile, buff, delim[0]);
            trim_both_sides(buff);
            remove_comment(buff, comment_headers);
            if(!buff.empty())
                return infile;
            else
                buff.clear();
        } while((!infile.fail()) or (!infile.eof()));
        if(infile.eof())
            return infile;
        else
            throw ARBD_Exception(IFSTREAM_ERROR);
    }

    bool get_line(std::string& buff, std::ifstream& infile, const std::string& token, const std::string& comment_headers, const std::string& delim)
    {
        while(Utility::get(buff, infile, comment_headers, delim))
        {
            if(buff.find(token) != std::string::npos)
                return true;
            buff.clear();
        }
        return false;
    }

    template<>
    double string_to_number<double>(const std::string& s)
    {
        return std::strtod(s.c_str(), NULL);
    }

    template<>
    float string_to_number<float>(const std::string& s)
    {
        return (float)std::strtod(s.c_str(), NULL);
    }

    template<>
    int string_to_number<int>(const std::string& s)
    {
        return (int)std::strtol(s.c_str(), NULL, 10);
    }

    template<>
    unsigned long long string_to_number<unsigned long long>(const std::string& s)
    {
        return (unsigned long long)std::strtoull(s.c_str(), NULL, 10);
    }
}
