#include <vector>
#include <fstream>
#include "Utility.h"
#include "RestraintHarmonic.h"
#include "ARBDException.h"
RestraintHarmonic::RestraintHarmonic(const std::string& filename) : _param(nullptr), _index(nullptr), _num(0)
{
    std::ifstream infile(filename);
    if(!infile.is_open())
        throw ARBD_Exception(FILE_OPEN_ERROR, filename.c_str());
    std::string line;
    float4* param;
    int* index;
    while(std::getline(infile, line))
    {
        Utility::trim_both_sides(line);
        if(line.empty())
            continue;
        if(line == std::string("#Restraint"))
        {
            std::getline(infile, line);
            Utility::trim_both_sides(line);
            _num = Utility::string_to_number<int>(line);
            param = new float4 [_num];
            index = new int [_num];
            for(auto i = 0; i < _num; ++i)
            {
                std::vector<std::string> element;
                std::getline(infile,line);
                Utility::trim_both_sides(line); 
                Utility::split(element, line);
                index[i] = Utility::string_to_number<int>(element[0]);
                param[i] = make_float4(Utility::string_to_number<float>(element[1]),
                                       Utility::string_to_number<float>(element[2]),
                                       Utility::string_to_number<float>(element[3]),
                                       Utility::string_to_number<float>(element[4])); 
            }       
        }
    }
    infile.close();
    MemcpyHtoD(param, index);
    delete [] param;
    delete [] index;
}

RestraintHarmonic::~RestraintHarmonic()
{
    cudaFree(_param);
    cudaFree(_index);
}

void RestraintHarmonic::MemcpyHtoD(float4* param, int* index)
{
    cudaMalloc((void**)&_param, sizeof(float4)*_num);
    cudaMalloc((void**)&_index, sizeof(int)*_num);
    cudaMemcpy(_param, param, sizeof(float4)*_num, cudaMemcpyHostToDevice);
    cudaMemcpy(_index, index, sizeof(int)*_num, cudaMemcpyHostToDevice);
}

__global__ void AddForceRestraintHarmonicKernel(Vector3*, Vector3*,float4*, int*, int);
__global__ void AddEnergyRestraintHarmonicKernel(float*, Vector3*, float4*, int*, int);

void RestraintHarmonic::AddForce(Vector3* force, Vector3* position)
{
    int numBlocks = (_num+127)/128;
    numBlocks = (numBlocks > 65535) ? 65535 : numBlocks;
    AddForceRestraintHarmonicKernel<<<numBlocks, 128>>>(force, position, _param, _index, _num);
    cudaDeviceSynchronize();
}

void RestraintHarmonic::AddEnergy(float* energy, Vector3* position)
{
    int numBlocks = (_num+127)/128;
    numBlocks = (numBlocks > 65535) ? 65535 : numBlocks;
    AddEnergyRestraintHarmonicKernel<<<numBlocks, 128>>>(energy, position, _param, _index, _num);
    cudaDeviceSynchronize();
}
