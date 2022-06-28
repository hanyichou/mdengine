#include <fstream>
#include <vector>
#include "Utility.h"
#include "BendingTorsionONC.h"
#include "ARBDException.h"

BendingTorsionONC::BendingTorsionONC(const std::string& filename) : _bondList(nullptr), _numList(0), _params(nullptr)
{
    float3 params;
    int2* bondList = nullptr;
    std::ifstream infile(filename);
    if(!infile.is_open())
        throw ARBD_Exception(FILE_OPEN_ERROR, filename.c_str());

    std::string line;
    while(std::getline(infile, line))
    {
        Utility::trim_both_sides(line);
        if(line.empty())
            continue;
        if(line == std::string("#Bond Coeff"))
        {
            std::getline(infile,line);
            std::vector<std::string> elements;
            Utility::trim_both_sides(line);
            Utility::split(elements, line);
            params = make_float3(Utility::string_to_number<float>(elements[0]),
                                 Utility::string_to_number<float>(elements[1]),
                                 Utility::string_to_number<float>(elements[2]));
        }
        else if(line == std::string("#Bond"))
        {
            std::getline(infile, line);
            Utility::trim_both_sides(line);
            _numList = Utility::string_to_number<int>(line);
            bondList = new int2[_numList];
            for(auto i = 0; i < _numList; ++i)
            {
                std::vector<std::string> elements;
                std::getline(infile,line);
                Utility::trim_both_sides(line);
                Utility::split(elements, line); 
                bondList[i] = make_int2(Utility::string_to_number<int>(elements[0]),
                                        Utility::string_to_number<int>(elements[1])); 
            }
        }
    }
    infile.close();
    MemcpyHtoD(bondList, params);
    delete [] bondList;
}

BendingTorsionONC::~BendingTorsionONC()
{
    cudaFree(_bondList);
    cudaFree(_params);
}

void BendingTorsionONC::MemcpyHtoD(int2* bondList, const float3& params)
{
    cudaMalloc((void**)&_bondList, sizeof(int2)*_numList);
    cudaMalloc((void**)&_params, sizeof(float3));
    cudaMemcpy(_bondList, bondList, sizeof(int2)*_numList, cudaMemcpyHostToDevice);
    cudaMemcpy(_params, &params, sizeof(float3), cudaMemcpyHostToDevice);
}

__global__ void ComputeForceBendingTorsionONCKernel(Vector3*, Vector3*, int2*, int, float3*);
__global__ void ComputeEnergyBendingTorsionONCKernel(float*, Vector3*, int2*, int, float3*);

void BendingTorsionONC::ComputeForce(Vector3* force, Vector3* position)
{
    int numBlocks = (_numList+127)/128;
    numBlocks = (numBlocks > 65535) ? 65535 : numBlocks;
    ComputeForceBendingTorsionONCKernel<<<numBlocks, 128>>>(force, position, _bondList, _numList, _params);
    cudaDeviceSynchronize();
}

void BendingTorsionONC::ComputeEnergy(float* energy, Vector3* position)
{
    int numBlocks = (_numList+127)/128;
    numBlocks = (numBlocks > 65535) ? 65535 : numBlocks;
    ComputeEnergyBendingTorsionONCKernel<<<numBlocks, 128>>>(energy, position, _bondList, _numList, _params);
    cudaDeviceSynchronize();
}

