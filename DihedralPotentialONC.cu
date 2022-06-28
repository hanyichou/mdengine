#include <vector>
#include <fstream>
#include "Utility.h"
#include "DihedralPotentialONC.h"
#include "ARBDException.h"

DihedralPotentialONC::DihedralPotentialONC(const std::string& filename) : _numDihedral(0), _dihedralList(nullptr),
_type(nullptr), _alpha(nullptr)
{
    int4* dihedralList = nullptr;
    int* type = nullptr;
    float* alpha = nullptr; 

    std::ifstream infile(filename);
    if(!infile.is_open())
        throw ARBD_Exception(FILE_OPEN_ERROR, filename.c_str());

    std::string line;
    while(std::getline(infile, line))
    {
        Utility::trim_both_sides(line);
        if(line.empty())
            continue;
        if(line == std::string("#Dihedral Coeff"))
        {
            alpha = new float[9*13];
            for(auto i = 0; i < 9; ++i)
            { 
                line.clear();
                std::vector<std::string> elements;
                std::getline(infile, line);
                Utility::trim_both_sides(line); 
                Utility::split(elements, line);
                int mytype = Utility::string_to_number<int>(elements[0]);
                for(auto j = 1; j <= 13; ++j)
                    alpha[13*mytype+j-1] = Utility::string_to_number<float>(elements[j]);
            }
        }
        else if(line == std::string("#Dihedral")) 
        {
            line.clear();
            std::getline(infile,line);
            Utility::trim_both_sides(line);
            _numDihedral = Utility::string_to_number<int>(line);
            dihedralList = new int4[_numDihedral];
            type = new int[_numDihedral];
            for(auto i = 0; i < _numDihedral; ++i)
            {
                std::getline(infile,line);
                std::vector<std::string> elements;
                Utility::trim_both_sides(line);
                Utility::split(elements, line);
                type[i] = Utility::string_to_number<int>(elements[0]);
                dihedralList[i] = make_int4(Utility::string_to_number<int>(elements[1]),
                                            Utility::string_to_number<int>(elements[2]),
                                            Utility::string_to_number<int>(elements[3]),
                                            Utility::string_to_number<int>(elements[4]));
            }
        }
    }
    infile.close();
    MemcpyHtoD(dihedralList, type, alpha);
    delete [] dihedralList;
    delete [] type;
    delete [] alpha;
}

void DihedralPotentialONC::MemcpyHtoD(int4* dihedralList, int* type, float* alpha)
{
    cudaMalloc((void**)&_dihedralList, sizeof(float4)*_numDihedral);
    cudaMalloc((void**)&_type, sizeof(int)*_numDihedral);
    cudaMalloc((void**)&_alpha, sizeof(float)*(13*9));
    cudaMemcpy(_dihedralList, dihedralList, sizeof(float4)*_numDihedral, cudaMemcpyHostToDevice);
    cudaMemcpy(_type, type, sizeof(int)*_numDihedral, cudaMemcpyHostToDevice);
    cudaMemcpy(_alpha, alpha, sizeof(float)*13*9, cudaMemcpyHostToDevice);
}

DihedralPotentialONC::~DihedralPotentialONC()
{
    cudaFree(_dihedralList);
    cudaFree(_type);
    cudaFree(_alpha);
}

//declare gobal functions here
__global__ void ComputeForceDihedralONCKernel(Vector3*, Vector3*, int4*, int*, int, float*);
__global__ void ComputeEnergyDihedralONCKernel(float*, Vector3*, int4*, int*, int, float*);
 
void DihedralPotentialONC::ComputeForce(Vector3* force, Vector3* position)
{
    int numBlocks = (_numDihedral+63)/64;
    numBlocks = (numBlocks > 65535) ? 65535 : numBlocks;   
    ComputeForceDihedralONCKernel<<<numBlocks, 64>>>(force, position, _dihedralList, _type, _numDihedral, _alpha);
    cudaDeviceSynchronize();
}

void DihedralPotentialONC::ComputeEnergy(float* energy, Vector3* position)
{
    int numBlocks = (_numDihedral+63)/64;
    numBlocks = (numBlocks > 65535) ? 65535 : numBlocks;   
    ComputeEnergyDihedralONCKernel<<<numBlocks, 64>>>(energy, position, _dihedralList, _type, _numDihedral, _alpha);
    cudaDeviceSynchronize();
 
}
