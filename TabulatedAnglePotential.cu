#include <vector>
#include <fstream>
#include "TabulatedAnglePotential.h"
#include "Utility.h"
#include "ARBDException.h"
TabulatedAnglePotential::TabulatedAnglePotential(const std::string& filename) : _angleList(nullptr), _x(nullptr), _y(nullptr), 
_tableLength(nullptr), _numAngle(0), _numType(0)
{
    float** x = nullptr, **y = nullptr;
    int* tab_len = nullptr;
    int4* angle_list = nullptr;

    std::ifstream infile(filename);
    if(!infile.is_open()) 
        throw ARBD_Exception(FILE_OPEN_ERROR, filename.c_str());
    std::string line;
    while(std::getline(infile, line))
    {
        Utility::trim_both_sides(line);
        if(line.empty())
            continue;
        if(line == std::string("#Angle Coeff"))
        {
            line.clear();
            std::getline(infile,line);
            Utility::trim_both_sides(line);
            _numType = Utility::string_to_number<int>(line);
            x  = new float* [_numType];
            y  = new float* [_numType];
            tab_len  = new int[_numType];
            for(int i = 0; i < _numType; ++i)
            {
                x[i] = nullptr;
                y[i] = nullptr;
                tab_len[i] = 0;
            }
            for(int i = 0; i < _numType; ++i)
            {
                std::vector<std::string> element;
                line.clear();
                std::getline(infile, line);
                Utility::split(element, line);
                int my_type = Utility::string_to_number<int>(element[0]);
                int my_len  = Utility::string_to_number<int>(element[1]);
                if(x[my_type] != nullptr or y[my_type] != nullptr or tab_len[my_type] != 0)
                    throw ARBD_Exception(ILL_FORMATTED_FILE, filename.c_str()); 
                x[my_type] = new float[my_len];
                y[my_type] = new float[my_len];
                tab_len[my_type]  = my_len;
                ReadTable(x[my_type], y[my_type], my_len, element[2]);   
            }
        }
        else if(line == std::string("#Angle"))
        {
            line.clear();
            std::getline(infile,line);
            Utility::trim_both_sides(line);
            _numAngle = Utility::string_to_number<int>(line);
            angle_list = new int4[_numAngle];
            for(int i = 0; i < _numAngle; ++i)
            {
                line.clear();
                std::getline(infile,line);
                Utility::trim_both_sides(line);
                std::vector<std::string> element;
                Utility::split(element, line);
                angle_list[i] = make_int4(Utility::string_to_number<int>(element[0]),
                                          Utility::string_to_number<int>(element[1]),
                                          Utility::string_to_number<int>(element[2]),
                                          Utility::string_to_number<int>(element[3]));
            }
        }
    }
    infile.close();
    for(int i = 0; i < _numType; ++i)
    {
        if(x[i] == nullptr or y[i] == nullptr or tab_len[i] == 0)
            throw ARBD_Exception(ILL_FORMATTED_FILE, filename.c_str());
    }
    MemcpyHtoD(x,y, tab_len, angle_list);
    for(auto i = 0; i < _numType; ++i)
    {
        delete [] x[i];
        delete [] y[i];
    }
    delete [] x;
    delete [] y;
    delete [] tab_len; 
}

TabulatedAnglePotential::~TabulatedAnglePotential()
{
    float** tmp = new float* [_numType];
    cudaMemcpy(tmp, _x, sizeof(float*)*_numType, cudaMemcpyDeviceToHost);
    for(int i = 0; i < _numType; ++i)
        cudaFree(tmp[i]);
    cudaMemcpy(tmp, _y, sizeof(float*)*_numType, cudaMemcpyDeviceToHost);
    for(int i = 0; i < _numType; ++i)
        cudaFree(tmp[i]);
    cudaFree(_x);
    cudaFree(_y);
    cudaFree(_tableLength);
}

void TabulatedAnglePotential::ReadTable(float* x, float* y, const int& len, const std::string& filename)
{
    int count = 0;
    std::ifstream infile(filename);
    if(!infile.is_open())   
        ARBD_Exception(FILE_OPEN_ERROR, filename.c_str());
    float a, b;
    while(infile >> a >> b)
    {
        if(count == len)
            throw ARBD_Exception(ILL_FORMATTED_FILE, filename.c_str());
        x[count] = a/180.f*3.1415926536f;
        y[count] = b;
        count++;
    }
    if(count != len)
        throw ARBD_Exception(ILL_FORMATTED_FILE, filename.c_str());
    infile.close();
}

void TabulatedAnglePotential::MemcpyHtoD(float** x, float** y, int* tab_len, int4* angle_list)
{
    cudaMalloc((void***)&_x, sizeof(float*)*_numType);
    cudaMalloc((void***)&_y, sizeof(float*)*_numType);
    cudaMalloc((void**)&_tableLength, sizeof(int)*_numType);
    float** x_tmp = new float* [_numType];
    float** y_tmp = new float* [_numType];
    for(int i = 0 ; i < _numType; ++i)
    {
        cudaMalloc((void**)&x_tmp[i], sizeof(float)*tab_len[i]);
        cudaMalloc((void**)&y_tmp[i], sizeof(float)*tab_len[i]);
        cudaMemcpy(x_tmp[i], x[i], sizeof(float)*tab_len[i], cudaMemcpyHostToDevice);
        cudaMemcpy(y_tmp[i], y[i], sizeof(float)*tab_len[i], cudaMemcpyHostToDevice); 
    }
    cudaMemcpy(_x, x_tmp, sizeof(float*)*_numType, cudaMemcpyHostToDevice);
    cudaMemcpy(_y, y_tmp, sizeof(float*)*_numType, cudaMemcpyHostToDevice);
    cudaMemcpy(_tableLength, tab_len, sizeof(int)*_numType, cudaMemcpyHostToDevice); 

    delete [] y_tmp;
    delete [] x_tmp;
    cudaMalloc((void**)&_angleList, sizeof(int4)*_numAngle);
    cudaMemcpy(_angleList, angle_list, sizeof(int4)*_numAngle, cudaMemcpyHostToDevice);
}

__global__ void ComputeForceTabulatedAngleKernel (Vector3*, Vector3*, int4*, int, float**, float**, int*);
__global__ void ComputeEnergyTabulatedAngleKernel(float*, Vector3*, int4*, int, float**, float**, int*);
 
//declare global function here
void TabulatedAnglePotential::ComputeForce(Vector3* force, Vector3* position)
{
    int numBlocks = (_numAngle+63)/64;
    numBlocks = (numBlocks > 65535) ? 65535 : numBlocks;
    ComputeForceTabulatedAngleKernel<<<numBlocks, 64>>>(force, position, _angleList, _numAngle, _x, _y, _tableLength);
    cudaDeviceSynchronize();
}

void TabulatedAnglePotential::ComputeEnergy(float* energy, Vector3* position)
{
    int numBlocks = (_numAngle+63)/64;
    numBlocks = (numBlocks > 65535) ? 65535 : numBlocks;
    ComputeEnergyTabulatedAngleKernel<<<numBlocks,64>>>(energy, position, _angleList, _numAngle, _x, _y, _tableLength);
    cudaDeviceSynchronize();
}

