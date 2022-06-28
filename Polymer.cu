#include <cmath>
#include <iostream>
#include <fstream>
#include <cstring>
#include "Polymer.h"
#include "CudaUtil.cuh"
#include "Utility.h"
#include "ExternalPotentialGrid.h"
#include "ConfinementGrid.h"
#include "TabulatedAnglePotential.h"
#include "DihedralPotentialONC.h"
#include "BendingTorsionONC.h"
#include "RestraintHarmonic.h"
#include <thrust/device_vector.h>
#include "ARBDException.h"
#define cubic_root_2 1.25992104989f 
#define Pi 3.14159265359f 
#define Restrict __restrict__
struct Bond
{
    int num;
    int2* bond;
};

struct VerletCell
{
    int2* cellList;
    int2* indexRange;
    int*  prefixSum;
    int*  numPairs; 
    int2* pairs;
    float a0;
    int4 L;
    int decompFreq;
};

/*
 * param.y : sigma
 * param.z : lambda
 * param.w : charge
 */

struct Particle
{
    float mass;
    float3 param;
};

struct compare : public thrust::greater<int2>
{
    __host__ __device__
    bool operator() (const int2& a, const int2& b)
    {
        return ((a.x < b.x) || (a.x==b.x && a.y < b.y));
    }
};


#define __MAXTYPES 40
#define __MAXBONDTYPES 16
//#define __CUTOFF 0.6f
#define __MAXPAIRS ((1<<29))
__constant__ SystemBox systemBox_const[1];
__constant__ float particleMass_const[__MAXTYPES];
__constant__ float3 particleAttribute_const[__MAXTYPES];
__constant__ float particleNBStrength_const[__MAXTYPES*__MAXTYPES];
__constant__ float2 feneData_const[__MAXBONDTYPES];
__constant__ float2 harmonicData_const[__MAXBONDTYPES];
__constant__ float parameterZ_const[1];
__constant__ float truncateDistance_const[1];

__constant__ float verletCellCutoff_const[1];
__constant__ int4  verletCellDim_const[1];

__constant__ float Kappa_const[1];
__constant__ float Dielectric_const[1];

__device__ Vector3 wrap(const Vector3& pos)
{
    SystemBox sys = systemBox_const[0];
    Vector3 r;
    r.x = pos.x - sys.origin.x;
    r.y = pos.y - sys.origin.y;
    r.z = pos.z - sys.origin.z;
    r.x -= floorf(__fdividef(r.x, sys.L.x))*sys.L.x;
    r.y -= floorf(__fdividef(r.y, sys.L.y))*sys.L.y;
    r.z -= floorf(__fdividef(r.z, sys.L.z))*sys.L.z;
    r.x += sys.origin.x;
    r.y += sys.origin.y;
    r.z += sys.origin.z;
    return r;
}

__device__ Vector3 wrapVecDiff(const Vector3& v1, const Vector3& v2)
{
    Vector3 r     = v1-v2;
    float3 l = systemBox_const[0].L;
    r.x -= floorf(__fdividef(r.x, l.x)+0.5f)*l.x;
    r.y -= floorf(__fdividef(r.y, l.y)+0.5f)*l.y;
    r.z -= floorf(__fdividef(r.z, l.z)+0.5f)*l.z;
    return r;
}

__device__ float wrapDiff(const Vector3& v1, const Vector3& v2)
{
    Vector3 r     = v1-v2;
    float3 l = systemBox_const[0].L;
    r.x -= floorf(__fdividef(r.x, l.x)+0.5f)*l.x;
    r.y -= floorf(__fdividef(r.y, l.y)+0.5f)*l.y;
    r.z -= floorf(__fdividef(r.z, l.z)+0.5f)*l.z;
    return r.length();
}

__device__ float wrapDiff2(const Vector3& v1, const Vector3& v2)
{
    Vector3 r     = v1-v2;
    float3 l = systemBox_const[0].L;
    r.x -= floorf(__fdividef(r.x, l.x)+0.5f)*l.x;
    r.y -= floorf(__fdividef(r.y, l.y)+0.5f)*l.y;
    r.z -= floorf(__fdividef(r.z, l.z)+0.5f)*l.z;
    return r.length2();
}

void ComputeOrientation(Vector3* inertia, Matrix3* orientation, Vector3* pos, Vector3* com, float* m, int2* groupID, float* scaleFactor, int numGrouped, int numRigid);

__global__ void WrapPositionKernel(Vector3* position, int num)
{
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if(tid < num)
    {
        Vector3 tmp = position[tid];
        position[tid] = wrap(tmp);
    }
}

Polymer::Polymer(const std::string& filename) : _systemBox(SetUpSystemBox(filename)), _kT(0.58622522f), _modelType(0), _num(0), _position(nullptr), _momentum(nullptr), _force(nullptr), _repulsiveForce(nullptr), 
_type(nullptr), _numTypes(0), _particle(nullptr), _numGrouped(0), _particleID(nullptr), _groupID(nullptr), _groupRange(nullptr), _noGroupedID(nullptr),  _rigidBodyID(nullptr), _numRigid(0), 
_positionCOM(nullptr), _momentumCOM(nullptr), _forceCOM(nullptr), _positionRelative(nullptr), _orientation(nullptr), 
_angularMomentum(nullptr), _torque(nullptr), _inertia(nullptr),  _scaleFactor(nullptr), _energy(nullptr), _verletCell(nullptr), _state(nullptr), _coordinate_writer(nullptr), 
_momentum_writer(nullptr), _force_writer(nullptr), _buffer(nullptr), _numPairs(0), _exclusionMap(nullptr), _virial(nullptr)
{
    float dist_tmp = 0.f;
    cudaMemcpyToSymbol(truncateDistance_const, &dist_tmp, sizeof(float));
    std::ifstream infile(filename);
    std::string line;
    bool is_coords_read = false;
    bool is_topology_read = false;
    bool is_momentum_read = false;
    bool is_random_number_initialized = false;

    if(infile.is_open())
    {
        while(std::getline(infile, line))
        {
            Utility::remove_comment(line, std::string("#"));
            Utility::trim_both_sides(line);
            if(line.empty())
                continue;
            std::vector<std::string> elements;
            Utility::split(elements, line, std::string(" "));
            if(elements[0] == std::string("Coordinates"))
            {
                ReadCoordinates(elements[1]);
                is_coords_read = true;
                 std::cout << "Reading Coordinates" << std::endl;
            }
            else if(elements[0] == std::string("Momentum"))
            {
                if(is_coords_read == false)
                    throw ARBD_Exception(SIMULATION_TERMINATED, "Read coordinates first"); 
                ReadMomentum(elements[1]);
                is_momentum_read = true;
            }
            else if(elements[0] == std::string("Topology"))
            {
                if(is_coords_read == false)
                    throw ARBD_Exception(SIMULATION_TERMINATED, "Read coordinates first");
                ReadTopology(elements[1]);
                is_topology_read = true;
                std::cout << "Reading Topology" << std::endl;
            }
            else if(elements[0] == std::string("ScreenLength"))
            {
                float len = Utility::string_to_number<float>(elements[1]);
                cudaMemcpyToSymbol(Kappa_const, &len, sizeof(float));
            }
            else if(elements[0] == std::string("Dielectric"))
            {
                float eps = Utility::string_to_number<float>(elements[1]);
                cudaMemcpyToSymbol(Dielectric_const, &eps, sizeof(float));
            }
            else if(elements[0] == std::string("ParameterZ"))
            {
                float z = Utility::string_to_number<float>(elements[1]);
                cudaMemcpyToSymbol(parameterZ_const, &z, sizeof(float));
            }
            else if(elements[0] == std::string("Seed"))
            {
                unsigned long long seed = Utility::string_to_number<unsigned long long>(elements[1]);
                InitializeRandomNumber(seed);
                is_random_number_initialized = true;
            }
            else if(elements[0] == std::string("TruncateDistance"))
            {
                float dist = Utility::string_to_number<float>(elements[1]);
                cudaMemcpyToSymbol(truncateDistance_const, &dist, sizeof(float));
            }
            else if(elements[0] == std::string("Temperature"))
                _kT = Utility::string_to_number<float>(elements[1]);
            line.clear();
        }
        infile.close();
    }
    else
        throw ARBD_Exception(FILE_OPEN_ERROR, filename.c_str());
    if(is_coords_read == false or is_topology_read == false)
        throw ARBD_Exception(SIMULATION_TERMINATED, "Coordinates or Topology is not initialized");
    if(is_momentum_read == false)
    {
        if(is_random_number_initialized == false)
        {
            InitializeRandomNumber(1989ull);
            is_random_number_initialized = true;
        }
        InitializeMomentum();
        is_momentum_read = true;
    }
    else
    {
        if(_numRigid>0)
            ComputeRigidPartMomentum();
    }
    cudaMalloc((void**)&_energy, 65536*sizeof(float));
    cudaMemset(_energy, 0,65536*sizeof(float));
    cudaMalloc((void**)&_virial, 65536*6*sizeof(float));
    cudaMemset(_virial, 0,65536*6*sizeof(float));

    _buffer = new Vector3 [_num];
    //wrap postion here
    WrapPositionKernel<<<(_num+255)/256, 256>>>(_position, _num);
    cudaDeviceSynchronize();
}

SystemBox Polymer::SetUpSystemBox(const std::string& filename)
{
    std::ifstream infile(filename);
    std::string line;
    SystemBox tmp;
    if(infile.is_open())
    {
        while(std::getline(infile, line))
        {
            Utility::remove_comment(line, std::string("#"));
            Utility::trim_both_sides(line);
            if(line.empty())
                continue;
            std::vector<std::string> elements;
            Utility::split(elements, line, std::string(" "));
            if(elements[0] == std::string("Origin"))
            {
                float x = Utility::string_to_number<float>(elements[1]);
                float y = Utility::string_to_number<float>(elements[2]);
                float z = Utility::string_to_number<float>(elements[3]);
                tmp.origin = make_float3(x,y,z);
            }
            else if(elements[0] == std::string("Size"))
            {
                float x = Utility::string_to_number<float>(elements[1]);
                float y = Utility::string_to_number<float>(elements[2]);
                float z = Utility::string_to_number<float>(elements[3]);
                tmp.L = make_float3(x,y,z);
            }
            line.clear();
        }
        infile.close();
    }
    else
        throw ARBD_Exception(FILE_OPEN_ERROR, filename.c_str());
    cudaMemcpyToSymbol(systemBox_const, &tmp, sizeof(SystemBox));
    return tmp;
}

Polymer::~Polymer()
{
    cudaFree(_virial);
    cudaFree(_exclusionMap);
    cudaFree(_state);
    cudaFree(_verletCell->cellList);
    cudaFree(_verletCell->indexRange);
    cudaFree(_verletCell->prefixSum);
    cudaFree(_verletCell->numPairs); 
    cudaFree(_verletCell->pairs);
    delete _verletCell;
    for(int i = 0; i < _harmonic.size(); ++i)
    {
        cudaFree(_harmonic[i]->bond);
        delete _harmonic[i];
    }
    for(int i = 0; i < _fene.size(); ++i)
    {
        cudaFree(_fene[i]->bond);
        delete _fene[i];
    }
    cudaFree(_scaleFactor);
    cudaFree(_inertia);
    cudaFree(_torque);
    cudaFree(_angularMomentum);
    cudaFree(_orientation);
    cudaFree(_positionRelative);
    cudaFree(_forceCOM);
    cudaFree(_momentumCOM);
    cudaFree(_positionCOM);
    cudaFree(_rigidBodyID);
    cudaFree(_noGroupedID);
    cudaFree(_groupRange);
    cudaFree(_groupID);
    cudaFree(_particleID);
    cudaFree(_particle);
    cudaFree(_type);
    cudaFree(_repulsiveForce);
    cudaFree(_force);
    cudaFree(_momentum);
    cudaFree(_position);
    delete _coordinate_writer;
    delete _momentum_writer;
    delete _force_writer;
    delete [] _buffer;
    for(auto it = _externalPotentialBase.begin(); it != _externalPotentialBase.end(); ++it)
        delete *it;
    for(auto it = _angleTorsionBase.begin(); it != _angleTorsionBase.end(); ++it)
        delete *it;
    for(auto it = _restraintBase.begin(); it != _restraintBase.end(); ++it)
        delete *it;
}

void Polymer::ReadCoordinates(const std::string& filename)
{
    std::ifstream infile(filename);
    std::vector<int> type;
    std::vector<Vector3> position;
    if(infile.is_open())
    {
        int x;
        Vector3 v;
        while(infile >> x >> v)
        {
            type.push_back(x);
            position.push_back(v);
        }
        _num = position.size();
        cudaMalloc((void**)&_position, sizeof(Vector3)*_num);
        cudaMemcpy(_position, position.data(), sizeof(Vector3)*_num, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&_type, sizeof(int)*_num);
        cudaMemcpy(_type, type.data(), sizeof(int)*_num, cudaMemcpyHostToDevice); 
        if(_momentum == nullptr)
        {
            cudaMalloc((void**)&_momentum, sizeof(Vector3)*_num);
            cudaMemset(_momentum, 0, sizeof(Vector3)*_num);
        }
        if(_force == nullptr)
        {
            cudaMalloc((void**)&_force, sizeof(Vector3)*_num);  
            cudaMemset(_force, 0, sizeof(Vector3)*_num); 
        }
        if(_repulsiveForce == nullptr)
        {
            cudaMalloc((void**)&_repulsiveForce, sizeof(Vector3)*_num);
            cudaMemset(_repulsiveForce, 0, sizeof(Vector3)*_num);    
        }
        infile.close();
    }
    else
        throw ARBD_Exception(FILE_OPEN_ERROR, filename.c_str());
}

__global__ void ComputeAngularMomentumKernel(Vector3* Restrict angularMomentum, Vector3* Restrict momentum, Vector3* Restrict positionRelative, 
Matrix3* Restrict orientation, int2* groupRange, int* particleID)
{
    __shared__ float4 mom_com[64];
    mom_com[threadIdx.x] = make_float4(0.f, 0.f, 0.f, 0.f);
    int2 range = groupRange[blockIdx.x];
    int tid = threadIdx.x;
    for(int i = tid+range.x; i < range.y; i += 64)
    {
        int idx = particleID[i];
        Vector3 p = momentum[idx];
        p = orientation[blockIdx.x].transpose() * p;
        Vector3 pos_r = positionRelative[i];
        p = pos_r.cross(p);
        mom_com[threadIdx.x].x += p.x;
        mom_com[threadIdx.x].y += p.y;
        mom_com[threadIdx.x].z += p.z;
    }
    __syncthreads();
    if(tid<32)
    {
        mom_com[tid].x += mom_com[tid+32].x;
        mom_com[tid].y += mom_com[tid+32].y;
        mom_com[tid].z += mom_com[tid+32].z;
    }
    __syncthreads();
    if(tid<16)
    {
        mom_com[tid].x += mom_com[tid+16].x;
        mom_com[tid].y += mom_com[tid+16].y;
        mom_com[tid].z += mom_com[tid+16].z;
    }
    __syncthreads();
    if(tid<8)
    {
        mom_com[tid].x += mom_com[tid+8].x;
        mom_com[tid].y += mom_com[tid+8].y;
        mom_com[tid].z += mom_com[tid+8].z;
    }
    __syncthreads();
    if(tid<4)
    {
        mom_com[tid].x += mom_com[tid+4].x;
        mom_com[tid].y += mom_com[tid+4].y;
        mom_com[tid].z += mom_com[tid+4].z;
    }
    __syncthreads();
    if(tid<2)
    {
        mom_com[tid].x += mom_com[tid+2].x;
        mom_com[tid].y += mom_com[tid+2].y;
        mom_com[tid].z += mom_com[tid+2].z;
    }
    __syncthreads();
    if(tid == 0)
    {
        mom_com[0].x += mom_com[1].x;
        mom_com[0].y += mom_com[1].y;
        mom_com[0].z += mom_com[1].z;

        Vector3 tmp = Vector3(mom_com[0].x, mom_com[0].y, mom_com[0].z);
        angularMomentum[blockIdx.x] = tmp;
    }

}

__global__ void ComputeMomentumCOMKernel(Vector3* Restrict momentumCOM, Vector3* Restrict momentum, int2* groupRange, int* particleID)
{
    __shared__ float4 mom_com[64];
    mom_com[threadIdx.x] = make_float4(0.f, 0.f, 0.f, 0.f);
    int2 range = groupRange[blockIdx.x];
    int tid = threadIdx.x; 
    for(int i = threadIdx.x+range.x; i < range.y; i += 64)
    {
        int idx = particleID[i];
        Vector3 p = momentum[idx];
        mom_com[threadIdx.x].x += p.x;
        mom_com[threadIdx.x].y += p.y;
        mom_com[threadIdx.x].z += p.z;
    }
    __syncthreads();
    if(tid<32)
    {
        mom_com[tid].x += mom_com[tid+32].x;
        mom_com[tid].y += mom_com[tid+32].y;
        mom_com[tid].z += mom_com[tid+32].z;
    }
    __syncthreads();
    if(tid<16)
    {
        mom_com[tid].x += mom_com[tid+16].x;
        mom_com[tid].y += mom_com[tid+16].y;
        mom_com[tid].z += mom_com[tid+16].z;
    }
    __syncthreads();
    if(tid<8)
    {
        mom_com[tid].x += mom_com[tid+8].x;
        mom_com[tid].y += mom_com[tid+8].y;
        mom_com[tid].z += mom_com[tid+8].z;
    }
    __syncthreads();
    if(tid<4)
    {
        mom_com[tid].x += mom_com[tid+4].x;
        mom_com[tid].y += mom_com[tid+4].y;
        mom_com[tid].z += mom_com[tid+4].z;
    }
    __syncthreads();
    if(tid<2)
    {
        mom_com[tid].x += mom_com[tid+2].x;
        mom_com[tid].y += mom_com[tid+2].y;
        mom_com[tid].z += mom_com[tid+2].z;
    }
    __syncthreads();
    if(tid == 0)
    {
        mom_com[0].x += mom_com[1].x;
        mom_com[0].y += mom_com[1].y;
        mom_com[0].z += mom_com[1].z;

        Vector3 tmp = Vector3(mom_com[0].x, mom_com[0].y, mom_com[0].z);
        momentumCOM[blockIdx.x] = tmp;
    }
}

void Polymer::ComputeRigidPartMomentum()
{
    ComputeMomentumCOMKernel<<<_numRigid, 64>>>(_momentumCOM, _momentum, _groupRange, _particleID);
    ComputeAngularMomentumKernel<<<_numRigid, 64>>>(_angularMomentum, _momentum, _positionRelative, _orientation, _groupRange, _particleID);
    cudaDeviceSynchronize();
}

void Polymer::ReadMomentum(const std::string& filename)
{
    std::ifstream infile(filename);
    std::vector<Vector3> momentum;
    if(infile.is_open())
    {
        Vector3 v;
        while(infile >> v)
        {
            momentum.push_back(v);
        }
        if(_num != momentum.size())
            throw ARBD_Exception(ILL_FORMATTED_FILE, filename.c_str());
        if(_momentum == nullptr)
            cudaMalloc((void**)&_momentum, sizeof(Vector3)*_num);
        cudaMemcpy(_momentum, momentum.data(), sizeof(Vector3)*_num, cudaMemcpyHostToDevice);
        infile.close();
    }
    else
        throw ARBD_Exception(FILE_OPEN_ERROR, filename.c_str());
}

void Polymer::ReadTopology(const std::string& filename)
{
    std::ifstream infile(filename);
    std::vector<float2> fene_data;
    std::vector<float2> harmonic_data;
    if(infile.is_open())
    {
        std::string line;
        cudaMalloc((void**)&_exclusionMap, sizeof(uint8_t)*_num);
        cudaMemset(_exclusionMap, 0, sizeof(uint8_t)*_num);
        while(std::getline(infile, line))
        {
            Utility::trim_both_sides(line);
            if(line.empty())
                continue;
            if(line == std::string("#Parameter"))
            {
                line.clear();
                std::vector<std::string> elements;
                std::getline(infile,line);
                Utility::trim_both_sides(line);
                Utility::split(elements, line, std::string(" ")); 
                _numTypes = Utility::string_to_number<int>(elements[0]);

                std::vector<Particle> particle;
                particle.resize(_numTypes);
                for(int i = 0; i < _numTypes; ++i)
                {
                    line.clear();
                    elements.clear();
                    std::getline(infile, line);
                    Utility::trim_both_sides(line); 
                    Utility::split(elements, line, std::string(" "));
                    Particle tmp;
                    int idx     = Utility::string_to_number<int>(elements[0]);
                    tmp.mass    = Utility::string_to_number<float>(elements[1]);
                    tmp.param.x = Utility::string_to_number<float>(elements[2]);
                    tmp.param.y = Utility::string_to_number<float>(elements[3]);
                    tmp.param.z = Utility::string_to_number<float>(elements[4]);
                    particle[idx] = tmp;
                }
            
                cudaMalloc((void**)&_particle, sizeof(Particle)*_numTypes);
                cudaMemcpy(_particle, particle.data(), sizeof(Particle)*_numTypes, cudaMemcpyHostToDevice);
                for(int i = 0; i < particle.size(); ++i)
                {
                    cudaMemcpyToSymbol(particleMass_const, &(particle[i].mass), sizeof(float), sizeof(float)*i);
                    cudaMemcpyToSymbol(particleAttribute_const, &(particle[i].param), sizeof(float3), sizeof(float3)*i);
                }
            }
            else if(line == std::string("#Bond"))
            {
                line.clear();
                std::vector<std::string> elements;
                std::getline(infile, line);
                Utility::trim_both_sides(line);
                Utility::split(elements, line, std::string(" ")); 
                std::string type = elements[0];
                float k0  = Utility::string_to_number<float>(elements[1]);
                float r0  = Utility::string_to_number<float>(elements[2]);
                int num   = Utility::string_to_number<int> (elements[3]);
                std::vector<int2> bond;
                bond.reserve(num); 
                for(int i =0; i < num; ++i)
                {
                    line.clear();
                    elements.clear();
                    std::getline(infile,line);
                    Utility::trim_both_sides(line);
                    Utility::split(elements, line, std::string(" "));
                    int a = Utility::string_to_number<int>(elements[0]);
                    int b = Utility::string_to_number<int>(elements[1]);
                    bond.push_back(make_int2(a,b));
                }
                Bond* tmp = new Bond;
                tmp->num = num;
                cudaMalloc((void**)(&(tmp->bond)), sizeof(int2)*num);
                cudaMemcpy(tmp->bond, bond.data(), sizeof(int2)*num, cudaMemcpyHostToDevice);
           
                if(type == std::string("FENE"))
                {
                    float2 data = make_float2(k0,r0);
                    cudaMemcpyToSymbol(feneData_const, (void*)&data, sizeof(float2), sizeof(float2)*_fene.size());
                    _fene.push_back(tmp);
                }
                else if(type == std::string("Harmonic"))
                {
                    float2 data = make_float2(k0,r0);
                    cudaMemcpyToSymbol(harmonicData_const, (void*)&data, sizeof(float2), sizeof(float2)*_harmonic.size());
                    _harmonic.push_back(tmp);
                }
            }
            else if(line == std::string("#NonBonded")) 
            {
                line.clear();
                std::getline(infile, line);
                Utility::trim_both_sides(line);
                std::vector<std::string> elements;
                Utility::split(elements, line, std::string(" "));
                float cutoff = Utility::string_to_number<float>(elements[0]);
                _verletCell = new VerletCell;
                VerletCell* _cell = _verletCell;

                _cell->cellList = nullptr;
                _cell->indexRange = nullptr;
                _cell->prefixSum = nullptr;
                _cell->numPairs = nullptr;
                _cell->pairs = nullptr;

                _cell->a0  = cutoff;
                _cell->L.x = std::ceil(_systemBox.L.x / _cell->a0);
                _cell->L.y = std::ceil(_systemBox.L.y / _cell->a0);
                _cell->L.z = std::ceil(_systemBox.L.z / _cell->a0);
                _cell->L.w = _cell->L.x * _cell->L.y * _cell->L.z;
                _cell->decompFreq = Utility::string_to_number<int>(elements[1]);

                cudaMemcpyToSymbol(verletCellCutoff_const, &(_cell->a0), sizeof(float));
                cudaMemcpyToSymbol(verletCellDim_const,    &(_cell->L),  sizeof(int4));

                cudaMalloc((void**)&(_cell->cellList),  sizeof(int2)* _num);
                cudaMalloc((void**)&(_cell->indexRange),sizeof(int2)*(_cell->L.w));
                cudaMalloc((void**)&(_cell->prefixSum), sizeof(int) *(_cell->L.w+1));
                cudaMalloc((void**)&(_cell->numPairs), sizeof(int));
                cudaMalloc((void**)&(_cell->pairs),    sizeof(int2)*__MAXPAIRS);

                cudaMemset(_cell->cellList,   0, sizeof(int2)*_num);
                cudaMemset(_cell->indexRange, 0, sizeof(int2)*(_cell->L.w));
                cudaMemset(_cell->prefixSum,  0, sizeof(int)*(_cell->L.w+1));
                cudaMemset(_cell->numPairs, 0, sizeof(int));
                cudaMemset(_cell->pairs, 0, sizeof(int2)*__MAXPAIRS);

                int numTypes = Utility::string_to_number<int>(elements[2]);
                _modelType = Utility::string_to_number<int>(elements[3]);
                int total = numTypes*(numTypes-1)/2+numTypes;
                std::vector<float> eps;
                eps.resize(numTypes*numTypes); 
                for(int i = 0; i < total; ++i)
                {
                    line.clear();
                    std::getline(infile, line);
                    Utility::trim_both_sides(line);
                    elements.clear();
                    Utility::split(elements, line, std::string(" "));
                    int p1, p2;
                    p1 = Utility::string_to_number<int>(elements[0]);
                    p2 = Utility::string_to_number<int>(elements[1]);
                    eps[p1+numTypes*p2] = Utility::string_to_number<float>(elements[2]);
                    eps[p2+numTypes*p1] = eps[p1+numTypes*p2];
                }
                cudaMemcpyToSymbol(particleNBStrength_const, eps.data(), sizeof(float)*eps.size());
            }
            else if(line == std::string("#Group"))
            {
                line.clear();
                std::getline(infile, line);
                Utility::trim_both_sides(line);
                ReadGroup(line);
            }
            else if(line == std::string("#Exclusion"))
            {
                line.clear();
                std::getline(infile, line);
                Utility::trim_both_sides(line);
                ReadExclusion(line);
            }
            else if(line == std::string("#ExternalPotential"))
            {
                line.clear();
                std::getline(infile, line);
                Utility::trim_both_sides(line);

                if(line == std::string("ExternalGridByType"))
                {
                    if(_numTypes == 0)
                         std::cerr << "Must initialize type first" << std::endl;
                    line.clear();
                    std::getline(infile, line);
                    Utility::trim_both_sides(line);

                    ExternalPotentialBase* tmp = new ExternalPotentialGrid(line, _numTypes);
                    _externalPotentialBase.push_back(tmp);
                }
                else if(line == std::string("ConfinementGrid"))
                {
                    line.clear();
                    std::getline(infile, line);
                    Utility::trim_both_sides(line);
                    ExternalPotentialBase* tmp = new ConfinementGrid(line);
                    _externalPotentialBase.push_back(tmp);
                }
            }
            else if(line == std::string("#AnglePotential"))
            {
                line.clear();
                std::getline(infile,line);
                Utility::trim_both_sides(line);
                if(line == std::string("TabulatedAnglePotential"))
                {
                    line.clear();
                    std::getline(infile,line);
                    Utility::trim_both_sides(line);
                    AngleTorsionBase* tmp = new TabulatedAnglePotential(line);
                    _angleTorsionBase.push_back(tmp);
                }
            }
            else if(line == std::string("#DihedralPotential"))
            {
                line.clear();
                std::getline(infile,line);
                Utility::trim_both_sides(line);
                if(line == std::string("DihedralPotentialONC"))
                {
                    line.clear();
                    std::getline(infile,line);
                    Utility::trim_both_sides(line);
                    AngleTorsionBase* tmp = new DihedralPotentialONC(line);
                    _angleTorsionBase.push_back(tmp);
                }
            }
            else if(line == std::string("#BendingTorsionCoupling"))
            {
                line.clear();
                std::getline(infile, line);
                Utility::trim_both_sides(line);
                if(line == std::string("BendingTorsionONC"))
                {
                    line.clear();
                    std::getline(infile,line);
                    Utility::trim_both_sides(line);
                    AngleTorsionBase* tmp = new BendingTorsionONC(line);
                    _angleTorsionBase.push_back(tmp);
                }
            }
            else if(line == std::string("#Restraint"))
            {
                line.clear();
                std::getline(infile,line);
                Utility::trim_both_sides(line);
                if(line == std::string("Harmonic"))
                {
                    line.clear();
                    std::getline(infile, line);
                    Utility::trim_both_sides(line);
                    RestraintBase* tmp = new RestraintHarmonic(line);
                    _restraintBase.push_back(tmp);   
                }
            }
            line.clear();
        }
        infile.close();
        if(_verletCell == nullptr or _particle == nullptr)
            throw ARBD_Exception(SIMULATION_TERMINATED, "Must read particle parameters and verlet cell info");
    }
    else
        throw ARBD_Exception(FILE_OPEN_ERROR, filename.c_str());
}

void Polymer::ReadGroup(const std::string& filename)
{
    std::vector<int2> groupID;
    std::vector<int>  particleID;
    std::vector<int2> groupRange;
    std::vector<float> scaleFactor;

    std::ifstream infile(filename);
    if(infile.is_open())
    {
        std::string line;
        int rbID = 0;
        int numGrouped = 0;
        while(std::getline(infile, line))
        {
            Utility::trim_both_sides(line);

            if(line.empty())
                continue;

            if(line[0] == '#')
            {
                line.clear();
                std::getline(infile, line);
                Utility::trim_both_sides(line); 

                std::vector< std::string > tokens;
                Utility::split(tokens, line, std::string(" "));
                
                int num   = Utility::string_to_number<int>(tokens[0]);
                float fac = Utility::string_to_number<float>(tokens[1]);
                groupRange.push_back(make_int2(numGrouped, num+numGrouped));
                numGrouped += num;
                scaleFactor.push_back(fac);
 
                num = (num+7)/8;
                for(int i = 0; i < num; ++i)
                {
                    line.clear();
                    std::vector< std::string > elements;
                    std::getline(infile, line);
                    Utility::trim_both_sides(line);
                    Utility::split(elements, line, std::string(" "));
                    for(int j = 0; j < elements.size(); ++j)
                    {
                        int idx = Utility::string_to_number<int>(elements[j]);
                        groupID.push_back(make_int2(rbID, idx));
                        particleID.push_back(idx);
                    }
                }
                rbID++;
            }
            line.clear();
        }
        _numRigid   = rbID;
        _numGrouped = numGrouped;
        std::vector<int> rigidBodyID;
        rigidBodyID.resize(_num);
        std::fill(rigidBodyID.begin(), rigidBodyID.end(), -1);
        for(int i = 0; i < _numGrouped; ++i)
        {
            int2 a = groupID[i];
            rigidBodyID[a.y] = a.x;
        } 
        if(_num > _numGrouped)
        {
            std::vector<int> noGroupedID;
            noGroupedID.resize(_num-_numGrouped);
            int count = 0;
            for(int i = 0; i < _num; ++i)
            {
                if(rigidBodyID[i] == -1)
                    noGroupedID[count++] = i;
            }
            cudaMalloc((void**)&_noGroupedID, sizeof(int)*(_num-_numGrouped));
            cudaMemcpy(_noGroupedID, noGroupedID.data(), sizeof(int)*(_num-_numGrouped), cudaMemcpyHostToDevice);
        }
        cudaMalloc((void**)&_particleID, sizeof(int)*_numGrouped);
        cudaMalloc((void**)&_groupID,   sizeof(int2)*_numGrouped);
        cudaMalloc((void**)&_groupRange,sizeof(int2)*_numRigid);
        cudaMalloc((void**)&_rigidBodyID, sizeof(int)*_num);
        cudaMalloc((void**)&_scaleFactor, sizeof(float)*_numRigid);

        cudaMemcpy(_particleID, particleID.data(), sizeof(int)*_numGrouped, cudaMemcpyHostToDevice); 
        cudaMemcpy(_groupID, groupID.data(),      sizeof(int2)*_numGrouped, cudaMemcpyHostToDevice);
        cudaMemcpy(_groupRange, groupRange.data(),sizeof(int2)*_numRigid, cudaMemcpyHostToDevice);
        cudaMemcpy(_rigidBodyID, rigidBodyID.data(), sizeof(int)*_num, cudaMemcpyHostToDevice);
        cudaMemcpy(_scaleFactor, scaleFactor.data(), sizeof(float)*_numRigid, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&_positionCOM, sizeof(Vector3)*_numRigid);
        cudaMalloc((void**)&_momentumCOM, sizeof(Vector3)*_numRigid);
        cudaMalloc((void**)&_forceCOM, sizeof(Vector3)*_numRigid);
        cudaMalloc((void**)&_positionRelative, sizeof(Vector3)*_numGrouped);
        cudaMalloc((void**)&_orientation, sizeof(Matrix3)*_numRigid);
        cudaMalloc((void**)&_angularMomentum, sizeof(Vector3)*_numRigid);
        cudaMalloc((void**)&_torque, sizeof(Vector3)*_numRigid);
        cudaMalloc((void**)&_inertia, sizeof(float4)*_numRigid);

        cudaMemset(_positionCOM, 0, sizeof(Vector3)*_numRigid);        
        cudaMemset(_momentumCOM, 0, sizeof(Vector3)*_numRigid);
        cudaMemset(_forceCOM, 0, sizeof(Vector3)*_numRigid);
        cudaMemset(_positionRelative, 0,sizeof(Vector3)*_numGrouped);
        cudaMemset(_orientation, 0, sizeof(Matrix3)*_numRigid);
        cudaMemset(_angularMomentum, 0, sizeof(Vector3)*_numRigid);
        cudaMemset(_torque, 0, sizeof(Vector3)*_numRigid);
        cudaMemset(_inertia, 0, sizeof(float4)*_numRigid);
        InitializeRigidPart();
    }
    else
        throw ARBD_Exception(FILE_OPEN_ERROR, filename.c_str());
}

void Polymer::ReadExclusion(const std::string& filename)
{
    std::ifstream infile(filename);
    uint8_t* exclusionMap;

    if(infile.is_open())
    {
        exclusionMap = new uint8_t[_num];
        memset(exclusionMap, 0, sizeof(uint8_t)*_num);
        std::string line;
        while(std::getline(infile, line))
        {
            Utility::trim_both_sides(line);
            if(line.empty())
                continue;
            int num = Utility::string_to_number<int>(line);
            for(int i = 0; i < num; ++i)
            {
                line.clear();
                std::getline(infile, line);
                Utility::trim_both_sides(line);
                std::vector<std::string> elements;
                Utility::split(elements, line, std::string(" "));
                int a = Utility::string_to_number<int>(elements[0]);
                int b = Utility::string_to_number<int>(elements[1]);
                if(a == b)
                    throw ARBD_Exception(ILL_FORMATTED_FILE, filename.c_str());
                if(a > b)
                {
                    int c = a;
                    a = b;
                    b = c;
                } 
                exclusionMap[a] |= (1<<(b-a-1));
            }   
        }
        infile.close();
        cudaMemcpy(_exclusionMap, exclusionMap, sizeof(uint8_t)*_num, cudaMemcpyHostToDevice);
        delete [] exclusionMap;
    }
    else
        throw ARBD_Exception(FILE_OPEN_ERROR, filename.c_str());
}

__global__ void InitializeRandomNumberKernel(unsigned long long seed, curandStatePhilox4_32_10_t *state, int num) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < num)
        curand_init(seed, idx, 0, &state[idx]);
}

void Polymer::InitializeRandomNumber(unsigned long long seed)
{
    if(_state == nullptr)
        cudaMalloc((void**)&_state, sizeof(curandStatePhilox4_32_10_t)*_num);
    InitializeRandomNumberKernel<<<(_num+127)/128, 128>>>(seed, _state, _num);
    cudaDeviceSynchronize();
}

__global__ void InitializeRelativePositionKernel(Vector3* positionRelative, Vector3* position, Vector3* positionCOM, 
Matrix3* orientation, int2* groupID, int numGrouped)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < numGrouped)
    {
        int2 i = groupID[idx];
        Matrix3 R = orientation[i.x];
        positionRelative[idx] = R.transpose()*(position[i.y] - positionCOM[i.x]);
    }
}

__global__ void CopyMassDtoHKernel(float* mass, int* type, int num)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < num)
    {
        int t = type[idx];
        float m = particleMass_const[t];
        mass[idx] = m;
    }
}

void Polymer::CopyMassDtoH(float* m)
{
    float* m_d;
    cudaMalloc((void**)&m_d, sizeof(float)*_num);
    int numBlocks = (_num+127) / 128;
    CopyMassDtoHKernel<<<numBlocks, 128>>>(m_d, _type, _num);
    cudaMemcpy(m, m_d, sizeof(float)*_num, cudaMemcpyDeviceToHost);
    cudaFree(m_d);
}

__global__ void CopyInertiaHtoDKernel(float4* inertia, Vector3* src, int numRigid)
{
    int idx = threadIdx.x + 64 * blockIdx.x;
    if(idx < numRigid)
    {
        Vector3 tmp1 = src[idx];
        inertia[idx].x = tmp1.x;
        inertia[idx].y = tmp1.y;
        inertia[idx].z = tmp1.z;
    }
}

void Polymer::CopyInertiaHtoD(Vector3* inertia)
{
    Vector3* tmp;
    cudaMalloc((void**)&tmp, sizeof(Vector3)*_numRigid);
    cudaMemcpy(tmp, inertia, sizeof(Vector3)*_numRigid, cudaMemcpyHostToDevice);
    CopyInertiaHtoDKernel<<<((_numRigid+63)/64),64>>>(_inertia, tmp, _numRigid);
    cudaFree(tmp);
}

//use eigen here to get principal axis and the orientation
void Polymer::InitializeOrientation()
{
    Vector3* pos  = new Vector3 [_num];
    Vector3* com  = new Vector3 [_numRigid];
    float* m      = new float   [_num];
    int2* groupID = new int2 [_numGrouped];
    float* scaleFactor = new float[_numRigid];

    cudaMemcpy(pos, _position,   sizeof(Vector3)*_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(com, _positionCOM,sizeof(Vector3)*_numRigid, cudaMemcpyDeviceToHost);
    CopyMassDtoH(m);
    cudaMemcpy(groupID, _groupID, sizeof(int2)*_numGrouped, cudaMemcpyDeviceToHost);
    cudaMemcpy(scaleFactor, _scaleFactor, sizeof(float)*_numRigid, cudaMemcpyDeviceToHost);

    //diagonize moment here by Eigen
    Vector3* inertia     = new Vector3[_numRigid];
    Matrix3* orientation = new Matrix3[_numRigid];
    ComputeOrientation(inertia, orientation, pos, com, m, groupID, scaleFactor, _numGrouped, _numRigid);
    //copy back device
    cudaMemcpy(_orientation, orientation, sizeof(Matrix3)*_numRigid, cudaMemcpyHostToDevice);
    CopyInertiaHtoD(inertia);
    delete [] scaleFactor;
    delete [] inertia;
    delete [] orientation;
    delete [] m;
    delete [] groupID;
    delete [] com;
    delete [] pos;
}

__global__ void RedistributePositionKernel(Vector3* Restrict position, Vector3* Restrict positionCOM, Vector3* Restrict positionRelative, 
Matrix3* orientation, int2* groupID, int numGrouped)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < numGrouped)
    {
        int2 i = groupID[idx];
        Vector3 pos_r   = positionRelative[idx];
        Vector3 pos_com = positionCOM[i.x];
        Matrix3 R = orientation[i.x];
        position[i.y] = wrap(R*pos_r+pos_com);
    }
}

__global__ void RedistributeMomentumKernel(Vector3* Restrict momentum, int* Restrict type, Vector3* Restrict positionRelative, Vector3* Restrict momentumCOM,
Matrix3* Restrict orientation, Vector3* Restrict angularMomentum, float4* Restrict inertia, float* Restrict scaleFactor, int2* Restrict groupID, int numGrouped)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < numGrouped)
    {
        int2 i = groupID[idx];
        int t = type[i.y];

        Vector3 pos_r   = positionRelative[idx];
        Vector3 L = angularMomentum[i.x];
        float4 I = inertia[i.x];
        Matrix3 R = orientation[i.x];
        Vector3 p_com = momentumCOM[i.x];
        //get angular velocity in rb frame
        L.x = L.x / I.x;
        L.y = L.y / I.y;
        L.z = L.z / I.z;
        float m = particleMass_const[t] * scaleFactor[i.x];
        momentum[i.y] = m*((R * (L.cross(pos_r)))+(p_com/I.w));
    }
}

void Polymer::RedistributePosition()
{
    RedistributePositionKernel<<<(_numGrouped+127)/128, 128>>>(_position, _positionCOM, _positionRelative, _orientation, _groupID, _numGrouped);
    cudaDeviceSynchronize();
}

void Polymer::RedistributeMomentum()
{
    RedistributeMomentumKernel<<<(_numGrouped+127)/128, 128>>>(_momentum, _type, _positionRelative, _momentumCOM, _orientation, _angularMomentum, _inertia, _scaleFactor, _groupID, _numGrouped);
    cudaDeviceSynchronize();
}

__global__ void InitializeCOMKernel(Vector3* Restrict positionCOM, float4* Restrict inertia, Vector3* Restrict position, int* Restrict type, 
float* Restrict scaleFactor, int2* Restrict groupRange, int* Restrict particleID)
{
    __shared__ float M[64];
    __shared__ float4 com[64];
    const int rbID = blockIdx.x;
    const int tid  = threadIdx.x;
    float f = scaleFactor[rbID];
    int2 range = groupRange[rbID];
    M[tid]     = 0.f;
    com[tid]   = make_float4(0.f, 0.f, 0.f, 0.f);
    for(int i = tid+range.x; i < range.y; i+=64)
    {
        int idx = particleID[i];
        int t = type[idx];
        float m = particleMass_const[t] * f;
        Vector3 tmp = position[idx];
        tmp = m * tmp;
        M[tid] += m;
        com[tid].x += tmp.x;
        com[tid].y += tmp.y;
        com[tid].z += tmp.z;
    }
    __syncthreads();
    if(tid<32)
    {
        M[tid] += M[tid+32];
        com[tid].x += com[tid+32].x;
        com[tid].y += com[tid+32].y;
        com[tid].z += com[tid+32].z;
    }
    __syncthreads();
    if(tid<16)
    {
        M[tid] += M[tid+16];
        com[tid].x += com[tid+16].x;
        com[tid].y += com[tid+16].y;
        com[tid].z += com[tid+16].z;
    }
    __syncthreads();
    if(tid<8)
    {
        M[tid] += M[tid+8];
        com[tid].x += com[tid+8].x;
        com[tid].y += com[tid+8].y;
        com[tid].z += com[tid+8].z;
    }
    __syncthreads();
    if(tid<4)
    {
        M[tid] += M[tid+4];
        com[tid].x += com[tid+4].x;
        com[tid].y += com[tid+4].y;
        com[tid].z += com[tid+4].z;
    }
    __syncthreads();
    if(tid<2)
    {
        M[tid] += M[tid+2];
        com[tid].x += com[tid+2].x;
        com[tid].y += com[tid+2].y;
        com[tid].z += com[tid+2].z;
    }
    __syncthreads();
    if(tid == 0)
    {
        M[0] += M[1];
        com[0].x += com[1].x;
        com[0].y += com[1].y;
        com[0].z += com[1].z;
        float m = M[0];
        inertia[rbID].w = m;

        Vector3 tmp = Vector3(com[0].x, com[0].y, com[0].z);
        tmp = tmp/m;
        positionCOM[rbID] = tmp;
    } 
}

void Polymer::InitializeRigidPart()
{
    InitializeCOMKernel<<<_numRigid, 64>>>(_positionCOM, _inertia, _position, _type, _scaleFactor, _groupRange, _particleID);
    InitializeOrientation();
    int numBlocks = (_numGrouped+63)/64;
    InitializeRelativePositionKernel<<<numBlocks, 64>>>(_positionRelative, _position, _positionCOM, _orientation, _groupID, _numGrouped);
    cudaDeviceSynchronize();
}

__global__ void InitializeMomentumKernel(Vector3* momentum, int* type, curandStatePhilox4_32_10_t *state, float kT, int num)
{
    int tid = threadIdx.x + 128 * blockIdx.x;
    if(tid < num)
    {
        int t = type[tid];
        curandStatePhilox4_32_10_t localState = state[tid];
        float2 a = curand_normal2(&localState);
        float  b = curand_normal (&localState);
        float sigma = __fsqrt_rn(kT * particleMass_const[t]);
        momentum[tid] = Vector3(a.x, a.y, b)*sigma;
        state[tid] = localState;
    }
}

__global__ void InitializeRigidPartMomentumKernel(Vector3* momentumCOM, Vector3* angularMomentum, float4* inertia, curandStatePhilox4_32_10_t *state, float kT, int numRigid)
{
    int tid = threadIdx.x + 64 * blockIdx.x;
    if(tid < numRigid)
    {
        curandStatePhilox4_32_10_t localState = state[tid];
        float4 m = inertia[tid];
        float4 a = curand_normal4(&localState);
        float2 b = curand_normal2(&localState);
        float4 sigma;
        sigma.x = __fsqrt_rn(kT * m.x);
        sigma.y = __fsqrt_rn(kT * m.y);
        sigma.z = __fsqrt_rn(kT * m.z);
        sigma.w = __fsqrt_rn(kT * m.w);
        angularMomentum[tid] = Vector3(a.x*sigma.x, a.y*sigma.y, a.z*sigma.z);
        momentumCOM[tid] = Vector3(a.w, b.x, b.y)*sigma.w;
        state[tid] = localState;
    }
}


void Polymer::InitializeMomentum()
{
    InitializeMomentumKernel<<<((_num+127)/128), 128>>>(_momentum, _type, _state, _kT, _num);
    if(_numRigid > 0)
    {
        InitializeRigidPartMomentumKernel<<<((_numRigid+63)/64), 64>>>(_momentumCOM, _angularMomentum, _inertia, _state, _kT, _numRigid);
        cudaDeviceSynchronize(); 
        RedistributeMomentum();
    }
}

__global__ void ComputeRigidForceTorqueKernel(Vector3* Restrict forceCOM, Vector3* Restrict torque, Vector3* Restrict force, Vector3* Restrict positionRelative, 
Matrix3* Restrict orientation, int2* Restrict groupRange, int* Restrict particleID)
{
    __shared__ float4 totalForce [64];
    __shared__ float4 totalTorque[64];

    totalForce [threadIdx.x]  = make_float4(0.f, 0.f, 0.f, 0.f);
    totalTorque[threadIdx.x]  = make_float4(0.f, 0.f, 0.f, 0.f); 

    int2 range = groupRange[blockIdx.x];
    for(int j = range.x+threadIdx.x; j < range.y; j += 64)
    {
        int idx = particleID[j];
        Vector3 f = force[idx];
        totalForce[threadIdx.x].x += f.x;
        totalForce[threadIdx.x].y += f.y;
        totalForce[threadIdx.x].z += f.z; 
        Vector3 r   = positionRelative[j];
        Matrix3 R   = orientation[blockIdx.x];
        Vector3 tau = r.cross(R.transpose()*f);
        totalTorque[threadIdx.x].x += tau.x;
        totalTorque[threadIdx.x].y += tau.y;
        totalTorque[threadIdx.x].z += tau.z;
    }
    __syncthreads();
    if(threadIdx.x < 32)
    {
        totalForce [threadIdx.x].x += totalForce [threadIdx.x+32].x;
        totalForce [threadIdx.x].y += totalForce [threadIdx.x+32].y;
        totalForce [threadIdx.x].z += totalForce [threadIdx.x+32].z;
        totalTorque[threadIdx.x].x += totalTorque[threadIdx.x+32].x;
        totalTorque[threadIdx.x].y += totalTorque[threadIdx.x+32].y;
        totalTorque[threadIdx.x].z += totalTorque[threadIdx.x+32].z;
    }
    __syncthreads();
    if(threadIdx.x < 16)
    {
        totalForce [threadIdx.x].x += totalForce [threadIdx.x+16].x;
        totalForce [threadIdx.x].y += totalForce [threadIdx.x+16].y;
        totalForce [threadIdx.x].z += totalForce [threadIdx.x+16].z;
        totalTorque[threadIdx.x].x += totalTorque[threadIdx.x+16].x;
        totalTorque[threadIdx.x].y += totalTorque[threadIdx.x+16].y;
        totalTorque[threadIdx.x].z += totalTorque[threadIdx.x+16].z;
    }
    __syncthreads();
    if(threadIdx.x < 8)
    {
        totalForce [threadIdx.x].x += totalForce [threadIdx.x+8].x;
        totalForce [threadIdx.x].y += totalForce [threadIdx.x+8].y;
        totalForce [threadIdx.x].z += totalForce [threadIdx.x+8].z;
        totalTorque[threadIdx.x].x += totalTorque[threadIdx.x+8].x;
        totalTorque[threadIdx.x].y += totalTorque[threadIdx.x+8].y;
        totalTorque[threadIdx.x].z += totalTorque[threadIdx.x+8].z;
    }
    __syncthreads();
    if(threadIdx.x < 4)
    {
        totalForce [threadIdx.x].x += totalForce [threadIdx.x+4].x;
        totalForce [threadIdx.x].y += totalForce [threadIdx.x+4].y;
        totalForce [threadIdx.x].z += totalForce [threadIdx.x+4].z;
        totalTorque[threadIdx.x].x += totalTorque[threadIdx.x+4].x;
        totalTorque[threadIdx.x].y += totalTorque[threadIdx.x+4].y;
        totalTorque[threadIdx.x].z += totalTorque[threadIdx.x+4].z;
    }
    __syncthreads();
    if(threadIdx.x < 2)
    {
        totalForce [threadIdx.x].x += totalForce [threadIdx.x+2].x;
        totalForce [threadIdx.x].y += totalForce [threadIdx.x+2].y;
        totalForce [threadIdx.x].z += totalForce [threadIdx.x+2].z;
        totalTorque[threadIdx.x].x += totalTorque[threadIdx.x+2].x;
        totalTorque[threadIdx.x].y += totalTorque[threadIdx.x+2].y;
        totalTorque[threadIdx.x].z += totalTorque[threadIdx.x+2].z;
    }
    __syncthreads();
    if(threadIdx.x == 0)
    {
        totalForce [0].x += totalForce [1].x;
        totalForce [0].y += totalForce [1].y;
        totalForce [0].z += totalForce [1].z;
        totalTorque[0].x += totalTorque[1].x;
        totalTorque[0].y += totalTorque[1].y;
        totalTorque[0].z += totalTorque[1].z;
        forceCOM[blockIdx.x] = Vector3(totalForce[0].x, totalForce[0].y, totalForce[0].z);
        torque[blockIdx.x]   = Vector3(totalTorque[0].x, totalTorque[0].y, totalTorque[0].z);
    }
}

__global__ void CreateCellListKernel(int2* Restrict cellList, int* Restrict local_sum, Vector3* Restrict position, int num)
{
    int tid = threadIdx.x + 128 * blockIdx.x;
    if(tid < num)
    {
        float3 origin = systemBox_const[0].origin;
        float a0 = verletCellCutoff_const[0];
        Vector3 pos = position[tid];
        pos = wrap(pos);
        int4 cellId;
        int4 L = verletCellDim_const[0];
        cellId.x = floorf(__fdividef(pos.x-origin.x, a0));
        cellId.y = floorf(__fdividef(pos.y-origin.y, a0)); 
        cellId.z = floorf(__fdividef(pos.z-origin.z, a0));
        cellId.w = cellId.z + L.z * (cellId.y + L.y * cellId.x);
        cellList[tid] = make_int2(cellId.w, tid);
        atomicAdd(&local_sum[cellId.w+1],1);
    }
}

__global__ void CreateRangeKernel(int2* Restrict range, int* Restrict prefixSum)
{
    int tid = threadIdx.x + 64 * blockIdx.x;
    int numCells = verletCellDim_const[0].w;
    if(tid < numCells)
    {
        int a = prefixSum[tid];
        int b = prefixSum[tid+1];
        range[tid] = make_int2(a,b);
    }
}

__device__ inline int NeighborCellID(int dx, int cellID)
{
    int4 verletCellDim = verletCellDim_const[0];
    
    int u = (cellID / verletCellDim.z / verletCellDim.y % verletCellDim.x) + ((dx / 3 / 3  % 3)-1);
    int v = (cellID / verletCellDim.z % verletCellDim.y)                   + ((dx / 3 % 3)-1);
    int w = (cellID % verletCellDim.z)                                     + ((dx % 3)-1);

    if (verletCellDim.x == 1 and u != 0) 
        return -1;
    if (verletCellDim.y == 1 and v != 0) 
        return -1;
    if (verletCellDim.z == 1 and w != 0) 
        return -1;
    if (verletCellDim.x == 2 and (u < 0 || u > 1)) 
        return -1;
    if (verletCellDim.y == 2 and (v < 0 || v > 1)) 
        return -1;
    if (verletCellDim.z == 2 and (w < 0 || w > 1)) 
        return -1;
    
    u = (u + verletCellDim.x) % verletCellDim.x;
    v = (v + verletCellDim.y) % verletCellDim.y;
    w = (w + verletCellDim.z) % verletCellDim.z;

    return w + verletCellDim.z * (v + verletCellDim.y * u);
}
 
template<const int BlockSize,const int Size,const int N>
__global__ void CreatePairListKernel(int2* Restrict g_pair, int* Restrict g_numPairs, Vector3* Restrict position, 
                                const int2* Restrict verletCellList, const int2* Restrict verletCellRange, uint8_t* Restrict exclusionMap, float pairlistdist)
{
    __shared__ float4 __align__(16) particle[N];
    __shared__ int     Index_i[N];

    const int cellid_i     = ( blockIdx.x + gridDim.x * blockIdx.y) / Size;
    const int pid_start    = ((blockIdx.x + gridDim.x * blockIdx.y) % Size) * N;
    const int tid          =   threadIdx.x + blockDim.x * threadIdx.y
                                           + blockDim.x *  blockDim.y * threadIdx.z;
    const int idx_  = tid % N;
    const int idx__ = tid / N;
    const int Step1 = Size * N;
    //const int Step2 = Size / N; 
    const int Step2 = BlockSize / N;
    int2 rangeI = verletCellRange[cellid_i];
    int Ni = rangeI.y-rangeI.x;

    for(int pid_i = pid_start; pid_i < Ni; pid_i += Step1)
    {
        __syncthreads();
        if(tid + pid_i < Ni && tid < N)
        {
            Index_i [tid] = verletCellList[rangeI.x+pid_i+tid].y;
            Vector3 tmp = position[Index_i[tid]];
            particle[tid] = make_float4(tmp.x, tmp.y, tmp.z, 0.f);
        }
        __syncthreads();

        if(idx_ + pid_i < Ni)
        {
            int ai = Index_i[idx_];
            Vector3 A(particle[idx_]);

            //loop over neighbor directions
            for(int idx = 0; idx < 27; ++idx)
            {
                //int neighbor_cell = tex1Dfetch(NeighborsTex,idx+27*cellid_i);
                int neighbor_cell = NeighborCellID(idx, cellid_i);
                if(neighbor_cell < 0)
                    continue;

                int2 rangeJ = verletCellRange[neighbor_cell];
                int Nj = rangeJ.y-rangeJ.x;

                // In each neighbor cell, loop over particles
                for(int pid_j = idx__; pid_j < Nj; pid_j += Step2)
                {
                            
                    int aj  = verletCellList[pid_j+rangeJ.x].y;
                    if( aj <= ai )
                        continue;
                    if(aj-ai <= 8)
                    {
                        uint8_t tmp = exclusionMap[ai];
                        if(tmp & (1<<(aj-ai-1)))
                            continue;
                    }

                    Vector3 B = position[aj];

                    float dr = wrapDiff2(A,B);
                    if(dr <= pairlistdist*pairlistdist)
                    {
                        int gid = atomicAggInc( g_numPairs);
                        g_pair[gid] = make_int2(ai,aj);
                    }
                }
            }
        }
    }
}

template<const int BlockSize,const int Size,const int N>
__global__ void CreateRigidPairListKernel(int2* Restrict g_pair, int* Restrict g_numPairs, Vector3* Restrict position, 
                                     const int2* Restrict verletCellList, const int2* Restrict verletCellRange, const int* Restrict rigidBodyID, uint8_t* Restrict exclusionMap, float pairlistdist)
{

    __shared__ float4 __align__(16) particle[N];
    __shared__ int     Index_i[N];

    const int cellid_i     = ( blockIdx.x + gridDim.x * blockIdx.y) / Size;
    const int pid_start    = ((blockIdx.x + gridDim.x * blockIdx.y) % Size) * N;
    const int tid          =   threadIdx.x + blockDim.x * threadIdx.y
                                           + blockDim.x *  blockDim.y * threadIdx.z;
    const int idx_  = tid % N;
    const int idx__ = tid / N;
    const int Step1 = Size * N;
    //const int Step2 = Size / N; 
    const int Step2 = BlockSize / N;
    int2 rangeI = verletCellRange[cellid_i];
    int Ni = rangeI.y-rangeI.x;

    for(int pid_i = pid_start; pid_i < Ni; pid_i += Step1)
    {
        __syncthreads();
        if(tid + pid_i < Ni && tid < N)
        {
            Index_i [tid] = verletCellList[rangeI.x+pid_i+tid].y;
            Vector3 tmp = position[Index_i[tid]];
            particle[tid] = make_float4(tmp.x, tmp.y, tmp.z, 0.f);
        }
        __syncthreads();

        if(idx_ + pid_i < Ni)
        {
            int ai = Index_i[idx_];
            Vector3 A(particle[idx_]);
            int rb_i = rigidBodyID[ai];

            //loop over neighbor directions
            for(int idx = 0; idx < 27; ++idx)
            {
                //int neighbor_cell = tex1Dfetch(NeighborsTex,idx+27*cellid_i);
                int neighbor_cell = NeighborCellID(idx, cellid_i);
                if(neighbor_cell < 0)
                    continue;

                int2 rangeJ = verletCellRange[neighbor_cell];
                int Nj = rangeJ.y-rangeJ.x;

                // In each neighbor cell, loop over particles
                for(int pid_j = idx__; pid_j < Nj; pid_j += Step2)
                {
                            
                    int aj  = verletCellList[pid_j+rangeJ.x].y;
                    if( aj <= ai)
                        continue;

                    if(aj-ai <= 8)
                    {
                        uint8_t tmp = exclusionMap[ai];
                        if(tmp & (1<<(aj-ai-1)))
                            continue;
                    }

                    int rb_j = rigidBodyID[aj];

                    if(rb_i >= 0 and rb_j >= 0)
                    {
                        if(rb_i == rb_j)
                            continue;
                    }

                    Vector3 B = position[aj];

                    float dr = wrapDiff2(A,B);
                    if(dr <= pairlistdist*pairlistdist)
                    {
                        unsigned gid = atomicAggInc( g_numPairs);
                        g_pair[gid] = make_int2(ai,aj);
                    }
                }
            }
        }
    }
}

//Update Verlet Cell list
void Polymer::UpdateVerletCellList()
{
    cudaMemset(_verletCell->prefixSum, 0, sizeof(int)*(_verletCell->L.w+1));
    CreateCellListKernel<<<(_num+127)/128, 128>>>(_verletCell->cellList, _verletCell->prefixSum, _position, _num);
    cudaDeviceSynchronize();

    thrust::device_ptr<int2> vec(_verletCell->cellList);
    thrust::sort(vec, vec+_num, compare());
    thrust::copy(thrust::device, vec, vec+_num, _verletCell->cellList);
    thrust::inclusive_scan(thrust::device, _verletCell->prefixSum, (_verletCell->prefixSum)+(_verletCell->L.w+1), _verletCell->prefixSum);
    CreateRangeKernel<<<(_verletCell->L.w+63)/64, 64>>>(_verletCell->indexRange, _verletCell->prefixSum);
    //zero the number of pairs here
    cudaMemset(_verletCell->numPairs, 0, sizeof(int));
    if(_numGrouped > 0)
        CreateRigidPairListKernel<64,64,8><<<dim3(_verletCell->L.w,64),64>>>(_verletCell->pairs, _verletCell->numPairs, _position, _verletCell->cellList, _verletCell->indexRange, _rigidBodyID, _exclusionMap, _verletCell->a0);
    else
        CreatePairListKernel<64,64,8><<<dim3(_verletCell->L.w,64),64>>>(_verletCell->pairs, _verletCell->numPairs, _position, _verletCell->cellList, _verletCell->indexRange, _exclusionMap, _verletCell->a0);
    cudaMemcpy(&_numPairs, _verletCell->numPairs, sizeof(int), cudaMemcpyDeviceToHost);
}

__global__ void ComputeFENEKernel(Vector3* Restrict force, Vector3* Restrict position, int2* Restrict bonds, int num, int idx)
{
    int tid = threadIdx.x + NUM_THREADS * blockIdx.x;
    if(tid < num)
    {
        int2 id      = bonds[tid];
        float2 param = feneData_const[idx];
        Vector3 p1 = position[id.x];
        Vector3 p2 = position[id.y];
        p2 = wrapVecDiff(p2,p1);
        float tmp = 1.f-__fdividef(p2.length2(), param.y*param.y);
        p2 = param.x * p2 * __frcp_rn(tmp);
        atomicAdd(&force[id.x], p2);
        atomicAdd(&force[id.y],-p2);
    }
}

__global__ void ComputeFENEEnergyKernel(float* Restrict energy, Vector3* Restrict position, int2* Restrict bonds, int num, int idx)
{
    __shared__ float local[NUM_THREADS];
    int tid = threadIdx.x + NUM_THREADS * blockIdx.x;
    if(tid < num)
    {
        int2 id      = bonds[tid];
        float2 param = feneData_const[idx];
        Vector3 p1 = position[id.x];
        Vector3 p2 = position[id.y];
        p2 = wrapVecDiff(p1,p2);
        float tmp = 1.f-__fdividef(p2.length2(), param.y*param.y);
        local[threadIdx.x] = -0.5f * param.x * __logf(tmp);
    }
    else
        local[threadIdx.x] = 0.f;
    __syncthreads();
    if(threadIdx.x < 64)
        local[threadIdx.x] += local[threadIdx.x+64];
    __syncthreads();
    if(threadIdx.x < 32)
        local[threadIdx.x] += local[threadIdx.x+32];
    __syncthreads();
    if(threadIdx.x < 16)
        local[threadIdx.x] += local[threadIdx.x+16];
    __syncthreads();
    if(threadIdx.x < 8)
        local[threadIdx.x] += local[threadIdx.x+8];
    __syncthreads();
    if(threadIdx.x < 4)
        local[threadIdx.x] += local[threadIdx.x+4];
    __syncthreads();
    if(threadIdx.x < 2)
        local[threadIdx.x] += local[threadIdx.x+2];
    __syncthreads();
    if(threadIdx.x < 1)
    {
        local[0] += local[1];
        energy[blockIdx.x] += local[0];
    }
}

//TODO complete following kernel to compute virial
__global__ void ComputeFENEVirialKernel(Vector3* Restrict force, float* Restrict virial, Vector3* Restrict position, int2* Restrict bonds, int num, int idx)
{
    int tid = threadIdx.x + NUM_THREADS * blockIdx.x;
    __shared__ float local[6*NUM_THREADS];
    for(int i = 0; i < 6; ++i)
        local[6*threadIdx.x+i] = 0.;
    if(tid < num)
    {
        int2 id      = bonds[tid];
        float2 param = feneData_const[idx];
        Vector3 p1 = position[id.x];
        Vector3 p2 = position[id.y];
        p2 = wrapVecDiff(p2,p1);
        float tmp = 1.f-__fdividef(p2.length2(), param.y*param.y);
        Vector3 f = param.x * p2 * __frcp_rn(tmp);
        atomicAdd(&force[id.x], f);
        atomicAdd(&force[id.y],-f);
        local[6*threadIdx.x+0] -= f.x*p2.x;
        local[6*threadIdx.x+1] -= f.x*p2.y;
        local[6*threadIdx.x+2] -= f.x*p2.z;
        local[6*threadIdx.x+3] -= f.y*p2.y;
        local[6*threadIdx.x+4] -= f.y*p2.z;
        local[6*threadIdx.x+5] -= f.z*p2.z;
    }
    __syncthreads();
    //partially parallel reduction
    if(threadIdx.x < 64)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+64)+i];
    }__syncthreads();
    if(threadIdx.x < 32)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+32)+i];
    }__syncthreads();
    if(threadIdx.x < 16)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+16)+i];
    }__syncthreads();
    if(threadIdx.x < 8)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+8)+i];
    }__syncthreads();
    if(threadIdx.x < 4)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+4)+i];
    }__syncthreads();
    if(threadIdx.x < 2)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+2)+i];
    }__syncthreads();
    if(threadIdx.x < 1)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
        {
            local[i] += local[6+i];
            virial[6*blockIdx.x+i]+=local[i];
        }
    }
}

__global__ void ComputeFENEOnlyVirialKernel(float* Restrict virial, Vector3* Restrict position, int2* Restrict bonds, int num, int idx)
{
    int tid = threadIdx.x + NUM_THREADS * blockIdx.x;
    __shared__ float local[6*NUM_THREADS];
    for(int i = 0; i < 6; ++i)
        local[6*threadIdx.x+i] = 0.;
    if(tid < num)
    {
        int2 id      = bonds[tid];
        float2 param = feneData_const[idx];
        Vector3 p1 = position[id.x];
        Vector3 p2 = position[id.y];
        p2 = wrapVecDiff(p2,p1);
        float tmp = 1.f-__fdividef(p2.length2(), param.y*param.y);
        Vector3 f = param.x * p2 * __frcp_rn(tmp);
        local[6*threadIdx.x+0] -= f.x*p2.x;
        local[6*threadIdx.x+1] -= f.x*p2.y;
        local[6*threadIdx.x+2] -= f.x*p2.z;
        local[6*threadIdx.x+3] -= f.y*p2.y;
        local[6*threadIdx.x+4] -= f.y*p2.z;
        local[6*threadIdx.x+5] -= f.z*p2.z;
    }
    __syncthreads();
    //partially parallel reduction
    if(threadIdx.x < 64)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+64)+i];
    }__syncthreads();
    if(threadIdx.x < 32)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+32)+i];
    }__syncthreads();
    if(threadIdx.x < 16)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+16)+i];
    }__syncthreads();
    if(threadIdx.x < 8)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+8)+i];
    }__syncthreads();
    if(threadIdx.x < 4)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+4)+i];
    }__syncthreads();
    if(threadIdx.x < 2)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+2)+i];
    }__syncthreads();
    if(threadIdx.x < 1)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
        {
            local[i] += local[6+i];
            virial[6*blockIdx.x+i]+=local[i];
        }
    }
}

__global__ void ComputeHarmonicKernel(Vector3* Restrict force, Vector3* Restrict position, int2* Restrict bonds, int num, int idx)
{
    int tid = threadIdx.x + NUM_THREADS * blockIdx.x;
    if(tid < num)
    {
        int2 id      = bonds[tid];
        float2 param = harmonicData_const[idx];
        Vector3 p1 = position[id.x];
        Vector3 p2 = position[id.y];
        p2 = wrapVecDiff(p1,p2);
        float l = p2.length();
        p2 = -param.x * p2 * (1.f-__fdividef(param.y, l));
        atomicAdd(&force[id.x], p2);
        atomicAdd(&force[id.y],-p2);
    }

}

__global__ void ComputeHarmonicEnergyKernel(float* Restrict energy, Vector3* Restrict position, int2* Restrict bonds, int num, int idx)
{
    __shared__ float local[NUM_THREADS];
    int tid = threadIdx.x + NUM_THREADS * blockIdx.x;
    if(tid < num)
    {
        int2 id      = bonds[tid];
        float2 param = harmonicData_const[idx];
        Vector3 p1 = position[id.x];
        Vector3 p2 = position[id.y];
        p2 = wrapVecDiff(p1,p2);
        float l = p2.length();
        l = l-param.y;
        local[threadIdx.x] = 0.5f*param.x*l*l;
    }
    else
        local[threadIdx.x] = 0.f;
    __syncthreads();
    if(threadIdx.x < 64)
        local[threadIdx.x] += local[threadIdx.x+64];
    __syncthreads();
    if(threadIdx.x < 32)
        local[threadIdx.x] += local[threadIdx.x+32];
    __syncthreads();
    if(threadIdx.x < 16)
        local[threadIdx.x] += local[threadIdx.x+16];
    __syncthreads();
    if(threadIdx.x < 8)
        local[threadIdx.x] += local[threadIdx.x+8];
    __syncthreads();
    if(threadIdx.x < 4)
        local[threadIdx.x] += local[threadIdx.x+4];
    __syncthreads();
    if(threadIdx.x < 2)
        local[threadIdx.x] += local[threadIdx.x+2];
    __syncthreads();
    if(threadIdx.x < 1)
    {
        local[0] += local[1];
        energy[blockIdx.x] += local[0];
    }
}

__global__ void ComputeHarmonicVirialKernel(Vector3* Restrict force, float* Restrict virial, Vector3* Restrict position, int2* Restrict bonds, int num, int idx)
{
    int tid = threadIdx.x + NUM_THREADS * blockIdx.x;
    __shared__ float local[6*NUM_THREADS];
    for(int i = 0; i < 6; ++i)
        local[6*threadIdx.x+i] = 0.f;
    if(tid < num)
    {
        int2 id      = bonds[tid];
        float2 param = harmonicData_const[idx];
        Vector3 p1 = position[id.x];
        Vector3 p2 = position[id.y];
        p2 = wrapVecDiff(p1,p2);
        float l = p2.length();
        Vector3 f = -param.x * p2 * (1.f-__fdividef(param.y, l));
        atomicAdd(&force[id.x], f);
        atomicAdd(&force[id.y],-f);
        local[6*threadIdx.x+0] += f.x*p2.x;
        local[6*threadIdx.x+1] += f.x*p2.y;
        local[6*threadIdx.x+2] += f.x*p2.z;
        local[6*threadIdx.x+3] += f.y*p2.y;
        local[6*threadIdx.x+4] += f.y*p2.z;
        local[6*threadIdx.x+5] += f.z*p2.z;
    }
    __syncthreads();
    //partially parallel reduction
    if(threadIdx.x < 64)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+64)+i];
    }__syncthreads();
    if(threadIdx.x < 32)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+32)+i];
    }__syncthreads();
    if(threadIdx.x < 16)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+16)+i];
    }__syncthreads();
    if(threadIdx.x < 8)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+8)+i];
    }__syncthreads();
    if(threadIdx.x < 4)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+4)+i];
    }__syncthreads();
    if(threadIdx.x < 2)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+2)+i];
    }__syncthreads();
    if(threadIdx.x < 1)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
        {
            local[i] += local[6+i];
            virial[6*blockIdx.x+i]+=local[i];
        }
    }
}

__global__ void ComputeHarmonicOnlyVirialKernel(float* Restrict virial, Vector3* Restrict position, int2* Restrict bonds, int num, int idx)
{
    int tid = threadIdx.x + NUM_THREADS * blockIdx.x;
    __shared__ float local[6*NUM_THREADS];
    for(int i = 0; i < 6; ++i)
        local[6*threadIdx.x+i] = 0.f;
    if(tid < num)
    {
        int2 id      = bonds[tid];
        float2 param = harmonicData_const[idx];
        Vector3 p1 = position[id.x];
        Vector3 p2 = position[id.y];
        p2 = wrapVecDiff(p1,p2);
        float l = p2.length();
        Vector3 f = -param.x * p2 * (1.f-__fdividef(param.y, l));
        local[6*threadIdx.x+0] += f.x*p2.x;
        local[6*threadIdx.x+1] += f.x*p2.y;
        local[6*threadIdx.x+2] += f.x*p2.z;
        local[6*threadIdx.x+3] += f.y*p2.y;
        local[6*threadIdx.x+4] += f.y*p2.z;
        local[6*threadIdx.x+5] += f.z*p2.z;
    }
    __syncthreads();
    //partially parallel reduction
    if(threadIdx.x < 64)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+64)+i];
    }__syncthreads();
    if(threadIdx.x < 32)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+32)+i];
    }__syncthreads();
    if(threadIdx.x < 16)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+16)+i];
    }__syncthreads();
    if(threadIdx.x < 8)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+8)+i];
    }__syncthreads();
    if(threadIdx.x < 4)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+4)+i];
    }__syncthreads();
    if(threadIdx.x < 2)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+2)+i];
    }__syncthreads();
    if(threadIdx.x < 1)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
        {
            local[i] += local[6+i];
            virial[6*blockIdx.x+i]+=local[i];
        }
    }
}

__global__ void ComputeHPSModelKernel(Vector3* Restrict force, Vector3* Restrict repulsiveForce, Vector3* Restrict position, int* Restrict type, int2* Restrict Pairs, int* Restrict numPairs, int numTypes)
{
    int tid = threadIdx.x + 64 * blockIdx.x;
    int num = numPairs[0];
    for(int i = tid; i < num; i += 64 * gridDim.x)
    {
        int2 p = Pairs[i];
        int t1 = type[p.x];
        int t2 = type[p.y];
        float3 param1 = particleAttribute_const[t1];
        float3 param2 = particleAttribute_const[t2];
        Vector3 pos = position[p.x];
        pos = wrapVecDiff(pos, position[p.y]);
        float r2 = pos.length2();
        float sigma2 = 0.5f*(param1.x+param2.x);
        sigma2 = sigma2*sigma2;
        float eps = particleNBStrength_const[t1+numTypes*t2];
        if(r2 <= cubic_root_2*sigma2)
        {
            float trunc = truncateDistance_const[0];
            if(r2 < sigma2*trunc*trunc)
            {
                r2 = sigma2*trunc*trunc;
                pos = pos / pos.length();
                pos = trunc*0.5f*(param1.x+param2.x)*pos;
            }
            sigma2 = __fdividef(sigma2, r2);
            sigma2 = sigma2*sigma2*sigma2;
            sigma2 = 24.f*eps*sigma2*(2.f*sigma2-1.f);
            Vector3 f = (__fdividef(sigma2,r2)) * pos;
            atomicAdd(&repulsiveForce[p.x], f);
            atomicAdd(&repulsiveForce[p.y],-f);
        }
        else
        {
            sigma2 = __fdividef(sigma2, r2);
            sigma2 = sigma2*sigma2*sigma2;
            sigma2 = 24.f*eps*sigma2*(2.f*sigma2-1.f);
            Vector3 f = (__fdividef(sigma2,r2)) * pos;
            eps = 0.5f*(param1.y+param2.y); // aka lambda
            f = eps * f; // aka lambda
            atomicAdd(&force[p.x], f);
            atomicAdd(&force[p.y],-f);
        }
        //charge
        if(param1.z != 0.f and param2.z != 0.f)
        {
            float Kappa = Kappa_const[0];
            r2 = __fsqrt_rn(r2);
            eps = __fdividef(r2,Kappa);
            eps = __fdividef(param1.z*param2.z*(1.f+eps)*__expf(-eps), r2*r2*4.f*Pi*Dielectric_const[0]*r2);
            pos = eps*pos;
            atomicAdd(&force[p.x], pos);
            atomicAdd(&force[p.y],-pos);
        }
    }
}

__global__ void ComputeHPSModelEnergyKernel(float* Restrict energy, Vector3* Restrict position, int* Restrict type, int2* Restrict Pairs, int* Restrict numPairs, int numTypes)
{
    __shared__ float local[64];
    int tid = threadIdx.x + 64 * blockIdx.x;
    int num = numPairs[0];
    local[threadIdx.x] = 0.f;
    for(int i = tid; i < num; i += 64 * gridDim.x)
    {
        int2 p = Pairs[i];
        int t1 = type[p.x];
        int t2 = type[p.y];
        float3 param1 = particleAttribute_const[t1];
        float3 param2 = particleAttribute_const[t2];
        Vector3 pos = position[p.x];
        pos = wrapVecDiff(pos, position[p.y]);
        float r2 = pos.length2();
        float sigma2 = 0.5f*(param1.x+param2.x);
        sigma2 = sigma2*sigma2;
        float eps = particleNBStrength_const[t1+numTypes*t2];
        if(r2 <= cubic_root_2*sigma2)
        {
            sigma2 = __fdividef(sigma2, r2);
            sigma2 = sigma2*sigma2*sigma2;
            sigma2 = 4.f*eps*sigma2*(sigma2-1.f);
            local[threadIdx.x] += sigma2+(1.f-0.5f*(param1.y+param2.y))*eps;
        }
        else
        {
            sigma2 = __fdividef(sigma2, r2);
            sigma2 = sigma2*sigma2*sigma2;
            sigma2 = 4.f*eps*sigma2*(sigma2-1.f);
            local[threadIdx.x] += 0.5f*(param1.y+param2.y)*sigma2;
        }
        //charge
        if(param1.z != 0.f and param2.z != 0.f)
        {
            float Kappa = Kappa_const[0];
            r2 = __fsqrt_rn(r2);
            eps = __fdividef(r2,Kappa);
            eps = __fdividef(param1.z*param2.z,  4.f*Pi*Dielectric_const[0]*r2)*__expf(-eps);
            local[threadIdx.x] += eps;
        }
    }
    __syncthreads();
    //if(threadIdx.x < 64)
    //    local[threadIdx.x] += local[threadIdx.x+64];
    //__syncthreads();
    if(threadIdx.x < 32)
        local[threadIdx.x] += local[threadIdx.x+32];
    __syncthreads();
    if(threadIdx.x < 16)
        local[threadIdx.x] += local[threadIdx.x+16];
    __syncthreads();
    if(threadIdx.x < 8)
        local[threadIdx.x] += local[threadIdx.x+8];
    __syncthreads();
    if(threadIdx.x < 4)
        local[threadIdx.x] += local[threadIdx.x+4];
    __syncthreads();
    if(threadIdx.x < 2)
        local[threadIdx.x] += local[threadIdx.x+2];
    __syncthreads();
    if(threadIdx.x < 1)
    {
        local[0] += local[1];
        energy[blockIdx.x] += local[0];
    }
}

__global__ void ComputeHPSModelVirialKernel(Vector3* Restrict force, Vector3* Restrict repulsiveForce, float* Restrict virial, Vector3* Restrict position, int* Restrict type, int2* Restrict Pairs, int* Restrict numPairs, int numTypes)
{
    int tid = threadIdx.x + 64 * blockIdx.x;
    int num = numPairs[0];

    __shared__ float local[6*64];
    for(int i = 0; i < 6; ++i)
        local[6*threadIdx.x+i] = 0.;

    for(int i = tid; i < num; i += 64 * gridDim.x)
    {
        int2 p = Pairs[i];
        int t1 = type[p.x];
        int t2 = type[p.y];
        float3 param1 = particleAttribute_const[t1];
        float3 param2 = particleAttribute_const[t2];
        Vector3 pos = position[p.x];
        pos = wrapVecDiff(pos, position[p.y]);
        float r2 = pos.length2();
        float sigma2 = 0.5f*(param1.x+param2.x);
        sigma2 = sigma2*sigma2;
        float eps = particleNBStrength_const[t1+numTypes*t2];
        if(r2 <= cubic_root_2*sigma2)
        {
            float trunc = truncateDistance_const[0];
            if(r2 < sigma2*trunc*trunc)
            {
                r2 = sigma2*trunc*trunc;
                pos = pos / pos.length();
                pos = trunc*0.5f*(param1.x+param2.x)*pos;
            }
            sigma2 = __fdividef(sigma2, r2);
            sigma2 = sigma2*sigma2*sigma2;
            sigma2 = 24.f*eps*sigma2*(2.f*sigma2-1.f);
            Vector3 f = (__fdividef(sigma2,r2)) * pos;
            atomicAdd(&repulsiveForce[p.x], f);
            atomicAdd(&repulsiveForce[p.y],-f);
            //add virial contribution here
            local[6*threadIdx.x+0] += f.x*pos.x;
            local[6*threadIdx.x+1] += f.x*pos.y;
            local[6*threadIdx.x+2] += f.x*pos.z;
            local[6*threadIdx.x+3] += f.y*pos.y;
            local[6*threadIdx.x+4] += f.y*pos.z;
            local[6*threadIdx.x+5] += f.z*pos.z;
        }
        else
        {
            sigma2 = __fdividef(sigma2, r2);
            sigma2 = sigma2*sigma2*sigma2;
            sigma2 = 24.f*eps*sigma2*(2.f*sigma2-1.f);
            Vector3 f = (__fdividef(sigma2,r2)) * pos;
            eps = 0.5f*(param1.y+param2.y); // aka lambda
            f = eps * f; // aka lambda
            atomicAdd(&force[p.x], f);
            atomicAdd(&force[p.y],-f);
            //add virial contribution here
            local[6*threadIdx.x+0] += f.x*pos.x;
            local[6*threadIdx.x+1] += f.x*pos.y;
            local[6*threadIdx.x+2] += f.x*pos.z;
            local[6*threadIdx.x+3] += f.y*pos.y;
            local[6*threadIdx.x+4] += f.y*pos.z;
            local[6*threadIdx.x+5] += f.z*pos.z;
        }
        //charge
        if(param1.z != 0.f and param2.z != 0.f)
        {
            float Kappa = Kappa_const[0];
            r2 = __fsqrt_rn(r2);
            eps = __fdividef(r2,Kappa);
            eps = __fdividef(param1.z*param2.z*(1.f+eps)*__expf(-eps), r2*r2*4.f*Pi*Dielectric_const[0]*r2);
            Vector3 f = eps*pos;
            atomicAdd(&force[p.x], f);
            atomicAdd(&force[p.y],-f);
            //add virial contribution here
            local[6*threadIdx.x+0] += f.x*pos.x;
            local[6*threadIdx.x+1] += f.x*pos.y;
            local[6*threadIdx.x+2] += f.x*pos.z;
            local[6*threadIdx.x+3] += f.y*pos.y;
            local[6*threadIdx.x+4] += f.y*pos.z;
            local[6*threadIdx.x+5] += f.z*pos.z;
        }
    }
    __syncthreads();
    //partially parallel reduction
    if(threadIdx.x < 32)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+32)+i];
    }__syncthreads();
    if(threadIdx.x < 16)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+16)+i];
    }__syncthreads();
    if(threadIdx.x < 8)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+8)+i];
    }__syncthreads();
    if(threadIdx.x < 4)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+4)+i];
    }__syncthreads();
    if(threadIdx.x < 2)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+2)+i];
    }__syncthreads();
    if(threadIdx.x < 1)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
        {
            local[i] += local[6+i];
            virial[6*blockIdx.x+i]+=local[i];
        }
    }
}

__global__ void ComputeHPSModelOnlyVirialKernel(float* Restrict virial, Vector3* Restrict position, int* Restrict type, int2* Restrict Pairs, int* Restrict numPairs, int numTypes)
{
    int tid = threadIdx.x + 64 * blockIdx.x;
    int num = numPairs[0];

    __shared__ float local[6*64];
    for(int i = 0; i < 6; ++i)
        local[6*threadIdx.x+i] = 0.;

    for(int i = tid; i < num; i += 64 * gridDim.x)
    {
        int2 p = Pairs[i];
        int t1 = type[p.x];
        int t2 = type[p.y];
        float3 param1 = particleAttribute_const[t1];
        float3 param2 = particleAttribute_const[t2];
        Vector3 pos = position[p.x];
        pos = wrapVecDiff(pos, position[p.y]);
        float r2 = pos.length2();
        float sigma2 = 0.5f*(param1.x+param2.x);
        sigma2 = sigma2*sigma2;
        float eps = particleNBStrength_const[t1+numTypes*t2];
        if(r2 <= cubic_root_2*sigma2)
        {
            float trunc = truncateDistance_const[0];
            if(r2 < sigma2*trunc*trunc)
            {
                r2 = sigma2*trunc*trunc;
                pos = pos / pos.length();
                pos = trunc*0.5f*(param1.x+param2.x)*pos;
            }
            sigma2 = __fdividef(sigma2, r2);
            sigma2 = sigma2*sigma2*sigma2;
            sigma2 = 24.f*eps*sigma2*(2.f*sigma2-1.f);
            Vector3 f = (__fdividef(sigma2,r2)) * pos;
            //add virial contribution here
            local[6*threadIdx.x+0] += f.x*pos.x;
            local[6*threadIdx.x+1] += f.x*pos.y;
            local[6*threadIdx.x+2] += f.x*pos.z;
            local[6*threadIdx.x+3] += f.y*pos.y;
            local[6*threadIdx.x+4] += f.y*pos.z;
            local[6*threadIdx.x+5] += f.z*pos.z;
        }
        else
        {
            sigma2 = __fdividef(sigma2, r2);
            sigma2 = sigma2*sigma2*sigma2;
            sigma2 = 24.f*eps*sigma2*(2.f*sigma2-1.f);
            Vector3 f = (__fdividef(sigma2,r2)) * pos;
            eps = 0.5f*(param1.y+param2.y); // aka lambda
            f = eps * f; // aka lambda
            //add virial contribution here
            local[6*threadIdx.x+0] += f.x*pos.x;
            local[6*threadIdx.x+1] += f.x*pos.y;
            local[6*threadIdx.x+2] += f.x*pos.z;
            local[6*threadIdx.x+3] += f.y*pos.y;
            local[6*threadIdx.x+4] += f.y*pos.z;
            local[6*threadIdx.x+5] += f.z*pos.z;
        }
        //charge
        if(param1.z != 0.f and param2.z != 0.f)
        {
            float Kappa = Kappa_const[0];
            r2 = __fsqrt_rn(r2);
            eps = __fdividef(r2,Kappa);
            eps = __fdividef(param1.z*param2.z*(1.f+eps)*__expf(-eps), r2*r2*4.f*Pi*Dielectric_const[0]*r2);
            Vector3 f = eps*pos;
            //add virial contribution here
            local[6*threadIdx.x+0] += f.x*pos.x;
            local[6*threadIdx.x+1] += f.x*pos.y;
            local[6*threadIdx.x+2] += f.x*pos.z;
            local[6*threadIdx.x+3] += f.y*pos.y;
            local[6*threadIdx.x+4] += f.y*pos.z;
            local[6*threadIdx.x+5] += f.z*pos.z;
        }
    }
    __syncthreads();
    //partially parallel reduction
    if(threadIdx.x < 32)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+32)+i];
    }__syncthreads();
    if(threadIdx.x < 16)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+16)+i];
    }__syncthreads();
    if(threadIdx.x < 8)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+8)+i];
    }__syncthreads();
    if(threadIdx.x < 4)
    {
        #pragma urnoll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+4)+i];
    }__syncthreads();
    if(threadIdx.x < 2)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+2)+i];
    }__syncthreads();
    if(threadIdx.x < 1)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
        {
            local[i] += local[6+i];
            virial[6*blockIdx.x+i]+=local[i];
        }
    }
}

//TODO complete this KH model
__global__ void ComputeKHModelKernel(Vector3* Restrict force, Vector3* Restrict repulsiveForce, Vector3* Restrict position, int* Restrict type, int2* Restrict Pairs, int* Restrict numPairs, int numTypes)
{
    int tid = threadIdx.x + 64 * blockIdx.x;
    int num = numPairs[0];
    for(int i = tid; i < num; i += 64 * gridDim.x)
    {
        int2 p = Pairs[i];
        int t1 = type[p.x];
        int t2 = type[p.y];
        float3 param1 = particleAttribute_const[t1];
        float3 param2 = particleAttribute_const[t2];
        Vector3 pos = position[p.x];
        pos = wrapVecDiff(pos, position[p.y]);
        float r2 = pos.length2();
        float sigma2 = 0.5f*(param1.x+param2.x);
        sigma2 = sigma2*sigma2;
        float eps = particleNBStrength_const[t1+numTypes*t2];
        param1.y = (eps > 0.f) ? -1.f : 1.f; //aka lambda
        param2.y = (eps < 0.f) ? -eps : eps; //absolute epsilon
        if(r2 <= cubic_root_2*sigma2)
        {
            float trunc = truncateDistance_const[0];
            if(r2 < sigma2*trunc*trunc)
            {
                r2 = sigma2*trunc*trunc;
                pos = pos / pos.length();
                pos = trunc*0.5f*(param1.x+param2.x)*pos;
            }
            sigma2 = __fdividef(sigma2, r2);
            sigma2 = sigma2*sigma2*sigma2;
            sigma2 = 24.f*param2.y*sigma2*(2.f*sigma2-1.f);
            Vector3 f = (__fdividef(sigma2,r2)) * pos;
            atomicAdd(&repulsiveForce[p.x], f);
            atomicAdd(&repulsiveForce[p.y],-f);
        }
        else
        {
            sigma2 = __fdividef(sigma2, r2);
            sigma2 = sigma2*sigma2*sigma2;
            sigma2 = -24.f*eps*sigma2*(2.f*sigma2-1.f);
            Vector3 f = (__fdividef(sigma2,r2)) * pos;
            atomicAdd(&force[p.x], f);
            atomicAdd(&force[p.y],-f);
        }
        //charge
        if(param1.z != 0.f and param2.z != 0.f)
        {
            float Kappa = Kappa_const[0];
            r2 = __fsqrt_rn(r2);
            eps = __fdividef(r2,Kappa);
            eps = __fdividef(param1.z*param2.z*(1.f+eps)*__expf(-eps), r2*r2*4.f*Pi*Dielectric_const[0]*r2);
            pos = eps*pos;
            atomicAdd(&force[p.x], pos);
            atomicAdd(&force[p.y],-pos);
        }
    }
}

__global__ void ComputeKHModelEnergyKernel(float* Restrict energy, Vector3* Restrict position, int* Restrict type, int2* Restrict Pairs, int* Restrict numPairs, int numTypes)
{
    __shared__ float local[64];
    int tid = threadIdx.x + 64 * blockIdx.x;
    int num = numPairs[0];
    local[threadIdx.x] = 0.f;
    for(int i = tid; i < num; i += 64 * gridDim.x)
    {
        int2 p = Pairs[i];
        int t1 = type[p.x];
        int t2 = type[p.y];
        float3 param1 = particleAttribute_const[t1];
        float3 param2 = particleAttribute_const[t2];
        Vector3 pos = position[p.x];
        pos = wrapVecDiff(pos, position[p.y]);
        float r2 = pos.length2();
        float sigma2 = 0.5f*(param1.x+param2.x);
        sigma2 = sigma2*sigma2;
        float eps = particleNBStrength_const[t1+numTypes*t2];
        param1.y = (eps > 0.f) ? -1.f : 1.f;
        param2.y = (eps < 0.f) ? -eps : eps;

        if(r2 <= cubic_root_2*sigma2)
        {
            sigma2 = __fdividef(sigma2, r2);
            sigma2 = sigma2*sigma2*sigma2;
            sigma2 = 4.f*param2.y*sigma2*(sigma2-1.f);
            local[threadIdx.x] += sigma2+(1.f-param1.y)*param2.y;
        }
        else
        {
            sigma2 = __fdividef(sigma2, r2);
            sigma2 = sigma2*sigma2*sigma2;
            sigma2 = -4.f*eps*sigma2*(sigma2-1.f);
            local[threadIdx.x] += sigma2;
        }
        //charge
        if(param1.z != 0.f and param2.z != 0.f)
        {
            float Kappa = Kappa_const[0];
            r2 = __fsqrt_rn(r2);
            eps = __fdividef(r2,Kappa);
            eps = __fdividef(param1.z*param2.z,  4.f*Pi*Dielectric_const[0]*r2)*__expf(-eps);
            local[threadIdx.x] += eps;
        }
    }
    __syncthreads();
    //if(threadIdx.x < 64)
    //    local[threadIdx.x] += local[threadIdx.x+64];
    //__syncthreads();
    if(threadIdx.x < 32)
        local[threadIdx.x] += local[threadIdx.x+32];
    __syncthreads();
    if(threadIdx.x < 16)
        local[threadIdx.x] += local[threadIdx.x+16];
    __syncthreads();
    if(threadIdx.x < 8)
        local[threadIdx.x] += local[threadIdx.x+8];
    __syncthreads();
    if(threadIdx.x < 4)
        local[threadIdx.x] += local[threadIdx.x+4];
    __syncthreads();
    if(threadIdx.x < 2)
        local[threadIdx.x] += local[threadIdx.x+2];
    __syncthreads();
    if(threadIdx.x < 1)
    {
        local[0] += local[1];
        energy[blockIdx.x] += local[0];
    }
}

__global__ void ComputeKHModelVirialKernel(Vector3* Restrict force, Vector3* Restrict repulsiveForce, float* Restrict virial, Vector3* Restrict position, int* Restrict type, int2* Restrict Pairs, int* Restrict numPairs, int numTypes)
{
    int tid = threadIdx.x + 64 * blockIdx.x;
    int num = numPairs[0];

    __shared__ float local[6*64];
    for(int i = 0; i < 6; ++i)
        local[6*threadIdx.x+i] = 0.;

    for(int i = tid; i < num; i += 64 * gridDim.x)
    {
        int2 p = Pairs[i];
        int t1 = type[p.x];
        int t2 = type[p.y];
        float3 param1 = particleAttribute_const[t1];
        float3 param2 = particleAttribute_const[t2];
        Vector3 pos = position[p.x];
        pos = wrapVecDiff(pos, position[p.y]);
        float r2 = pos.length2();
        float sigma2 = 0.5f*(param1.x+param2.x);
        sigma2 = sigma2*sigma2;
        float eps = particleNBStrength_const[t1+numTypes*t2];
        param1.y = (eps > 0.f) ? -1.f : 1.f;
        param2.y = (eps < 0.f) ? -eps : eps;

        if(r2 <= cubic_root_2*sigma2)
        {
            float trunc = truncateDistance_const[0];
            if(r2 < sigma2*trunc*trunc)
            {
                r2 = sigma2*trunc*trunc;
                pos = pos / pos.length();
                pos = trunc*0.5f*(param1.x+param2.x)*pos;
            }
            sigma2 = __fdividef(sigma2, r2);
            sigma2 = sigma2*sigma2*sigma2;
            sigma2 = 24.f*param2.y*sigma2*(2.f*sigma2-1.f);
            Vector3 f = (__fdividef(sigma2,r2)) * pos;
            atomicAdd(&repulsiveForce[p.x], f);
            atomicAdd(&repulsiveForce[p.y],-f);
            //add virial contribution here
            local[6*threadIdx.x+0] += f.x*pos.x;
            local[6*threadIdx.x+1] += f.x*pos.y;
            local[6*threadIdx.x+2] += f.x*pos.z;
            local[6*threadIdx.x+3] += f.y*pos.y;
            local[6*threadIdx.x+4] += f.y*pos.z;
            local[6*threadIdx.x+5] += f.z*pos.z;
        }
        else
        {
            sigma2 = __fdividef(sigma2, r2);
            sigma2 = sigma2*sigma2*sigma2;
            sigma2 = -24.f*eps*sigma2*(2.f*sigma2-1.f);
            Vector3 f = (__fdividef(sigma2,r2)) * pos;
            atomicAdd(&force[p.x], f);
            atomicAdd(&force[p.y],-f);
            //add virial contribution here
            local[6*threadIdx.x+0] += f.x*pos.x;
            local[6*threadIdx.x+1] += f.x*pos.y;
            local[6*threadIdx.x+2] += f.x*pos.z;
            local[6*threadIdx.x+3] += f.y*pos.y;
            local[6*threadIdx.x+4] += f.y*pos.z;
            local[6*threadIdx.x+5] += f.z*pos.z;
        }
        //charge
        if(param1.z != 0.f and param2.z != 0.f)
        {
            float Kappa = Kappa_const[0];
            r2 = __fsqrt_rn(r2);
            eps = __fdividef(r2,Kappa);
            eps = __fdividef(param1.z*param2.z*(1.f+eps)*__expf(-eps), r2*r2*4.f*Pi*Dielectric_const[0]*r2);
            Vector3 f = eps*pos;
            atomicAdd(&force[p.x], f);
            atomicAdd(&force[p.y],-f);
            //add virial contribution here
            local[6*threadIdx.x+0] += f.x*pos.x;
            local[6*threadIdx.x+1] += f.x*pos.y;
            local[6*threadIdx.x+2] += f.x*pos.z;
            local[6*threadIdx.x+3] += f.y*pos.y;
            local[6*threadIdx.x+4] += f.y*pos.z;
            local[6*threadIdx.x+5] += f.z*pos.z;
        }
    }
    __syncthreads();
    //partially parallel reduction
    if(threadIdx.x < 32)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+32)+i];
    }__syncthreads();
    if(threadIdx.x < 16)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+16)+i];
    }__syncthreads();
    if(threadIdx.x < 8)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+8)+i];
    }__syncthreads();
    if(threadIdx.x < 4)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+4)+i];
    }__syncthreads();
    if(threadIdx.x < 2)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+2)+i];
    }__syncthreads();
    if(threadIdx.x < 1)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
        {
            local[i] += local[6+i];
            virial[6*blockIdx.x+i]+=local[i];
        }
    }
}

__global__ void ComputeKHModelOnlyVirialKernel(float* Restrict virial, Vector3* Restrict position, int* Restrict type, int2* Restrict Pairs, int* Restrict numPairs, int numTypes)
{
    int tid = threadIdx.x + 64 * blockIdx.x;
    int num = numPairs[0];

    __shared__ float local[6*64];
    for(int i = 0; i < 6; ++i)
        local[6*threadIdx.x+i] = 0.;

    for(int i = tid; i < num; i += 64 * gridDim.x)
    {
        int2 p = Pairs[i];
        int t1 = type[p.x];
        int t2 = type[p.y];
        float3 param1 = particleAttribute_const[t1];
        float3 param2 = particleAttribute_const[t2];
        Vector3 pos = position[p.x];
        pos = wrapVecDiff(pos, position[p.y]);
        float r2 = pos.length2();
        float sigma2 = 0.5f*(param1.x+param2.x);
        sigma2 = sigma2*sigma2;
        float eps = particleNBStrength_const[t1+numTypes*t2];
        param1.y = (eps > 0.f) ? -1.f : 1.f;
        param2.y = (eps < 0.f) ? -eps : eps;

        if(r2 <= cubic_root_2*sigma2)
        {
            float trunc = truncateDistance_const[0];
            if(r2 < sigma2*trunc*trunc)
            {
                r2 = sigma2*trunc*trunc;
                pos = pos / pos.length();
                pos = trunc*0.5f*(param1.x+param2.x)*pos;
            }
            sigma2 = __fdividef(sigma2, r2);
            sigma2 = sigma2*sigma2*sigma2;
            sigma2 = 24.f*param2.y*sigma2*(2.f*sigma2-1.f);
            Vector3 f = (__fdividef(sigma2,r2)) * pos;
            //add virial contribution here
            local[6*threadIdx.x+0] += f.x*pos.x;
            local[6*threadIdx.x+1] += f.x*pos.y;
            local[6*threadIdx.x+2] += f.x*pos.z;
            local[6*threadIdx.x+3] += f.y*pos.y;
            local[6*threadIdx.x+4] += f.y*pos.z;
            local[6*threadIdx.x+5] += f.z*pos.z;
        }
        else
        {
            sigma2 = __fdividef(sigma2, r2);
            sigma2 = sigma2*sigma2*sigma2;
            sigma2 = -24.f*eps*sigma2*(2.f*sigma2-1.f);
            Vector3 f = (__fdividef(sigma2,r2)) * pos;
            //add virial contribution here
            local[6*threadIdx.x+0] += f.x*pos.x;
            local[6*threadIdx.x+1] += f.x*pos.y;
            local[6*threadIdx.x+2] += f.x*pos.z;
            local[6*threadIdx.x+3] += f.y*pos.y;
            local[6*threadIdx.x+4] += f.y*pos.z;
            local[6*threadIdx.x+5] += f.z*pos.z;
        }
        //charge
        if(param1.z != 0.f and param2.z != 0.f)
        {
            float Kappa = Kappa_const[0];
            r2 = __fsqrt_rn(r2);
            eps = __fdividef(r2,Kappa);
            eps = __fdividef(param1.z*param2.z*(1.f+eps)*__expf(-eps), r2*r2*4.f*Pi*Dielectric_const[0]*r2);
            Vector3 f = eps*pos;
            //add virial contribution here
            local[6*threadIdx.x+0] += f.x*pos.x;
            local[6*threadIdx.x+1] += f.x*pos.y;
            local[6*threadIdx.x+2] += f.x*pos.z;
            local[6*threadIdx.x+3] += f.y*pos.y;
            local[6*threadIdx.x+4] += f.y*pos.z;
            local[6*threadIdx.x+5] += f.z*pos.z;
        }
    }
    __syncthreads();
    //partially parallel reduction
    if(threadIdx.x < 32)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+32)+i];
    }__syncthreads();
    if(threadIdx.x < 16)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+16)+i];
    }__syncthreads();
    if(threadIdx.x < 8)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+8)+i];
    }__syncthreads();
    if(threadIdx.x < 4)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+4)+i];
    }__syncthreads();
    if(threadIdx.x < 2)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+2)+i];
    }__syncthreads();
    if(threadIdx.x < 1)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i)
        {
            local[i] += local[6+i];
            virial[6*blockIdx.x+i]+=local[i];
        }
    }
}

__global__ void ComputeTotalNonBondedForce(Vector3* Restrict f, Vector3* Restrict f0, int num)
{
    int tid = threadIdx.x + 256 * blockIdx.x;
    if(tid < num)
        f[tid] = f[tid] + f0[tid];
}

/*HPS model here*/
void Polymer::ComputeHPSModel()
{
    int numPairs = ((_numPairs+63)/64 > 65535) ? 65535 : (_numPairs+63)/64;
    ComputeHPSModelKernel<<<numPairs, 64>>>(_force, _repulsiveForce, _position, _type, _verletCell->pairs, _verletCell->numPairs, _numTypes);

    ComputeTotalNonBondedForce<<<(_num+255)/256, 256>>>(_force, _repulsiveForce, _num);
    cudaDeviceSynchronize();
}

void Polymer::ComputeHPSModelEnergy()
{
    int numPairs = ((_numPairs+63)/64 > 65535) ? 65535 : (_numPairs+63)/64;
    ComputeHPSModelEnergyKernel<<<numPairs, 64>>>(_energy, _position, _type, _verletCell->pairs, _verletCell->numPairs, _numTypes);
    cudaDeviceSynchronize();
}

void Polymer::ComputeHPSModelVirial()
{
    int numPairs = ((_numPairs+63)/64 > 65535) ? 65535 : (_numPairs+63)/64;
    ComputeHPSModelVirialKernel<<<numPairs, 64>>>(_force, _repulsiveForce, _virial, _position, _type, _verletCell->pairs, _verletCell->numPairs, _numTypes);

    ComputeTotalNonBondedForce<<<(_num+255)/256, 256>>>(_force, _repulsiveForce, _num);
    cudaDeviceSynchronize();
}


void Polymer::ComputeHPSModelOnlyVirial()
{
    int numPairs = ((_numPairs+63)/64 > 65535) ? 65535 : (_numPairs+63)/64;
    ComputeHPSModelOnlyVirialKernel<<<numPairs, 64>>>(_virial, _position, _type, _verletCell->pairs, _verletCell->numPairs, _numTypes);
    cudaDeviceSynchronize();
}
/* HPS model end here*/

/* KH model start here*/
void Polymer::ComputeKHModel()
{
    int numPairs = ((_numPairs+63)/64 > 65535) ? 65535 : (_numPairs+63)/64;
    ComputeKHModelKernel<<<numPairs, 64>>>(_force, _repulsiveForce, _position, _type, _verletCell->pairs, _verletCell->numPairs, _numTypes);

    ComputeTotalNonBondedForce<<<(_num+255)/256, 256>>>(_force, _repulsiveForce, _num);
    cudaDeviceSynchronize();
}

void Polymer::ComputeKHModelEnergy()
{
    int numPairs = ((_numPairs+63)/64 > 65535) ? 65535 : (_numPairs+63)/64;
    ComputeKHModelEnergyKernel<<<numPairs, 64>>>(_energy, _position, _type, _verletCell->pairs, _verletCell->numPairs, _numTypes);
    cudaDeviceSynchronize();
}

void Polymer::ComputeKHModelVirial()
{
    int numPairs = ((_numPairs+63)/64 > 65535) ? 65535 : (_numPairs+63)/64;
    ComputeKHModelVirialKernel<<<numPairs, 64>>>(_force, _repulsiveForce, _virial, _position, _type, _verletCell->pairs, _verletCell->numPairs, _numTypes);

    ComputeTotalNonBondedForce<<<(_num+255)/256, 256>>>(_force, _repulsiveForce, _num);
    cudaDeviceSynchronize();
}

void Polymer::ComputeKHModelOnlyVirial()
{
    int numPairs = ((_numPairs+63)/64 > 65535) ? 65535 : (_numPairs+63)/64;
    ComputeKHModelOnlyVirialKernel<<<numPairs, 64>>>(_virial, _position, _type, _verletCell->pairs, _verletCell->numPairs, _numTypes);
    cudaDeviceSynchronize();
}
/* KH model end here*/

__global__ void PairReductionKernel(float* Restrict energy)
{
    __shared__ float local[1024];
    int tid = threadIdx.x;
    local[tid] = 0.f;
    for(int i = tid; i < 65536; i += 1024)
        local[tid] += energy[i];
    __syncthreads();
    if(tid < 512)
        local[tid] += local[tid+512];
    __syncthreads();
    if(tid < 256)
        local[tid] += local[tid+256];
    __syncthreads();
    if(tid < 128)
        local[tid] += local[tid+128];
    __syncthreads();
    if(tid < 64)
        local[tid] += local[tid+64];
    __syncthreads();
    if(tid < 32)
        local[tid] += local[tid+32];
    __syncthreads();
    if(tid < 16)
        local[tid] += local[tid+16];
    __syncthreads();
    if(tid < 8)
        local[tid] += local[tid+8];
    __syncthreads();
    if(tid < 4)
        local[tid] += local[tid+4];
    __syncthreads();
    if(tid < 2)
        local[tid] += local[tid+2];
    __syncthreads();
    if(tid < 1)
    {
        local[0] += local[1];
        energy[0] = local[0];
    }

}

__global__ void ComputeParticleKineticEnergyKernel1(float* Restrict energy, Vector3* Restrict momentum, int* Restrict type, int num)
{
    int tid = threadIdx.x + 256 * blockIdx.x;
    __shared__ float local[256];
    local[threadIdx.x] = 0.f;
    if(tid < num)
    {
        int t = type[tid];
        local[threadIdx.x] = __fdividef(momentum[tid].length2(), particleMass_const[t])*0.5f;
    }
    __syncthreads();
    if(threadIdx.x < 128)
        local[threadIdx.x] += local[threadIdx.x+128];
    __syncthreads();
    if(threadIdx.x < 64)
        local[threadIdx.x] += local[threadIdx.x+64];
    __syncthreads();
    if(threadIdx.x < 32)
        local[threadIdx.x] += local[threadIdx.x+32];
    __syncthreads();
    if(threadIdx.x < 16)
        local[threadIdx.x] += local[threadIdx.x+16];
    __syncthreads();
    if(threadIdx.x < 8)
        local[threadIdx.x] += local[threadIdx.x+8];
    __syncthreads();
    if(threadIdx.x < 4)
        local[threadIdx.x] += local[threadIdx.x+4];
    __syncthreads();
    if(threadIdx.x < 2)
        local[threadIdx.x] += local[threadIdx.x+2];
    __syncthreads();
    if(threadIdx.x < 1)
    {
        local[0] += local[1];
        energy[blockIdx.x] += local[0];
    }
}

__global__ void ComputeParticleKineticEnergyKernel2(float* Restrict energy, Vector3* Restrict momentum, int* Restrict noGroupedID, int* Restrict type, int num)
{
    int tid = threadIdx.x + 256 * blockIdx.x;
    __shared__ float local[256];
    local[threadIdx.x] = 0.f;
    if(tid < num)
    {
        int id = noGroupedID[tid];
        int t = type[id];
        local[threadIdx.x] = __fdividef(momentum[id].length2(), particleMass_const[t])*0.5f;
    }
    __syncthreads();
    if(threadIdx.x < 128)
        local[threadIdx.x] += local[threadIdx.x+128];
    __syncthreads();
    if(threadIdx.x < 64)
        local[threadIdx.x] += local[threadIdx.x+64];
    __syncthreads();
    if(threadIdx.x < 32)
        local[threadIdx.x] += local[threadIdx.x+32];
    __syncthreads();
    if(threadIdx.x < 16)
        local[threadIdx.x] += local[threadIdx.x+16];
    __syncthreads();
    if(threadIdx.x < 8)
        local[threadIdx.x] += local[threadIdx.x+8];
    __syncthreads();
    if(threadIdx.x < 4)
        local[threadIdx.x] += local[threadIdx.x+4];
    __syncthreads();
    if(threadIdx.x < 2)
        local[threadIdx.x] += local[threadIdx.x+2];
    __syncthreads();
    if(threadIdx.x < 1)
    {
        local[0] += local[1];
        energy[blockIdx.x] += local[0];
    }
}

__global__ void ComputeRigidBodyKineticEnergyKernel(float* Restrict energy, Vector3* Restrict momentumCOM, Vector3* Restrict angularMomentum, float4* Restrict inertia, int num)
{
    __shared__ float local[128];
    int tid = threadIdx.x + 128 * blockIdx.x;
    local[threadIdx.x] = 0.f;
    if(tid < num)
    {
        Vector3 p = momentumCOM[tid];
        Vector3 l = angularMomentum[tid];
        float4  I = inertia[tid];
        float e = 0.5f*(__fdividef(p.length2(), I.w) + __fdividef(l.x*l.x, I.x) + __fdividef(l.y*l.y, I.y) + __fdividef(l.z*l.z, I.z));
        local[threadIdx.x] = e;
    }
    __syncthreads();
    if(threadIdx.x < 64)
        local[threadIdx.x] += local[threadIdx.x+64];
    __syncthreads();
    if(threadIdx.x < 32)
        local[threadIdx.x] += local[threadIdx.x+32];
    __syncthreads();
    if(threadIdx.x < 16)
        local[threadIdx.x] += local[threadIdx.x+16];
    __syncthreads();
    if(threadIdx.x < 8)
        local[threadIdx.x] += local[threadIdx.x+8];
    __syncthreads();
    if(threadIdx.x < 4)
        local[threadIdx.x] += local[threadIdx.x+4];
    __syncthreads();
    if(threadIdx.x < 2)
        local[threadIdx.x] += local[threadIdx.x+2];
    __syncthreads();
    if(threadIdx.x < 1)
    {
        local[0] += local[1];
        energy[blockIdx.x] += local[0];
    }
}

void Polymer::ComputeParticleKineticEnergy()
{
    ComputeParticleKineticEnergyKernel1<<<(_num+255)/256, 256>>>(_energy, _momentum, _type, _num);
    cudaDeviceSynchronize();
}

void Polymer::ComputeRigidBodyKineticEnergy()
{
    if(_num > _numGrouped)
        ComputeParticleKineticEnergyKernel2<<<(_num-_numGrouped+255)/256, 256>>>(_energy, _momentum, _noGroupedID, _type, _num-_numGrouped);
    ComputeRigidBodyKineticEnergyKernel<<<(_numRigid+127)/128, 128>>>(_energy, _momentumCOM, _angularMomentum, _inertia, _numRigid);
    cudaDeviceSynchronize();
}

#ifdef TEST
#define ExternalForce 0.06f
#define Scale 1
__global__ void AddExternalForceKernel(Vector3* force)
{
    force[0] += Vector3(0.f, 0.f, Scale*ExternalForce);
    force[1] -= Vector3(0.f, 0.f, Scale*ExternalForce);
}
#endif

void Polymer::ComputeForce(const int& s)
{
    cudaMemset(_force, 0, sizeof(Vector3)*_num);
    cudaMemset(_repulsiveForce, 0, sizeof(Vector3)*_num);

    if(s % _verletCell->decompFreq == 0)
        UpdateVerletCellList();
    if(_numPairs > 0)
    {
        if(_modelType == 0)
            ComputeHPSModel();
        else if(_modelType == 1)
            ComputeKHModel(); 
        else if(_modelType == 2)
            ComputeONCModel();
    }
    for(int i = 0; i < _fene.size(); ++i)
        ComputeFENEKernel<<<(_fene[i]->num+NUM_THREADS-1)/NUM_THREADS, NUM_THREADS>>>(_force, _position, _fene[i]->bond, _fene[i]->num, i);
    for(int i = 0; i < _harmonic.size(); ++i)
        ComputeHarmonicKernel<<<(_harmonic[i]->num+NUM_THREADS-1)/NUM_THREADS, NUM_THREADS>>>(_force, _position, _harmonic[i]->bond, _harmonic[i]->num, i);

    for(auto it = _angleTorsionBase.begin(); it != _angleTorsionBase.end(); ++it)
        (*it)->ComputeForce(_force, _position);

    //add externl potential
    for(auto it = _externalPotentialBase.begin(); it != _externalPotentialBase.end(); ++it)
        (*it)->AddForce(_force, _position, _type, _num);
    for(auto it = _restraintBase.begin(); it != _restraintBase.end(); ++it)
        (*it)->AddForce(_force, _position);

    #ifdef TEST
    AddExternalForceKernel<<<1,1>>>(_force);
    #endif
    if(_numGrouped > 0)
        ComputeRigidForceTorqueKernel<<<_numRigid, 64>>>(_forceCOM, _torque, _force, _positionRelative, _orientation, _groupRange, _particleID);
    cudaDeviceSynchronize();
}

void Polymer::ComputeTotalEnergy()
{
    cudaMemset(_energy, 0, sizeof(float)*65536);
    if(_numGrouped > 0)
        ComputeRigidBodyKineticEnergy();
    else
        ComputeParticleKineticEnergy();
    PairReductionKernel<<<1,1024>>>(_energy); 
    float energy;
    cudaMemcpy(&energy, _energy, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Kinetic energy is " << energy/_kT << " (kT), ";
    cudaMemset(_energy, 0, sizeof(float)*65536);
    if(_numPairs > 0)
    {
        if(_modelType == 0)
            ComputeHPSModelEnergy();
        else if(_modelType == 1)
            ComputeKHModelEnergy();
        else if(_modelType == 2)
            ComputeONCModelEnergy();
    }
    for(int i = 0; i < _fene.size(); ++i)
        ComputeFENEEnergyKernel<<<(_fene[i]->num+NUM_THREADS-1)/NUM_THREADS, NUM_THREADS>>>(_energy,  _position, _fene[i]->bond, _fene[i]->num, i);
    for(int i = 0; i < _harmonic.size(); ++i)
        ComputeHarmonicEnergyKernel<<<(_harmonic[i]->num+NUM_THREADS-1)/NUM_THREADS, NUM_THREADS>>>(_energy, _position, _harmonic[i]->bond, _harmonic[i]->num, i);

    for(auto it = _angleTorsionBase.begin(); it != _angleTorsionBase.end(); ++it)
        (*it)->ComputeEnergy(_energy, _position);
    for(auto it = _externalPotentialBase.begin(); it != _externalPotentialBase.end(); ++it)
        (*it)->AddEnergy(_energy, _position, _type, _num);
    for(auto it = _restraintBase.begin(); it != _restraintBase.end(); ++it)
        (*it)->AddEnergy(_energy, _position);
 
    PairReductionKernel<<<1,1024>>>(_energy);
    cudaMemcpy(&energy, _energy, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Potential energy is " << energy << " (kcal_mol)" << std::endl;
}

__global__ void PairReductionVirialKernel(float* virial)
{
    __shared__ float local[1024*6];
    int tid = threadIdx.x;
    for(int i = 0; i < 6; ++i)
        local[6*tid+i] = 0.f;
    for(int i = tid; i < 65536; i += 1024)
    {
        #pragma unroll
        for(int j = 0; j < 6; ++j)
            local[6*tid+j] += virial[6*i+j];
    }
    __syncthreads();
    if(tid < 512)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i) 
            local[6*tid+i] += local[6*(tid+512)+i];
    }__syncthreads();
    if(tid < 256)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i) 
            local[6*tid+i] += local[6*(tid+256)+i];
    }__syncthreads();
    if(tid < 128)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i) 
            local[6*tid+i] += local[6*(tid+128)+i];
    }__syncthreads();
    if(tid < 64)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i) 
            local[6*tid+i] += local[6*(tid+64)+i];
    }__syncthreads();
    if(tid < 32)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i) 
            local[6*tid+i] += local[6*(tid+32)+i];
    }__syncthreads();
    if(tid < 16)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i) 
            local[6*tid+i] += local[6*(tid+16)+i];
    }__syncthreads();
    if(tid < 8)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i) 
            local[6*tid+i] += local[6*(tid+8)+i];
    }__syncthreads();
    if(tid < 4)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i) 
            local[6*tid+i] += local[6*(tid+4)+i];
    }__syncthreads();
    if(tid < 2)
    {
        #pragma unroll
        for(int i = 0; i < 6; ++i) 
            local[6*tid+i] += local[6*(tid+2)+i];
    }__syncthreads();
    if(tid < 1)
    {
        #pragma unroll 
        for(int i = 0; i < 6; ++i)
        {
            local[i] += local[6+i];
            virial[i] = local[i];
        }
    }
}

void Polymer::ComputeForceVirial(const int& s)
{
    if(s % _verletCell->decompFreq == 0)
        UpdateVerletCellList();
    if(_numPairs > 0)
    {
        if(_modelType == 0)
            ComputeHPSModelVirial();
        else if(_modelType == 1)
            ComputeKHModelVirial(); 
        else if(_modelType == 2)
            ComputeONCModelVirial();
    }
    for(int i = 0; i < _fene.size(); ++i)
        ComputeFENEVirialKernel<<<(_fene[i]->num+NUM_THREADS-1)/NUM_THREADS, NUM_THREADS>>>(_force, _virial, _position, _fene[i]->bond, _fene[i]->num, i);
    for(int i = 0; i < _harmonic.size(); ++i)
        ComputeHarmonicVirialKernel<<<(_harmonic[i]->num+NUM_THREADS-1)/NUM_THREADS, NUM_THREADS>>>(_force, _virial, _position, _harmonic[i]->bond, _harmonic[i]->num, i);
    if(_numGrouped > 0)
        ComputeRigidForceTorqueKernel<<<_numRigid, 64>>>(_forceCOM, _torque, _force, _positionRelative, _orientation, _groupRange, _particleID);
    PairReductionVirialKernel<<<1,1024>>>(_virial);
    cudaDeviceSynchronize();
}

void Polymer::ComputeVirial()
{
    cudaMemset(_virial, 0, sizeof(float)*65536*6);
    if(_numPairs > 0)
    {
        if(_modelType == 0)
            ComputeHPSModelOnlyVirial();
        else if(_modelType == 1)
            ComputeKHModelOnlyVirial(); 
        else if(_modelType == 2)
            ComputeONCModelOnlyVirial();
    }
    for(int i = 0; i < _fene.size(); ++i)
        ComputeFENEOnlyVirialKernel<<<(_fene[i]->num+NUM_THREADS-1)/NUM_THREADS, NUM_THREADS>>>(_virial, _position, _fene[i]->bond, _fene[i]->num, i);
    for(int i = 0; i < _harmonic.size(); ++i)
        ComputeHarmonicOnlyVirialKernel<<<(_harmonic[i]->num+NUM_THREADS-1)/NUM_THREADS, NUM_THREADS>>>(_virial, _position, _harmonic[i]->bond, _harmonic[i]->num, i);
    PairReductionVirialKernel<<<1,1024>>>(_virial);
    float virial[6];
    cudaMemcpy(virial, _virial, sizeof(float)*6, cudaMemcpyDeviceToHost);
    std::cout << "Virials are " << virial[0] << " " << virial[1] << " " << virial[2] << " " << virial[3] << " " << virial[4] << " " << virial[5] << std::endl;
}

__global__ void UpdateBAOABKernel1(Vector3* Restrict pos, Vector3* Restrict momentum, curandStatePhilox4_32_10_t* Restrict state, 
                                  Vector3* Restrict force, int* Restrict type, float kT, float dt, float gamma, int num)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num)
    {
        
        Vector3 r0 = pos[idx];
        Vector3 p0 = momentum[idx];
        Vector3 f0 = force[idx];
        curandStatePhilox4_32_10_t local_state = state[idx];
        float2 tmp1 = curand_normal2(&local_state);
        float tmp2 = curand_normal(&local_state);
        Vector3 rando = Vector3(tmp1.x, tmp1.y, tmp2);

        int t = type[idx];

        float mass = particleMass_const[t];
        float inv_mass = __fdividef(dt, mass);

        p0 = p0 + 0.5f * dt * f0;
        r0 = r0 + 0.5f * inv_mass * p0;
        tmp2 = __expf(-dt*gamma);

        p0 = tmp2*p0 + __fsqrt_rn(mass*kT*(1.f-tmp2*tmp2))*rando;

        r0 = r0 + 0.5f * inv_mass * p0;
        
        pos[idx]      = wrap(r0);
        momentum[idx] = p0;
        state[idx] = local_state;         
    }
}

__global__ void UpdateBAOABKernel2(Vector3* Restrict momentum, Vector3* Restrict force, float dt, int num) 
{
    const int idx  = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num)
    {
        Vector3 f0 = force[idx];
        Vector3 p0 = momentum[idx];
     
        p0 = p0 + 0.5f * dt * f0;
        momentum[idx] = p0;
    }
}

void Polymer::ParticleNVTLangevin(const int& s, const float& dt, const float& gamma)
{
    UpdateBAOABKernel1<<<(_num+127)/128, 128>>>(_position, _momentum, _state, _force, _type, _kT, dt, gamma, _num);
    cudaDeviceSynchronize();
    ComputeForce(s);
    UpdateBAOABKernel2<<<(_num+127)/128, 128>>>(_momentum, _force, dt, _num);
    cudaDeviceSynchronize();
} 

__global__ void GeneralizedUpdateBAOABKernel1(Vector3* Restrict pos, Vector3* Restrict momentum, curandStatePhilox4_32_10_t* Restrict state, 
                                  Vector3* Restrict force, int* Restrict type, int* Restrict noGroupedID, float kT, float dt, float gamma, int num)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num)
    {
        int idx = noGroupedID[tid];

        Vector3 r0 = pos[idx];
        Vector3 p0 = momentum[idx];
        Vector3 f0 = force[idx];
        curandStatePhilox4_32_10_t local_state = state[tid];
        float2 tmp1 = curand_normal2(&local_state);
        float tmp2 = curand_normal(&local_state);
        Vector3 rando = Vector3(tmp1.x, tmp1.y, tmp2);

        int t = type[idx];

        float mass = particleMass_const[t];
        float inv_mass = __fdividef(dt, mass);

        p0 = p0 + 0.5f * dt * f0;
        r0 = r0 + 0.5f * inv_mass * p0;
        tmp2 = __expf(-dt*gamma);

        p0 = tmp2*p0 + __fsqrt_rn(mass*kT*(1.f-tmp2*tmp2))*rando;

        r0 = r0 + 0.5f * inv_mass * p0;
        
        pos[idx]      = wrap(r0);
        momentum[idx] = p0;
        state[tid] = local_state;
    }
}

DEVICE Matrix3 Rx(const float& t) 
{
    /*float qt = 0.25f*t*t;  // for approximate calculations of sin(t) and cos(t)
    float cos = (1.f-qt)/(1.f+qt);
    float sin = t/(1.f+qt);
    */
    float sin, cos;
    __sincosf(t, &sin, &cos);    
    return Matrix3(
    1.0f, 0.0f, 0.0f,
    0.0f,  cos, -sin,
    0.0f,  sin,  cos);
}

DEVICE Matrix3 Ry(const float& t) 
{
    /*float qt = 0.25f*t*t;  // for approximate calculations of sin(t) and cos(t)
    float cos = (1.f-qt)/(1.f+qt);
    float sin = t/(1.f+qt);
    */
    float sin, cos;
    __sincosf(t, &sin, &cos);    
    return Matrix3(
    cos,  0.0f,  sin,
    0.0f, 1.0f, 0.0f,
    -sin, 0.0f,  cos);
}

DEVICE Matrix3 Rz(const float& t) 
{
    /*float qt = 0.25f*t*t;  // for approximate calculations of sin(t) and cos(t)
    float cos = (1.f-qt)/(1.f+qt);
    float sin = t/(1.f+qt);
    */
    float sin, cos;
    __sincosf(t, &sin, &cos);        
    return Matrix3(
    cos,  -sin, 0.0f,
    sin,   cos, 0.0f,
    0.0f, 0.0f, 1.0f);
}

DEVICE void applyRotation(Matrix3& orientation, Vector3& angularMomentum, const Matrix3& R) 
{
    angularMomentum = R * angularMomentum;
    orientation = orientation * R;
    orientation = orientation.normalized();
}

__global__ void GeneralizedUpdateBAOABKernel2(Vector3* Restrict positionCOM, Vector3* Restrict momentumCOM, Vector3* Restrict forceCOM, Matrix3* Restrict orientation, 
Vector3* Restrict angularMomentum, Vector3* Restrict torque, float4* inertia, curandStatePhilox4_32_10_t* Restrict state, float dt, float kT, float gamma, int numNoGrouped, int numRigid)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < numRigid)
    {
    
        Vector3 p = momentumCOM[idx];
        Vector3 f = forceCOM[idx];
        Vector3 r = positionCOM[idx];
        float4  I = inertia[idx];

        Vector3 L   = angularMomentum[idx];
        Vector3 tau = torque[idx];
        Matrix3 R = orientation[idx];

        curandStatePhilox4_32_10_t local_state = state[2*idx+numNoGrouped];
        float2 tmp1 = curand_normal2(&local_state);
        float  tmp2 = curand_normal(&local_state);
        Vector3 rando1 = Vector3(tmp1.x,tmp1.y,tmp2);
        state[2*idx+numNoGrouped] = local_state;

        local_state = state[2*idx+numNoGrouped+1];
        tmp1 = curand_normal2(&local_state);
        tmp2 = curand_normal(&local_state);
        Vector3 rando2 = Vector3(tmp1.x, tmp1.y, tmp2);
        state[2*idx+numNoGrouped+1] = local_state;

        p += 0.5f * dt * f;
        r += 0.5f * dt * p / I.w; // update CoM a half timestep
        L += 0.5f * dt * tau;

        // update orientations a half timestep
        Matrix3 O; // represents a rotation about a principle axis
        O = Rx(0.25f * dt * __fdividef(L.x, I.x)); // R1
        applyRotation(R, L, O);

        O = Ry(0.25f * dt * __fdividef(L.y, I.y)); // R2
        applyRotation(R, L, O);
                        
        O = Rz(0.50f * dt * __fdividef(L.z, I.z)); // R3
        applyRotation(R, L, O);
                        
        O = Ry(0.25f * dt * __fdividef(L.y, I.y)); // R4
        applyRotation(R, L, O);

        O = Rx(0.25f * dt * __fdividef(L.x, I.x)); // R5
        applyRotation(R, L, O);             
        //R = R.normalized();
        tmp2 = __expf(-dt * gamma); 
        //p = R.transpose()*p;
        p = tmp2 * p + __fsqrt_rn(I.w * kT * (1.f-tmp2*tmp2)) * rando1;
        //p = R * p; 
        
        L.x = tmp2 * L.x + __fsqrt_rn(I.x * kT * (1.f-tmp2*tmp2)) * rando2.x;
        L.y = tmp2 * L.y + __fsqrt_rn(I.y * kT * (1.f-tmp2*tmp2)) * rando2.y;
        L.z = tmp2 * L.z + __fsqrt_rn(I.z * kT * (1.f-tmp2*tmp2)) * rando2.z;

        r += 0.5f * dt * p / I.w;         // update CoM a full timestep

        O = Rx(0.25f * dt * __fdividef(L.x, I.x)); // R1
        applyRotation(R, L, O);

        O = Ry(0.25f * dt * __fdividef(L.y, I.y)); // R2
        applyRotation(R, L, O);

        O = Rz(0.50f * dt * __fdividef(L.z, I.z)); // R3
        applyRotation(R, L, O);

        O = Ry(0.25f * dt * __fdividef(L.y, I.y)); // R4
        applyRotation(R, L, O);

        O = Rx(0.25f * dt * __fdividef(L.x, I.x)); // R5
        applyRotation(R, L, O);
        
        //write back to memory
        positionCOM[idx] = wrap(r);
        momentumCOM[idx] = p;
        orientation[idx] = R;
        angularMomentum[idx] = L;
    }
}

__global__ void GeneralizedUpdateBAOABKernel3(Vector3* Restrict momentum, Vector3* Restrict force, int* Restrict noGroupedID, float dt, int num) 
{
    const int tid  = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num)
    {
        int idx = noGroupedID[tid];
        
        Vector3 f0 = force[idx];
        Vector3 p0 = momentum[idx];
     
        p0 = p0 + 0.5f * dt * f0;
        momentum[idx] = p0;
    }
}

__global__ void GeneralizedUpdateBAOABKernel4(Vector3* momentumCOM, Vector3* forceCOM, Vector3* angularMomentum, Vector3* torque, float dt, int numRigid)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < numRigid)
    {
        Vector3 p = momentumCOM[idx];
        Vector3 f = forceCOM[idx];

        Vector3 L   = angularMomentum[idx];
        Vector3 tau = torque[idx];
        
        p += 0.5f * dt * f;
        L += 0.5f * dt * tau;

        //write back to memory
        momentumCOM[idx] = p;
        angularMomentum[idx] = L;
    }
}

void Polymer::GeneralizedParticleNVTLangevin(const int& s, const float& dt, const float& gamma)
{
    int num = _num-_numGrouped;
    if(num > 0)
        GeneralizedUpdateBAOABKernel1<<<(num+127)/128, 128>>>(_position, _momentum, _state, _force, _type, _noGroupedID, _kT, dt, gamma, num);
    GeneralizedUpdateBAOABKernel2<<<(_numRigid+63)/64, 64>>>(_positionCOM, _momentumCOM, _forceCOM, _orientation, _angularMomentum, _torque, _inertia, _state, dt, _kT, gamma, num, _numRigid);
    RedistributePosition();
    ComputeForce(s);
    if(num > 0)
        GeneralizedUpdateBAOABKernel3<<<(num+127)/128, 128>>>(_momentum, _force, _noGroupedID, dt, num);
    GeneralizedUpdateBAOABKernel4<<<(_numRigid+63)/64, 64>>>(_momentumCOM, _forceCOM, _angularMomentum, _torque, dt, _numRigid);
    //cudaDeviceSynchronize();
    RedistributeMomentum();
}

void Polymer::SetUpWriters(const float& dt, const int& outputFreq, const std::string& prefix)
{
    Matrix3 box(_systemBox.L.x, _systemBox.L.y, _systemBox.L.z);
    TrajectoryWriter *w1 = new TrajectoryWriter(prefix.c_str(), TrajectoryWriter::getFormatName(0), box, _num, dt, outputFreq);
    std::string momentum_prefix = prefix;
    momentum_prefix = momentum_prefix + "."+"momentum"; 
    TrajectoryWriter *w2 = new TrajectoryWriter(momentum_prefix.c_str(), TrajectoryWriter::getFormatName(0), box, _num, dt, outputFreq);
    std::string force_prefix = prefix;
    force_prefix = force_prefix + "." + "force";
    TrajectoryWriter *w3 = new TrajectoryWriter(force_prefix.c_str(), TrajectoryWriter::getFormatName(0), box, _num, dt, outputFreq);
    _coordinate_writer = w1;
    _momentum_writer = w2;
    _force_writer = w3; 
}

void Polymer::WriteStep(const int& compute_virial)
{
    cudaMemcpy(_buffer, _position, sizeof(Vector3)*_num, cudaMemcpyDeviceToHost);
    _coordinate_writer->appendDcd(_buffer);
    //cudaMemcpy(_buffer, _momentum, sizeof(Vector3)*_num, cudaMemcpyDeviceToHost);
    //_momentum_writer->appendDcd(_buffer);
    //cudaMemcpy(_buffer, _force, sizeof(Vector3)*_num, cudaMemcpyDeviceToHost);
    //_force_writer->appendDcd(_buffer);
    ComputeTotalEnergy();
    if(compute_virial)
        ComputeVirial();
}

void Polymer::NVTLangevinStep(const int& s, const float& dt, const float& gamma)
{
    if(_numGrouped == 0)
        ParticleNVTLangevin(s, dt, gamma);
    else
        GeneralizedParticleNVTLangevin(s, dt, gamma);
}

//#ifdef ONC
__global__ void ComputeONCModelKernel(Vector3* Restrict force, Vector3* Restrict repulsiveForce, Vector3* Restrict position, int* Restrict type, int2* Restrict Pairs, int* Restrict numPairs, int numTypes)
{
    int tid = threadIdx.x + 64 * blockIdx.x;
    int num = numPairs[0];
    for(int i = tid; i < num; i += 64 * gridDim.x)
    {
        int2 p = Pairs[i];
        int t1 = type[p.x];
        int t2 = type[p.y];
        float3 param1 = particleAttribute_const[t1];
        float3 param2 = particleAttribute_const[t2];
        Vector3 pos = position[p.x];
        pos = wrapVecDiff(pos, position[p.y]);
        float r2 = pos.length2();
        param1.x = param1.x*param1.x;
        float eps = particleNBStrength_const[t1+numTypes*t2];
        if(r2 <= param1.x)
        {
            float trunc = truncateDistance_const[0];
            if(r2 < param1.x*trunc*trunc)
            {
                r2 = param1.x*trunc*trunc;
                pos = pos / pos.length();
                pos = trunc*particleAttribute_const[t1].x*pos;    
            }
            param1.x = __fdividef(param1.x, r2);
            param1.x = 8.f*param1.x*param1.x*param1.x*(param1.y*param1.x-eps);
            Vector3 f = (__fdividef(param1.x,r2)) * pos;
            atomicAdd(&repulsiveForce[p.x], f);
            atomicAdd(&repulsiveForce[p.y],-f);
        }
        else
        {
            param1.x = __fdividef(param1.x, r2);
            Vector3 f = (8.f*(param1.y-eps)*__fdividef(param1.x*param1.x*param1.x*param1.x,r2)) * pos;
            atomicAdd(&force[p.x], f);
            atomicAdd(&force[p.y],-f);
        }
        //charge
        if(param1.z != 0.f and param2.z != 0.f)
        {
            float& r = param2.y;
            r = __fsqrt_rn(r2);
            float& exp_r_z = param2.x;

            float r_z;
            float r_k;
            r_z = __fdividef(r, parameterZ_const[0]);
            r_k = __fdividef(r, Kappa_const[0]);

            exp_r_z = __expf(r_z);
            float& Sz    = param1.x;
            float& dSzdr = param1.y;

            Sz    = 1.f - r_z*r_z*(__fdividef(exp_r_z, (exp_r_z-1.f)*(exp_r_z-1.f)));
            dSzdr = __fdividef(exp_r_z*r_z*r_z*(r_z+exp_r_z*r_z+2.f-2.f*exp_r_z), (exp_r_z-1.f)*(exp_r_z-1.f)*(exp_r_z-1.f));
            
            eps = __fdividef(param1.z*param2.z*__expf(-r_k), 4.f*Pi*Dielectric_const[0]*r2*r*Sz);
            eps = eps * ((1.f+r_k) + __fdividef(dSzdr, Sz));
            pos = eps*pos;
            atomicAdd(&force[p.x], pos);
            atomicAdd(&force[p.y],-pos);
        }
    }
   
}

__global__ void ComputeONCModelEnergyKernel(float* Restrict energy, Vector3* Restrict position, int* Restrict type, int2* Restrict Pairs, 
int* Restrict numPairs, int numTypes)
{
    __shared__ float local[64];
    int tid = threadIdx.x + 64 * blockIdx.x;
    int num = numPairs[0];
    local[threadIdx.x] = 0.f;
    for(int i = tid; i < num; i += 64 * gridDim.x)
    {
        int2 p = Pairs[i];
        int t1 = type[p.x];
        int t2 = type[p.y];
        float3 param1 = particleAttribute_const[t1];
        float3 param2 = particleAttribute_const[t2];
        Vector3 pos = position[p.x];
        pos = wrapVecDiff(pos, position[p.y]);
        float r2 = pos.length2();
        param1.x = param1.x*param1.x;
        float eps = particleNBStrength_const[t1+numTypes*t2];
        if(r2 <= param1.x)
        {
            param1.x = __fdividef(param1.x, r2);
            param1.x = param1.x*param1.x*param1.x*(param1.y*param1.x-1.33333333f*eps)+0.3333333333f*eps;
            local[threadIdx.x] += param1.x;
        }
        else
        {
            param1.x = __fdividef(param1.x, r2);
            local[threadIdx.x] += (param1.y-eps)*param1.x*param1.x*param1.x*param1.x;
        }
        //charge
        if(param1.z != 0.f and param2.z != 0.f)
        {
            float Kappa = Kappa_const[0];
            float z = parameterZ_const[0];
            r2 = __fsqrt_rn(r2);
            Kappa = __fdividef(r2, Kappa);
            z = __fdividef(r2, z);
            float tmp2 = __expf(z);
            eps = __fdividef(__expf(-Kappa)*param1.z*param2.z, 4.f*Pi*Dielectric_const[0]*r2*(1.f-z*z*__fdividef(tmp2, (tmp2-1.f)*(tmp2-1.f))));
            local[threadIdx.x] += eps;
        }
    }
    __syncthreads();
    //if(threadIdx.x < 64)
    //    local[threadIdx.x] += local[threadIdx.x+64];
    //__syncthreads();
    if(threadIdx.x < 32)
        local[threadIdx.x] += local[threadIdx.x+32];
    __syncthreads();
    if(threadIdx.x < 16)
        local[threadIdx.x] += local[threadIdx.x+16];
    __syncthreads();
    if(threadIdx.x < 8)
        local[threadIdx.x] += local[threadIdx.x+8];
    __syncthreads();
    if(threadIdx.x < 4)
        local[threadIdx.x] += local[threadIdx.x+4];
    __syncthreads();
    if(threadIdx.x < 2)
        local[threadIdx.x] += local[threadIdx.x+2];
    __syncthreads();
    if(threadIdx.x < 1)
    {
        local[0] += local[1];
        energy[blockIdx.x] += local[0];
    }     
}

__global__ void ComputeONCModelVirialKernel(Vector3* Restrict force, Vector3* Restrict repulsiveForce, float* Restrict virial, Vector3* Restrict position, int* Restrict type, int2* Restrict Pairs, int* Restrict numPairs, int numTypes)
{
    int tid = threadIdx.x + 64 * blockIdx.x;
    int num = numPairs[0];
    __shared__ float local[64*6];
    for(int i = 0; i < 6; ++i)
        local[6*threadIdx.x+i] = 0.f;
    for(int i = tid; i < num; i += 64 * gridDim.x)
    {
        int2 p = Pairs[i];
        int t1 = type[p.x];
        int t2 = type[p.y];
        float3 param1 = particleAttribute_const[t1];
        float3 param2 = particleAttribute_const[t2];
        Vector3 pos = position[p.x];
        pos = wrapVecDiff(pos, position[p.y]);
        float r2 = pos.length2();
        param1.x = param1.x*param1.x;
        float eps = particleNBStrength_const[t1+numTypes*t2];
        if(r2 <= param1.x)
        {
            float trunc = truncateDistance_const[0];
            if(r2 < param1.x*trunc*trunc)
            {
                r2 = param1.x*trunc*trunc;
                pos = pos / pos.length();
                pos = trunc*particleAttribute_const[t1].x*pos;    
            }
            param1.x = __fdividef(param1.x, r2);
            param1.x = 8.f*param1.x*param1.x*param1.x*(param1.y*param1.x-eps);
            Vector3 f = (__fdividef(param1.x,r2)) * pos;
            atomicAdd(&repulsiveForce[p.x], f);
            atomicAdd(&repulsiveForce[p.y],-f);
            local[6*threadIdx.x+0] += f.x*pos.x;
            local[6*threadIdx.x+1] += f.x*pos.y;
            local[6*threadIdx.x+2] += f.x*pos.z;
            local[6*threadIdx.x+3] += f.y*pos.y;
            local[6*threadIdx.x+4] += f.y*pos.z;
            local[6*threadIdx.x+5] += f.z*pos.z;
            
        }
        else
        {
            param1.x = __fdividef(param1.x, r2);
            Vector3 f = (8.f*(param1.y-eps)*__fdividef(param1.x*param1.x*param1.x*param1.x,r2)) * pos;
            atomicAdd(&force[p.x], f);
            atomicAdd(&force[p.y],-f);
            local[6*threadIdx.x+0] += f.x*pos.x;
            local[6*threadIdx.x+1] += f.x*pos.y;
            local[6*threadIdx.x+2] += f.x*pos.z;
            local[6*threadIdx.x+3] += f.y*pos.y;
            local[6*threadIdx.x+4] += f.y*pos.z;
            local[6*threadIdx.x+5] += f.z*pos.z;
        }
        //charge
        if(param1.z != 0.f and param2.z != 0.f)
        {
            float& r = param2.y;
            r = __fsqrt_rn(r2);
            float& exp_r_z = param2.x;

            float r_z;
            float r_k;
            r_z = __fdividef(r, parameterZ_const[0]);
            r_k = __fdividef(r, Kappa_const[0]);

            exp_r_z = __expf(r_z);
            float& Sz    = param1.x;
            float& dSzdr = param1.y;

            Sz    = 1.f - r_z*r_z*(__fdividef(exp_r_z, (exp_r_z-1.f)*(exp_r_z-1.f)));
            dSzdr = __fdividef(exp_r_z*r_z*r_z*(r_z+exp_r_z*r_z+2.f-2.f*exp_r_z), (exp_r_z-1.f)*(exp_r_z-1.f)*(exp_r_z-1.f));

            eps = __fdividef(param1.z*param2.z*__expf(-r_k), 4.f*Pi*Dielectric_const[0]*r2*r*Sz);
            eps = eps * ((1.f+r_k) + __fdividef(dSzdr, Sz));

            Vector3 f = eps*pos;

            atomicAdd(&force[p.x], f);
            atomicAdd(&force[p.y],-f);
            local[6*threadIdx.x+0] += f.x*pos.x;
            local[6*threadIdx.x+1] += f.x*pos.y;
            local[6*threadIdx.x+2] += f.x*pos.z;
            local[6*threadIdx.x+3] += f.y*pos.y;
            local[6*threadIdx.x+4] += f.y*pos.z;
            local[6*threadIdx.x+5] += f.z*pos.z;
        }
    }
    __syncthreads();
    //partially parallel reduction
    if(threadIdx.x < 32)
    {
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+32)+i];
    }__syncthreads();
    if(threadIdx.x < 16)
    {
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+16)+i];
    }__syncthreads();
    if(threadIdx.x < 8)
    {
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+8)+i];
    }__syncthreads();
    if(threadIdx.x < 4)
    {
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+4)+i];
    }__syncthreads();
    if(threadIdx.x < 2)
    {
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+2)+i];
    }__syncthreads();
    if(threadIdx.x < 1)
    {
        for(int i = 0; i < 6; ++i)
        {
            local[i] += local[6+i];
            virial[6*blockIdx.x+i]+=local[i];
        }
    }
}

__global__ void ComputeONCModelOnlyVirialKernel(float* Restrict virial, Vector3* Restrict position, int* Restrict type, int2* Restrict Pairs, int* Restrict numPairs, int numTypes)
{
    int tid = threadIdx.x + 64 * blockIdx.x;
    int num = numPairs[0];
    __shared__ float local[64*6];
    for(int i = 0; i < 6; ++i)
        local[6*threadIdx.x+i] = 0.f;
    for(int i = tid; i < num; i += 64 * gridDim.x)
    {
        int2 p = Pairs[i];
        int t1 = type[p.x];
        int t2 = type[p.y];
        float3 param1 = particleAttribute_const[t1];
        float3 param2 = particleAttribute_const[t2];
        Vector3 pos = position[p.x];
        pos = wrapVecDiff(pos, position[p.y]);
        float r2 = pos.length2();
        param1.x = param1.x*param1.x;
        float eps = particleNBStrength_const[t1+numTypes*t2];
        if(r2 <= param1.x)
        {
            float trunc = truncateDistance_const[0];
            if(r2 < param1.x*trunc*trunc)
            {
                r2 = param1.x*trunc*trunc;
                pos = pos / pos.length();
                pos = trunc*particleAttribute_const[t1].x*pos;    
            }
            param1.x = __fdividef(param1.x, r2);
            param1.x = 8.f*param1.x*param1.x*param1.x*(param1.y*param1.x-eps);
            Vector3 f = (__fdividef(param1.x,r2)) * pos;
            local[6*threadIdx.x+0] += f.x*pos.x;
            local[6*threadIdx.x+1] += f.x*pos.y;
            local[6*threadIdx.x+2] += f.x*pos.z;
            local[6*threadIdx.x+3] += f.y*pos.y;
            local[6*threadIdx.x+4] += f.y*pos.z;
            local[6*threadIdx.x+5] += f.z*pos.z;
            
        }
        else
        {
            param1.x = __fdividef(param1.x, r2);
            Vector3 f = (8.f*(param1.y-eps)*__fdividef(param1.x*param1.x*param1.x*param1.x,r2)) * pos;
            local[6*threadIdx.x+0] += f.x*pos.x;
            local[6*threadIdx.x+1] += f.x*pos.y;
            local[6*threadIdx.x+2] += f.x*pos.z;
            local[6*threadIdx.x+3] += f.y*pos.y;
            local[6*threadIdx.x+4] += f.y*pos.z;
            local[6*threadIdx.x+5] += f.z*pos.z;
        }
        //charge
        if(param1.z != 0.f and param2.z != 0.f)
        {
            float& r = param2.y;
            r = __fsqrt_rn(r2);
            float& exp_r_z = param2.x;

            float r_z;
            float r_k;
            r_z = __fdividef(r, parameterZ_const[0]);
            r_k = __fdividef(r, Kappa_const[0]);

            exp_r_z = __expf(r_z);
            float& Sz    = param1.x;
            float& dSzdr = param1.y;

            Sz    = 1.f - r_z*r_z*(__fdividef(exp_r_z, (exp_r_z-1.f)*(exp_r_z-1.f)));
            dSzdr = __fdividef(exp_r_z*r_z*r_z*(r_z+exp_r_z*r_z+2.f-2.f*exp_r_z), (exp_r_z-1.f)*(exp_r_z-1.f)*(exp_r_z-1.f));

            eps = __fdividef(param1.z*param2.z*__expf(-r_k), 4.f*Pi*Dielectric_const[0]*r2*r*Sz);
            eps = eps * ((1.f+r_k) + __fdividef(dSzdr, Sz));
            Vector3 f = eps*pos;

            local[6*threadIdx.x+0] += f.x*pos.x;
            local[6*threadIdx.x+1] += f.x*pos.y;
            local[6*threadIdx.x+2] += f.x*pos.z;
            local[6*threadIdx.x+3] += f.y*pos.y;
            local[6*threadIdx.x+4] += f.y*pos.z;
            local[6*threadIdx.x+5] += f.z*pos.z;
        }
    }
    __syncthreads();
    //partially parallel reduction
    if(threadIdx.x < 32)
    {
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+32)+i];
    }__syncthreads();
    if(threadIdx.x < 16)
    {
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+16)+i];
    }__syncthreads();
    if(threadIdx.x < 8)
    {
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+8)+i];
    }__syncthreads();
    if(threadIdx.x < 4)
    {
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+4)+i];
    }__syncthreads();
    if(threadIdx.x < 2)
    {
        for(int i = 0; i < 6; ++i)
            local[6*threadIdx.x+i] += local[6*(threadIdx.x+2)+i];
    }__syncthreads();
    if(threadIdx.x < 1)
    {
        for(int i = 0; i < 6; ++i)
        {
            local[i] += local[6+i];
            virial[6*blockIdx.x+i]+=local[i];
        }
    }
}

void Polymer::ComputeONCModel()
{
    int numPairs = ((_numPairs+63)/64 > 65535) ? 65535 : (_numPairs+63)/64;
    ComputeONCModelKernel<<<numPairs, 64>>>(_force, _repulsiveForce, _position, _type, _verletCell->pairs, _verletCell->numPairs, _numTypes);
    ComputeTotalNonBondedForce<<<(_num+255)/256, 256>>>(_force, _repulsiveForce, _num);
    cudaDeviceSynchronize();
}

void Polymer::ComputeONCModelEnergy()
{
    int numPairs = ((_numPairs+63)/64 > 65535) ? 65535 : (_numPairs+63)/64;
    ComputeONCModelEnergyKernel<<<numPairs, 64>>>(_energy, _position, _type, _verletCell->pairs, _verletCell->numPairs, _numTypes);
    cudaDeviceSynchronize();
}

void Polymer::ComputeONCModelVirial()
{
    int numPairs = ((_numPairs+63)/64 > 65535) ? 65535 : (_numPairs+63)/64;
    ComputeONCModelVirialKernel<<<numPairs, 64>>>(_force, _repulsiveForce, _virial, _position, _type, _verletCell->pairs, _verletCell->numPairs, _numTypes);
    ComputeTotalNonBondedForce<<<(_num+255)/256, 256>>>(_force, _repulsiveForce, _num);
    cudaDeviceSynchronize();
}

void Polymer::ComputeONCModelOnlyVirial()
{
    int numPairs = ((_numPairs+63)/64 > 65535) ? 65535 : (_numPairs+63)/64;
    ComputeONCModelOnlyVirialKernel<<<numPairs, 64>>>(_virial, _position, _type, _verletCell->pairs, _verletCell->numPairs, _numTypes);
    cudaDeviceSynchronize();
}

//#endif

/*
 * Compute bend and torsion force field
 * cuda kernel function
 *
 */

__global__ void ComputeForceTabulatedAngleKernel(Vector3* Restrict force, Vector3* Restrict position, int4* Restrict angleList, int numAngle, float** Restrict x, float** Restrict y, int* Restrict tableLength)
{
    for(int i = threadIdx.x+64*blockIdx.x; i < numAngle; i += 64*gridDim.x)
    {
        int4 angle_list = angleList[i];

        Vector3 f      = position[angle_list.z];
        Vector3 pos_ba = wrapVecDiff(position[angle_list.y], f); //ba
        Vector3 pos_bc = wrapVecDiff(position[angle_list.w], f); //bc
        f = wrapVecDiff(pos_bc,pos_ba); //ac

        float l0 = pos_ba.length();
        float l1 = pos_bc.length();
        float theta = __fdividef(l0*l0+l1*l1-f.length2(), l0*l1*2.f);
        theta =  __saturatef((theta+1.f)*0.5f)*2.f-1.f;
        theta = acosf(theta);

        float x0 = x[angle_list.x][0];
        float x1 = x[angle_list.x][1];

        int home = floorf(__fdividef(theta-x0,(x1-x0)));
        int _max = tableLength[angle_list.x];
        if(home+1 >= _max)
            home = _max - 2;
        float& dydx = theta;
        dydx = __fdividef((y[angle_list.x][home+1]-y[angle_list.x][home]), x1-x0);
        f = pos_ba.cross((pos_ba.cross(pos_bc)));
        f = -dydx*f/(l0*f.length());
        atomicAdd(&force[angle_list.y], f);

        Vector3 fc = pos_bc.cross((pos_ba.cross(pos_bc)));
        fc = dydx*fc/(l1*fc.length());

        atomicAdd(&force[angle_list.w], fc); 
        atomicAdd(&force[angle_list.z], -(fc+f));
    }
}

__global__ void ComputeEnergyTabulatedAngleKernel(float* Restrict energy, Vector3* Restrict position, int4* Restrict angleList, int numAngle, float** Restrict x, float** Restrict y, int* Restrict tableLength)
{
    __shared__ float local[64];
    local[threadIdx.x] = 0.f;
    for(int i = threadIdx.x+64*blockIdx.x; i < numAngle; i += 64*gridDim.x)
    {
        int4 angle_list = angleList[i];

        Vector3 f      = position[angle_list.z];
        Vector3 pos_ba = wrapVecDiff(position[angle_list.y], f); //ba 
        Vector3 pos_bc = wrapVecDiff(position[angle_list.w], f); //bc
        f = wrapVecDiff(pos_bc,pos_ba); //ac

        float l0 = pos_ba.length();
        float l1 = pos_bc.length();
        float theta = __fdividef(l0*l0+l1*l1-f.length2(), l0*l1*2.f);
        theta =  __saturatef((theta+1.f)*0.5f)*2.f-1.f;
        theta = acosf(theta);
 
        float x0 = x[angle_list.x][0];
        float x1 = x[angle_list.x][1];

        int home = floorf(__fdividef(theta-x0,(x1-x0)));
        int _max = tableLength[angle_list.x];
        if(home+1 >= _max)
            home = _max - 2;

        x0 = theta - x[angle_list.x][home];
        x1 = x[angle_list.x][home+1] - theta;

        float& e = theta;
        e = __fdividef((y[angle_list.x][home+1]*x0+y[angle_list.x][home]*x1), x1+x0);
        local[threadIdx.x] += e;
    }
    __syncthreads();
    if(threadIdx.x < 32)
        local[threadIdx.x] += local[threadIdx.x+32];
    __syncthreads();
    if(threadIdx.x < 16)
        local[threadIdx.x] += local[threadIdx.x+16];
    __syncthreads();
    if(threadIdx.x < 8)
        local[threadIdx.x] += local[threadIdx.x+8];
    __syncthreads();
    if(threadIdx.x < 4)
        local[threadIdx.x] += local[threadIdx.x+4];
    __syncthreads();
    if(threadIdx.x < 2)
        local[threadIdx.x] += local[threadIdx.x+2];
    __syncthreads();
    if(threadIdx.x < 1)
    {
        local[0] += local[1];
        energy[blockIdx.x] += local[0];
    }  
}

__global__ void ComputeForceDihedralONCKernel(Vector3* Restrict force, Vector3* Restrict position, int4* Restrict dihedralList, int* Restrict type, int numDihedral, float* Restrict coefficient)
{
    float sin_theta[6];
    float cos_theta[6];
    float coeff[13];
    for(int i = threadIdx.x+64*blockIdx.x; i < numDihedral; i += 64*gridDim.x)
    {
        int t   = type[i];
        #pragma unroll
        for(int j = 0; j < 13; ++j)
            coeff[j] = coefficient[13*t+j];

        int4 id = dihedralList[i];
        Vector3 pos_a = position[id.x];
        Vector3 pos_b = position[id.y];
        Vector3 pos_c = position[id.z];
        Vector3 pos_d = position[id.w];

        Vector3 pos_ab = wrapVecDiff(pos_b,pos_a);
        Vector3 pos_bc = wrapVecDiff(pos_c,pos_b);
        Vector3 pos_cd = wrapVecDiff(pos_d,pos_c);
        
        float lbc = pos_bc.length();
        Vector3 a = pos_ab.cross(pos_bc);
        Vector3 b = pos_bc.cross(pos_cd);
        float theta = atan2f(__fdividef((a.cross(b)).dot(pos_bc), lbc),a.dot(b));
        __sincosf(theta, &sin_theta[0], &cos_theta[0]);

        #pragma unroll
        for(int j = 1; j < 6; ++j)
        {
            sin_theta[j] = sin_theta[0]*cos_theta[j-1]+cos_theta[0]*sin_theta[j-1];
            cos_theta[j] = cos_theta[0]*cos_theta[j-1]-sin_theta[0]*sin_theta[j-1];
        }

        float& dydx = theta;
        dydx = 0.f;

        #pragma unroll
        for(int j = 0; j < 6; ++j)
            dydx += (j+1.)*(coeff[2*j]*cos_theta[j] - coeff[2*j+1]*sin_theta[j]);

        //compute fa
        float lab = pos_ab.length();
        float lac2 = wrapDiff2(pos_c,pos_a);
        float theta1 = __fdividef((lab*lab+lbc*lbc-lac2), 2.f*lab*lbc);
        theta1 =  __saturatef(theta1*theta1);

        Vector3 fa = pos_ab.cross(pos_bc);
        fa = __fdividef(dydx, lab*__fsqrt_rn(1.f-theta1)*fa.length())*fa;

        //compute fd
        lab = pos_cd.length(); //cd
        lac2 =  wrapDiff2(pos_d,pos_b); //bd
        theta1 = __fdividef((lbc*lbc+lab*lab-lac2), 2.f*lbc*lab);
        theta1 =  __saturatef(theta1*theta1);

        Vector3 fd = pos_cd.cross(pos_bc);
        fd = __fdividef(dydx, lab*__fsqrt_rn(1.f-theta1)*fd.length())*fd;

        //compute fc
        Vector3& pos_oc = pos_a;
        pos_oc = wrapVecDiff(pos_c, 0.5f*(pos_b+pos_c));
        Vector3& tc = pos_b;
        tc = -1.f*(pos_oc.cross(fd)+0.5f*pos_cd.cross(fd)+0.5f*fa.cross(pos_ab));
        Vector3& fc = pos_c;
        fc = __fdividef(1.f, pos_oc.length2())*tc.cross(pos_oc);
        atomicAdd(&force[id.x],fa);
        atomicAdd(&force[id.y],-(fa+fd+fc));
        atomicAdd(&force[id.z],fc);
        atomicAdd(&force[id.w],fd);        
    }
}

__global__ void ComputeEnergyDihedralONCKernel(float* Restrict energy, Vector3* Restrict position, int4* Restrict dihedralList, int* Restrict type, int numDihedral, float* Restrict coefficient)
{
    float sin_theta[6];
    float cos_theta[6];
    float coeff[13];
    __shared__ float local[64];
    local[threadIdx.x] = 0.f;
    for(int i = threadIdx.x+64*blockIdx.x; i < numDihedral; i += 64*gridDim.x)
    {
        int t   = type[i];
        #pragma unroll
        for(int j = 0; j < 13; ++j)
            coeff[j] = coefficient[13*t+j];

        int4 id = dihedralList[i];
        Vector3 pos_a = position[id.x];
        Vector3 pos_b = position[id.y];
        Vector3 pos_c = position[id.z];
        Vector3 pos_d = position[id.w];

        Vector3 pos_ab = wrapVecDiff(pos_b,pos_a);
        Vector3 pos_bc = wrapVecDiff(pos_c,pos_b);
        Vector3 pos_cd = wrapVecDiff(pos_d,pos_c);
        
        float lbc = pos_bc.length();
        Vector3 a = pos_ab.cross(pos_bc);
        Vector3 b = pos_bc.cross(pos_cd);
        float theta = atan2f(__fdividef((a.cross(b)).dot(pos_bc), lbc),a.dot(b));
        __sincosf(theta, &sin_theta[0], &cos_theta[0]);

        #pragma unroll
        for(int j = 1; j < 6; ++j)
        {
            sin_theta[j] = sin_theta[0]*cos_theta[j-1]+cos_theta[0]*sin_theta[j-1];
            cos_theta[j] = cos_theta[0]*cos_theta[j-1]-sin_theta[0]*sin_theta[j-1];
        }

        float& dydx = theta;
        dydx = 0.f;

        #pragma unroll
        for(int j = 0; j < 6; ++j)
            dydx += (coeff[2*j]*sin_theta[j] + coeff[2*j+1]*cos_theta[j]);

        dydx += coeff[12];
        local[threadIdx.x] += dydx;
    }
    __syncthreads();
    if(threadIdx.x < 32)
        local[threadIdx.x] += local[threadIdx.x+32];
    __syncthreads();
    if(threadIdx.x < 16)
        local[threadIdx.x] += local[threadIdx.x+16];
    __syncthreads();
    if(threadIdx.x < 8)
        local[threadIdx.x] += local[threadIdx.x+8];
    __syncthreads();
    if(threadIdx.x < 4)
        local[threadIdx.x] += local[threadIdx.x+4];
    __syncthreads();
    if(threadIdx.x < 2)
        local[threadIdx.x] += local[threadIdx.x+2];
    __syncthreads();
    if(threadIdx.x < 1)
    {
        local[0] += local[1];
        energy[blockIdx.x] += local[0];
    }  
}

__global__ void ComputeForceBendingTorsionONCKernel(Vector3* Restrict force, Vector3* Restrict position, int2* Restrict bond_list, int num, float3* params)
{
    __shared__ float3 param[1];
    if(!threadIdx.x)
        param[0] = *params;
    __syncthreads();
    for(int i = threadIdx.x + 128*blockIdx.x; i < num; i += 128*gridDim.x)
    {
        int2 id = bond_list[i];
        Vector3 r = wrapVecDiff(position[id.y], position[id.x]);
        float dr  = r.length();
        float b = param[0].x;
        float k = __fdividef(2.f*param[0].y,b);
        b = __fdividef(dr,b);
        r = __fdividef(k*(b-1.32f)*(b-1.32f)*(b-1.32f)*(b-2.63f)*(3.f*b-6.58f), dr)*r;
        atomicAdd(&force[id.x], r);
        atomicAdd(&force[id.y],-r);
    }
}

__global__ void ComputeEnergyBendingTorsionONCKernel(float* Restrict energy, Vector3* Restrict position, int2* Restrict bond_list, int num, float3* params)
{
    __shared__ float3 param[1];
    __shared__ float local[128];
    local[threadIdx.x] = 0.f;
    if(!threadIdx.x)
        param[0] = *params;
    __syncthreads();
    for(int i = threadIdx.x + 128*blockIdx.x; i < num; i += 128*gridDim.x)
    {
        int2 id = bond_list[i];
        float dr = wrapDiff(position[id.y], position[id.x]);
        float3 coeff = param[0];
        //float b = param[0].x;
        //float k = parma[0].y;
        //float c = param[0].z;
        coeff.x = __fdividef(dr,coeff.x);
        coeff.z = coeff.y*(coeff.x-1.32f)*(coeff.x-1.32f)*(coeff.x-1.32f)*(coeff.x-1.32f)*(coeff.x-2.63f)*(coeff.x-2.63f) - coeff.z;
        local[threadIdx.x] += coeff.z;
    }
    __syncthreads();
    if(threadIdx.x < 64)
        local[threadIdx.x] += local[threadIdx.x+64];
    __syncthreads();
    if(threadIdx.x < 32)
        local[threadIdx.x] += local[threadIdx.x+32];
    __syncthreads();
    if(threadIdx.x < 16)
        local[threadIdx.x] += local[threadIdx.x+16];
    __syncthreads();
    if(threadIdx.x < 8)
        local[threadIdx.x] += local[threadIdx.x+8];
    __syncthreads();
    if(threadIdx.x < 4)
        local[threadIdx.x] += local[threadIdx.x+4];
    __syncthreads();
    if(threadIdx.x < 2)
        local[threadIdx.x] += local[threadIdx.x+2];
    __syncthreads();
    if(threadIdx.x < 1)
    {
        local[0] += local[1];
        energy[blockIdx.x] += local[0];
    } 
}

__global__ void AddForceRestraintHarmonicKernel(Vector3* Restrict force, Vector3* Restrict position, float4* Restrict param, int* Restrict index, int num)
{
    for(int i = threadIdx.x+128*blockIdx.x; i < num; i += 128*gridDim.x)
    {
        int id = index[i];
        float4 coeff = param[i];
        Vector3 r0 = Vector3(coeff.y, coeff.z, coeff.w);
        r0 = -coeff.x * wrapVecDiff(position[id], r0);
        force[id] += r0;
    }
}

__global__ void AddEnergyRestraintHarmonicKernel(float* Restrict energy, Vector3* Restrict position, float4* Restrict param, int* Restrict index, int num)
{
    __shared__ float local[128];
    local[threadIdx.x] = 0.f;
    for(int i = threadIdx.x+128*blockIdx.x; i < num; i += 128*gridDim.x)
    {
        int id = index[i];
        float4 coeff = param[i];
        Vector3 r0 = Vector3(coeff.y, coeff.z, coeff.w);
        coeff.x = 0.5f * coeff.x * wrapDiff2(position[id], r0);
        local[threadIdx.x] += coeff.x;
    }
    __syncthreads();
    if(threadIdx.x < 64)
        local[threadIdx.x] += local[threadIdx.x+64];
    __syncthreads();
    if(threadIdx.x < 32)
        local[threadIdx.x] += local[threadIdx.x+32];
    __syncthreads();
    if(threadIdx.x < 16)
        local[threadIdx.x] += local[threadIdx.x+16];
    __syncthreads();
    if(threadIdx.x < 8)
        local[threadIdx.x] += local[threadIdx.x+8];
    __syncthreads();
    if(threadIdx.x < 4)
        local[threadIdx.x] += local[threadIdx.x+4];
    __syncthreads();
    if(threadIdx.x < 2)
        local[threadIdx.x] += local[threadIdx.x+2];
    __syncthreads();
    if(threadIdx.x < 1)
    {
        local[0] += local[1];
        energy[blockIdx.x] += local[0];
    }     
}


