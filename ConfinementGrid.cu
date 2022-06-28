#include "ConfinementGrid.h"
#include <fstream>
#include "Utility.h"
#include "InterpolateGrid.h"
#include "ARBDException.h"
#ifdef Restrict
    #undef Restrict
    #define Restrict __restrict__
#else
    #define Restrict __restrict__
#endif

ConfinementGrid::ConfinementGrid(const std::string& filename)
{
    Grid grid;
    ReadGridDxFormat(grid, filename);
    MemcpyHtoD(grid);
    std::cout << "Complete confinement potential initialization" << std::endl;
}

ConfinementGrid::~ConfinementGrid()
{
    Grid grid;
    cudaMemcpy(&grid, _grid, sizeof(Grid), cudaMemcpyDeviceToHost);
    cudaFree(grid.data);
    cudaFree(_grid);
}

void ConfinementGrid::ReadGridDxFormat(Grid& grid, const std::string& filename)
{
    std::ifstream infile(filename);
    if(!infile.is_open())
        throw ARBD_Exception(FILE_OPEN_ERROR, filename.c_str());

    std::string buff;
    std::vector<std::string> elements;

    Utility::get_line(buff, infile, std::string("object"), std::string("#"));
    Utility::split(elements, buff);

    int3 L = make_int3(Utility::string_to_number<int>(elements[elements.size()-3]),
                       Utility::string_to_number<int>(elements[elements.size()-2]),
                       Utility::string_to_number<int>(elements[elements.size()-1]));
    grid.L = L;
    int size = L.x*L.y*L.z;
    float* data = new float [size];

    Utility::get_line(buff, infile, std::string("origin"), std::string("#"));     
    Utility::split(elements, buff);
    grid.origin = make_float3(Utility::string_to_number<float>(elements[1]),
                              Utility::string_to_number<float>(elements[2]),
                              Utility::string_to_number<float>(elements[3]));

    float a[3];
    for(int i = 0; i < 3; ++i)
    {
        Utility::get_line(buff, infile, std::string("delta"), std::string("#"));
        Utility::split(elements, buff);
        a[i] = Utility::string_to_number<float>(elements[i+1]);
    }
    grid.dx    = make_float3(a[0], a[1], a[2]);
    grid.invdx = make_float3(1./a[0], 1./a[1], 1./a[2]);

    Utility::get_line(buff, infile, std::string("rank"), std::string("#"));
    Utility::split(elements, buff);
    if(elements[7] != std::string("0"))
        throw ARBD_Exception(ILL_FORMATTED_FILE, filename.c_str());

    grid.data = new float[size];
    int count = 0;
    while(Utility::get(buff, infile, std::string("#")))
    {
        Utility::split(elements, buff);
        if(count + elements.size() >size)
            throw ARBD_Exception(ILL_FORMATTED_FILE, filename.c_str());
        //parse here
        for(auto it = elements.begin(); it != elements.end(); ++it)
            grid.data[count++] = Utility::string_to_number<float>(*it);
        if(count >= size)
            break;
    }
    if(count != size)
        throw ARBD_Exception(ILL_FORMATTED_FILE, filename.c_str());
    infile.close();
}

void ConfinementGrid::MemcpyHtoD(Grid& grid)
{
    cudaMalloc((void**)&_grid, sizeof(Grid));

    float* val;
    int count = (grid.L.x*grid.L.y*grid.L.z);
    cudaMalloc((void**)&val, sizeof(float)*count);
    cudaMemcpy(val, grid.data, sizeof(float)*count, cudaMemcpyHostToDevice);
    delete [] grid.data;

    grid.data = val;
    cudaMemcpy(_grid, &grid, sizeof(Grid), cudaMemcpyHostToDevice);
}
#if 0
__device__ Vector3 InterpolateForce1(const Vector3& pos, const float3& origin, const float3& invdx, const int3& L, float* data)
{
    Vector3 l;

    l.x = pos.x - origin.x;
    l.y = pos.y - origin.y;
    l.z = pos.z - origin.z;

    l.x = invdx.x*l.x;
    l.y = invdx.y*l.y;
    l.z = invdx.z*l.z;
    
    // Find the home node.
    const int homeX = floor(l.x);
    const int homeY = floor(l.y);
    const int homeZ = floor(l.z);
    if(homeX < L.x and homeY < L.y and homeZ < L.z and homeX >= 0 and homeY >= 0 and homeZ >= 0)
    {
        const float wx = l.x - homeX;
        const float wy = l.y - homeY;
        const float wz = l.z - homeZ;

        float v[2][2][2];
        #pragma unroll
        for (int iz = 0; iz < 2; iz++)
        {
            int idz = homeZ+iz;
            idz = (idz >= L.z) ? L.z-1 : idz;
            #pragma unroll
            for (int iy = 0; iy < 2; iy++)
            {
                int idy = homeY+iy;
                idy = (idy >= L.y) ? L.y-1 : idy;
                #pragma unroll
                for (int ix = 0; ix < 2; ix++)
                {
                    int idx = homeX+ix;
                    idx = (idx >= L.x) ? L.x-1 : idx;
                    int id = idz + L.z * (idy + L.y * idx);
                    v[ix][iy][iz] = data[id];
                }
            }
        }

        float g3[3][2];
        #pragma unroll
        for (int iz = 0; iz < 2; iz++)
        {
            float g2[2][2];
            #pragma unroll
            for (int iy = 0; iy < 2; iy++)
            {
                g2[0][iy] = (v[1][iy][iz] - v[0][iy][iz]); /* f.x */
                g2[1][iy] = wx * (v[1][iy][iz] - v[0][iy][iz]) + v[0][iy][iz]; /* f.y & f.z */
            }
            // Mix along y.
            g3[0][iz] = wy * (g2[0][1] - g2[0][0]) + g2[0][0];
            g3[1][iz] = (g2[1][1] - g2[1][0]);
            g3[2][iz] = wy * (g2[1][1] - g2[1][0]) + g2[1][0];
        }
        // Mix along z.
        Vector3 f;
        f.x = -(wz * (g3[0][1] - g3[0][0]) + g3[0][0]);
        f.y = -(wz * (g3[1][1] - g3[1][0]) + g3[1][0]);
        f.z =       -(g3[2][1] - g3[2][0]);

        f.x = f.x*invdx.x;
        f.y = f.y*invdx.y;
        f.z = f.z*invdx.z;
        return f;
        //val = wz * (g3[2][1] - g3[2][0]) + g3[2][0];
    }
    else
    {
        //val = 0.f;
        //force = Vector3(0.f);
        return Vector3(0.f);
    }   
}

__device__ float InterpolateEnergy1(const Vector3& pos, const float3& origin, const float3& invdx, const int3& L, float* data)
{
    Vector3 l;

    l.x = pos.x - origin.x;
    l.y = pos.y - origin.y;
    l.z = pos.z - origin.z;

    l.x = invdx.x*l.x;
    l.y = invdx.y*l.y;
    l.z = invdx.z*l.z;
    
    // Find the home node.
    const int homeX = floor(l.x);
    const int homeY = floor(l.y);
    const int homeZ = floor(l.z);
    if(homeX < L.x and homeY < L.y and homeZ < L.z and homeX >= 0 and homeY >= 0 and homeZ >= 0)
    {
        const float wx = l.x - homeX;
        const float wy = l.y - homeY;
        const float wz = l.z - homeZ;

        float v[2][2][2];
        #pragma unroll
        for (int iz = 0; iz < 2; iz++)
        {
            int idz = homeZ+iz;
            idz = (idz >= L.z) ? L.z-1 : idz;
            #pragma unroll
            for (int iy = 0; iy < 2; iy++)
            {
                int idy = homeY+iy;
                idy = (idy >= L.y) ? L.y-1 : idy;
                #pragma unroll
                for (int ix = 0; ix < 2; ix++)
                {
                    int idx = homeX+ix;
                    idx = (idx >= L.x) ? L.x-1 : idx;
                    int id = idz + L.z * (idy + L.y * idx);
                    v[ix][iy][iz] = data[id];
                }
            }
        }

        float g3[3][2];
        #pragma unroll
        for (int iz = 0; iz < 2; iz++)
        {
            float g2[2][2];
            #pragma unroll
            for (int iy = 0; iy < 2; iy++)
            {
                g2[0][iy] = (v[1][iy][iz] - v[0][iy][iz]); /* f.x */
                g2[1][iy] = wx * (v[1][iy][iz] - v[0][iy][iz]) + v[0][iy][iz]; /* f.y & f.z */
            }
            // Mix along y.
            g3[0][iz] = wy * (g2[0][1] - g2[0][0]) + g2[0][0];
            g3[1][iz] = (g2[1][1] - g2[1][0]);
            g3[2][iz] = wy * (g2[1][1] - g2[1][0]) + g2[1][0];
        }
        return (wz * (g3[2][1] - g3[2][0]) + g3[2][0]);
    }
    else
    {
        return 0.f;
    }   
}
#endif

__global__ void AddEnergyKernel(float* Restrict energy, Vector3* Restrict position, Grid* Restrict grid, int numParticle)
{
    __shared__ float local[64];
    local[threadIdx.x] = 0.f;
    for(int id = threadIdx.x + 64 * blockIdx.x; id < numParticle; id += 64*gridDim.x)
    {
        Vector3 pos = position[id];
        float e;
        Grid g = *grid;
        e = InterpolateEnergy(pos, g.origin, g.invdx, g.L, g.data);
        //force[id] += f;
        local[threadIdx.x] += e;
        //atomic(&energy[blockIdx.x],e);
    }
    __syncthreads();
    //partially parallel reduction
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

__global__ void AddForceKernel(Vector3* Restrict force, Vector3* Restrict position, Grid* Restrict grid, int numParticle)
{
    for(int id = threadIdx.x + 64 * blockIdx.x; id < numParticle; id += 64*gridDim.x)
    {
        Vector3 pos = position[id];
        Vector3 f;
        Grid g = *grid;
        f = InterpolateForce(pos, g.origin, g.invdx, g.L, g.data);
        force[id] += f;
    }
}

void ConfinementGrid::AddForce(Vector3* force, Vector3* position, int* type, int numParticle)
{
    int numBlocks = (numParticle+63)/64;
    numBlocks = (numBlocks > 65535) ? 65535 : numBlocks;
    AddForceKernel<<<numBlocks, 64>>>(force, position, _grid, numParticle);
    cudaDeviceSynchronize();
}

void ConfinementGrid::AddEnergy(float* energy, Vector3* position, int* type, int numParticle)
{
    int numBlocks = (numParticle+63)/64;
    numBlocks = (numBlocks > 65535) ? 65535 : numBlocks;
    AddEnergyKernel<<<numBlocks, 64>>>(energy, position, _grid, numParticle);
    cudaDeviceSynchronize();
}
