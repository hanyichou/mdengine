#include "InterpolateGrid.h"
__device__ Vector3 InterpolateForce(const Vector3& pos, const float3& origin, const float3& invdx, const int3& L, float* data)
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

__device__ float InterpolateEnergy(const Vector3& pos, const float3& origin, const float3& invdx, const int3& L, float* data)
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
