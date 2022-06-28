#ifndef GRID_H_
#define GRID_H_
typedef struct Grid
{
    int3 L;
    float3 dx;
    float3 invdx;
    float3 origin;
    float* data;
} Grid;
#endif
