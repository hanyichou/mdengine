#ifndef INTERPOLATEGRID_H_
#define INTERPOLATEGRID_H_
#include "useful.h"
__device__ Vector3 InterpolateForce(const Vector3& pos, const float3& origin, const float3& invdx, const int3& L, float* data);
__device__ float InterpolateEnergy(const Vector3& pos, const float3& origin, const float3& invdx, const int3& L, float* data);
#endif
