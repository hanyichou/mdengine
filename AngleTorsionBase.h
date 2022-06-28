#ifndef ANGLETORSIONBASE_H_
#define ANGLETORSIONBASE_H_
#include "useful.h"
class AngleTorsionBase
{
  public:
    virtual void ComputeForce(Vector3* force, Vector3* position) = 0;
    virtual void ComputeEnergy(float* energy, Vector3* position) = 0;
};
#endif
