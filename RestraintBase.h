#ifndef RESTRAINTBASE_H_
#define RESTRAINTBASE_H_
#include "useful.h"
class RestraintBase
{
  public:
    virtual void AddForce(Vector3* force, Vector3* position) = 0;
    virtual void AddEnergy(float* energy, Vector3* position) = 0;
};

#endif
