#ifndef EXTERNALPOTENTIALBASE_H_
#define EXTERNALPOTENTIALBASE_H_
#include "useful.h"
class ExternalPotentialBase
{
  public:
    virtual void AddForce(Vector3* force, Vector3* position, int* type, int numParticle) = 0;
    virtual void AddEnergy(float* energy, Vector3* position, int* type, int numParticle) = 0;
};
#endif 
