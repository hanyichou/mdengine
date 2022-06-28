#ifndef RESTRAINTHARMONIC_H_
#define RESTRAINTHARMONIC_H_
#include <string>
#include "RestraintBase.h"
class RestraintHarmonic : public RestraintBase
{
  private:
    float4* _param;
    int* _index;
    int _num;
    void MemcpyHtoD(float4*, int*);
  public:
    RestraintHarmonic(const std::string& filename);
    ~RestraintHarmonic();
    virtual void AddForce(Vector3* force, Vector3* position);
    virtual void AddEnergy(float* energy, Vector3* position);
};
#endif
