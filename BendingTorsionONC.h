#ifndef BENDINGTORSIONONC_H_
#define BENDINGTORSIONONC_H_
#include <string>
#include "AngleTorsionBase.h"
class BendingTorsionONC : public AngleTorsionBase 
{
  private:
    int2* _bondList;
    int _numList;
    float3* _params;
    void MemcpyHtoD(int2*, const float3&);
  public:
    BendingTorsionONC(const std::string& filename);
    ~BendingTorsionONC();
    virtual void ComputeForce(Vector3* force, Vector3* position);    
    virtual void ComputeEnergy(float* energy, Vector3* position);
};
#endif
