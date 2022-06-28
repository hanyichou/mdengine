#ifndef DIHEDRALPOTENTIALONC_H_
#define DIHEDRALPOTENTIALONC_H_
#include <string>
#include "AngleTorsionBase.h"
class DihedralPotentialONC : public AngleTorsionBase
{
  private:
    int _numDihedral;
    int4* _dihedralList;
    int* _type;
    float* _alpha;
    void MemcpyHtoD(int4*, int*, float*);
  public:
    DihedralPotentialONC(const std::string& filename);
    ~DihedralPotentialONC();
    virtual void ComputeForce(Vector3* force, Vector3* position);
    virtual void ComputeEnergy(float* energy, Vector3* position);
};
#endif
