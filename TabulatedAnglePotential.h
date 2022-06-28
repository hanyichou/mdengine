#ifndef TABULATEDANGLEPOTENTIAL_H_
#define TABULATEDANGLEPOTENTIAL_H_
#include <string>
#include "AngleTorsionBase.h"
class TabulatedAnglePotential : public AngleTorsionBase
{
  private:
    int4*   _angleList;
    float** _x;
    float** _y;
    int* _tableLength;
    int _numAngle;
    int _numType;
    void MemcpyHtoD(float**, float**, int*, int4*);
    void ReadTable(float*, float*, const int&, const std::string&);
  public:
    TabulatedAnglePotential(const std::string& filename);
    ~TabulatedAnglePotential();
    virtual void ComputeForce(Vector3* force, Vector3* position);
    virtual void ComputeEnergy(float* energy, Vector3* position);
};
#endif
