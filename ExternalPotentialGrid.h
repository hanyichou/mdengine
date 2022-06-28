#ifndef EXTERNALPOTENTIALGRID_H_
#define EXTERNALPOTENTIALGRID_H_
#include <string>
#include "useful.h"
#include "Grid.h"
#include "ExternalPotentialBase.h"
class ExternalPotentialGrid : public ExternalPotentialBase
{
  private:
    int  _numTypes;
    int* _numGrid;
    Grid** _grid;
    void ReadGridDxFormat(Grid& grid, const std::string& filename,const float&);
    void MemcpyHtoD(Grid** grid, int* numGrid);
  public:
    ExternalPotentialGrid(const std::string& filename, const int& numTypes);
    ~ExternalPotentialGrid();
    //compute the contribution of each 
    virtual void AddForce(Vector3* force, Vector3* position, int* type, int numParticle);
    virtual void AddEnergy(float* energy, Vector3* position, int* type, int numParticle);
};
#endif
