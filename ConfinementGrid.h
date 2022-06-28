#ifndef CONFINEMENTGRID_H_
#define CONFINEMENTGRID_H_
#include <string>
#include "useful.h"
#include "Grid.h"
#include "ExternalPotentialBase.h"
class ConfinementGrid : public ExternalPotentialBase
{
  private:
    Grid* _grid;
    void ReadGridDxFormat(Grid& grid, const std::string& filename);
    void MemcpyHtoD(Grid& grid);
  public:
    ConfinementGrid(const std::string& filename);
    ~ConfinementGrid();
    //compute the contribution of each 
    virtual void AddForce(Vector3* force, Vector3* position, int* type, int numParticle);
    virtual void AddEnergy(float* energy, Vector3* position, int* type, int numParticle);
};
#endif
