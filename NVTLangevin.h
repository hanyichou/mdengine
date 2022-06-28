#ifndef NVTLANGEVIN_H_
#define NVTLANGEVIN_H_
#include <vector>
#include <string>
#include "Polymer.h"
class NVTLangevin
{
  private:
    int _totalSteps;
    int _outputFreq;
    float _dt;
    float _gamma;
    int _compute_virial;
  public:
    NVTLangevin(const std::string& filename);
    void run(Polymer* poly, const std::string& prefix);
};
#endif
