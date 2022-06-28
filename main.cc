#include <string>
#include <cstdlib>
#include "Polymer.h"
#include "NVTLangevin.h"
int main(int argc, const char** argv)
{
    cudaSetDevice(atoi(argv[3]));
    cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
    std::string filename = std::string(argv[1]);
    Polymer* poly = new Polymer(filename);
    NVTLangevin* md = new NVTLangevin(filename);
    std::string out_prefix = std::string(argv[2]);
    md->run(poly, out_prefix);
    delete md;
    delete poly;
    return 0;
}
