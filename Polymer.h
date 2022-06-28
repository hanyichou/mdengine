#ifndef POLYMER_H_
#define POLYMER_H_
#include <vector>
#include <string>
#include <list>
#include "SystemBox.h"
#include "TrajectoryWriter.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "useful.h"
#include "AngleTorsionBase.h"
#include "ExternalPotentialBase.h"
#include "RestraintBase.h"

//typedef struct CollisionCell CollisionCell;
typedef struct Particle Particle;
typedef struct Bond Bond;
typedef struct VerletCell VerletCell;

class Polymer
{
  private:
    SystemBox _systemBox;
    float _kT;

    int _modelType;

    int _num;
    Vector3* _position;
    Vector3* _momentum;
    Vector3* _force;
    Vector3* _repulsiveForce;
    int* _type;

    int _numTypes;
    Particle* _particle;

    int _numGrouped;
    int *_particleID;
    int2 *_groupID;
    int2 *_groupRange;
    int* _noGroupedID;
    int* _rigidBodyID;

    int _numRigid;    
    Vector3* _positionCOM;
    Vector3* _momentumCOM;
    Vector3* _forceCOM;
    Vector3* _positionRelative;
    Matrix3* _orientation;
    Vector3* _angularMomentum;
    Vector3* _torque;
    float4*  _inertia;
    float*   _scaleFactor;

    float* _energy;
    float* _virial;
    std::vector<Bond*> _fene;
    std::vector<Bond*> _harmonic;
    
    VerletCell* _verletCell;

    curandStatePhilox4_32_10_t* _state;

    TrajectoryWriter* _coordinate_writer;
    TrajectoryWriter* _momentum_writer;
    TrajectoryWriter* _force_writer;

    Vector3* _buffer;
    int _numPairs;
    uint8_t* _exclusionMap;

    std::list<AngleTorsionBase*> _angleTorsionBase;
    std::list<ExternalPotentialBase*> _externalPotentialBase;
    std::list<RestraintBase*> _restraintBase;

    SystemBox SetUpSystemBox(const std::string& filename);
    void ParticleNVTLangevin(const int&, const float&, const float&);
    void GeneralizedParticleNVTLangevin(const int&, const float&, const float&);
    void UpdateVerletCellList();
    void ReadCoordinates(const std::string& filename);
    void ReadMomentum(const std::string& filename);
    void ReadTopology(const std::string& filename);
    void ReadGroup(const std::string& filename);
    void ReadExclusion(const std::string& filename);
    void CopyMassDtoH(float*); 
    void CopyInertiaHtoD(Vector3*); 
    void InitializeOrientation();
    void RedistributePosition();
    void RedistributeMomentum(); 
    void InitializeRigidPart(); 
    void InitializeMomentum();
    void ComputeHPSModel();
    void ComputeHPSModelEnergy();
    void ComputeHPSModelVirial();
    void ComputeHPSModelOnlyVirial();
    void ComputeKHModel();
    void ComputeKHModelEnergy();
    void ComputeKHModelVirial();
    void ComputeKHModelOnlyVirial();
    void ComputeParticleKineticEnergy();
    void ComputeRigidBodyKineticEnergy();
    void ComputeTotalEnergy();
    void ComputeVirial();
    void ComputeRigidPartMomentum();
    void InitializeRandomNumber(unsigned long long);
    void ComputeONCModel();
    void ComputeONCModelEnergy();
    void ComputeONCModelVirial();
    void ComputeONCModelOnlyVirial();
  public:
    Polymer(const std::string& filename);
    ~Polymer();
    //void InitializeRigidPart(Vector3* position, float kT);
    void SetUpWriters(const float&, const int&, const std::string&);    
    void NVTLangevinStep(const int&, const float&, const float&);
    void WriteStep(const int&);
    void ComputeForce(const int&);
    void ComputeForceVirial(const int&);

};

#define NUM_THREADS 128
#endif
