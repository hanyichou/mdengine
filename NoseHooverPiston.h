#ifndef NOSEHOOVERLANGEVIN_H_
#define NOSEHOOVERLANGEVIN_H_
class NoseHooverLangevin
{
  private:
    SystemBox* _systemBox;
    float _kT;
    float _pressure;
    float _muT;
    float _muP;
    float _muR;
    float *_ksi_1, *_ksi_2, *eta;
    //random number state 
  public:
    void run(); 
}
#endif 
