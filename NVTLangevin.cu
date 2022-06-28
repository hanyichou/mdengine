#include <fstream>
#include <cstdio>
#include <string>
#include "NVTLangevin.h"
#include "WKFUtils.h"
#include "Utility.h"
NVTLangevin::NVTLangevin(const std::string& filename) : _totalSteps(-1), _outputFreq(-1),
_dt(-1.f), _gamma(-1.f), _compute_virial(0)
{
    std::ifstream infile(filename);
    if(infile.is_open())
    {
        std::string line;
        while(std::getline(infile, line))
        {
            Utility::remove_comment(line, std::string("#"));
            Utility::trim_both_sides(line);
            if(line.empty())
                continue;
            std::vector<std::string> elements;
            Utility::split(elements, line, std::string(" "));
            if(elements[0] == std::string("TotalSteps"))
                _totalSteps = Utility::string_to_number<int>(elements[1]); 
            else if(elements[0] == std::string("Step"))
                _dt = Utility::string_to_number<float>(elements[1]);
            else if(elements[0] == std::string("OutputFrequency"))
                _outputFreq = Utility::string_to_number<int>(elements[1]);
            else if(elements[0] == std::string("Damping"))
                _gamma = Utility::string_to_number<float>(elements[1]);
            else if(elements[0] == std::string("Virial"))
                _compute_virial = Utility::string_to_number<int>(elements[1]);
            line.clear();
        }
    }
    infile.close();
}

void NVTLangevin::run(Polymer* poly, const std::string& prefix)
{
    poly->SetUpWriters(_dt, _outputFreq, prefix);
    //cudaSetDevice(gpu_id);

    wkf_timerhandle timer0, timerS;
    timer0 = wkf_timer_create();
    timerS = wkf_timer_create();

    wkf_timer_start(timer0);
    wkf_timer_start(timerS);
    poly->ComputeForce(0); 
    for(int s = 1; s <= _totalSteps; ++s)
    {
        poly->NVTLangevinStep(s, _dt, _gamma);
        if(s % _outputFreq == 0)
        {
            cudaDeviceSynchronize();
            poly->WriteStep(_compute_virial);
            wkf_timer_stop(timerS);
            float percent = (100.0f * s) / _totalSteps;
            float msPerStep = wkf_timer_time(timerS) * 1000.0f / _outputFreq;
            float nsPerDay = 1. / msPerStep * 864E5f;

            printf("Step %ld [%.2f%% complete | %.3f ms/step | %.3E steps/day]\n",s, percent, msPerStep, nsPerDay);

            wkf_timer_start(timerS);
        }        
    }
    cudaDeviceSynchronize();
    wkf_timer_stop(timer0);
    const float elapsed = wkf_timer_time(timer0); // seconds
    int tot_hrs   = (int) std::fmod(elapsed / 3600.0f, 60.0f);
    int tot_min   = (int) std::fmod(elapsed / 60.0f, 60.0f);
    float tot_sec = std::fmod(elapsed, 60.0f);

    printf("\nFinal Step: %d\n", (int) _totalSteps);

    printf("Total Run Time: ");
    if (tot_hrs > 0) 
        printf("%dh%dm%.1fs\n", tot_hrs, tot_min, tot_sec);
    else if (tot_min > 0) 
        printf("%dm%.1fs\n", tot_min, tot_sec);
    else 
        printf("%.2fs\n", tot_sec);
}
