#pragma once
#include "environ.h"

// quatic angle: U=x^2(a(x-b)^2+c) with x=cos\theta-\cos\theta_0
// a>=0 && c>0 is required
class ANGLE
{
public:
    int natyp, nangles;
    int **angle;
    numtype **coef; // b0, 2a, b, 2c
    int *i_start,*i_end;
    ANGLE(int n, int ns);
    ~ANGLE();
    void setParam(int i, numtype theta_0, numtype ka, numtype kb, numtype kc);
    //use this to correct raw params seted
    void setParam();
    // this must be called before sorting atoms! 
    void autoGenerate();
    template <bool ev> void compute(int thread_id, numtype *erg, numtype *virial);
};
