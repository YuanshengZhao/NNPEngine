#pragma once
#include "environ.h"

// quatic bond: U=x^2(a(x-b)^2+c) with x=b-b0
// a>=0 && c>0 is required
class BOND
{
public:
    int nbtyp, nbonds;
    int **bond;
    numtype **coef; // b0, 2a, b, 2c
    int *i_start,*i_end;
    BOND(int n, int ns);
    ~BOND();
    void setParam(int i, numtype b0, numtype ka, numtype kb, numtype kc);
    //use this to correct raw params seted
    void setParam();
    // this must be called before sorting atoms! 
    void autoGenerate();
    template <bool ev> void compute(int thread_id, numtype *erg, numtype *virial);
};
