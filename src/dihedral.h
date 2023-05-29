#pragma once
#include "environ.h"

class DIHEDRAL
{
public:
    int ndtyp, ndihedrals;
    int **dihedral;
    numtype **coef; // b0, 2k
    int *i_start,*i_end;
    DIHEDRAL(int n, int ns);
    ~DIHEDRAL();
    void setParam(int i, numtype c0=0, numtype c1=0, numtype c2=0, numtype c3=0);
    //use this to correct raw params seted
    void setParam();
    // this must be called before sorting atoms! 
    void autoGenerate();
    template <bool ev> void compute(int thread_id, numtype *erg, numtype *virial);
};
