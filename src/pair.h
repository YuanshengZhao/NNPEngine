#pragma once
#include "environ.h"
#include "neighlist.h"

// #define DSF_ERFC

class LJCutCoulDSF
{
private:
    numtype **_48Epsilon;
    numtype **sigma;
    numtype **sigmasq;
    numtype **shift;
    numtype cutoff,cutsq;
    numtype vc, fc;
#ifdef DSF_ERFC
    numtype alpha, _2alpha_sqrtpi,alphasq;
#endif
    int ntyp;
public:
    LJCutCoulDSF(numtype cf,numtype alp);
    ~LJCutCoulDSF();
    void setParam(int it,int jt,numtype ep,numtype sg);
    void mixParam(char ty);
    template <bool ev> void compute(NEIGHLIST *list, numtype *erg, numtype *viral);
};

// pair potential and force for LJ
// coul po
class LJTableCoulDSF
{
private:
    numtype vc, fc;
    numtype *rr;
#ifdef DSF_ERFC
    numtype alpha, _2alpha_sqrtpi,alphasq;
#endif
public:
    numtype dr,r_dr;
    numtype **force_base,**potential_base;
    // force and potential, this may be different from *_base if modified when using RMDF or FMIRL; need to allocate using loadPotential
    numtype **force,**potential;
    // force and potential that ignore special, typically used for FMIRL; need to allocate using loadPotential
    numtype **force_nsp,**potential_nsp;
    int nbin_r,ntyp;
    numtype cutoff, cutsq;
    LJTableCoulDSF(numtype cf,int nbin);
    ~LJTableCoulDSF();
    // kind=0 (base) 1 (normal) or 2 (nsp); will create table if is 1 or 2; potfile can be null if only allocation is needed
    void loadPotential(int kind,const char *potfile=nullptr);
    template <bool ev, bool nsp>
    void compute(NEIGHLIST *list, numtype *erg, numtype *viral);
    void dumpForce(const char *fname, bool nsp, bool append=false);
};

