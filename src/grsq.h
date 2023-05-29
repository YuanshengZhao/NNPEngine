#pragma once
#include "environ.h"
#include "neighlist.h"
#include "pair.h"
#include "pair_nnp.h"

// use 2 cutoff: one 
class GR
{
public:
    numtype *rr,dr,r_dr,**gr,r_maxsq, r_maxsq_inner;
    int ***gr_local;
    numtype **norm_gr;
    int nbin_r,nbin_r_inner;
    int *n_start,*n_end;
    GR(int nbi, numtype r_max=-1, numtype r_max_inner=-1);
    ~GR();
    //if use neighlist for pair, must use inner true!
    template <bool inner> void tally (NEIGHLIST *list);
    // compute gr w/o neigh_list
    void tally(int **bin_list, LOCAL *local);
    void reduce_local(int thread_id);
    void dump(const char *fname, bool append=false);
};

class SQ
{
public: 
    numtype *qq,**sq,q_max;
    int nbin_q,nbin_r;
    numtype **ftmx;
    int *iq_start,*iq_end;
    GR *gr;
    SQ(int nq,numtype qmax, GR *ggr);
    ~SQ();
    void compute(int thread_id);
    void dump(const char *fname, bool append=false);
};

class SQD
{
public: 
    numtype *qq,**sq, q_max;
    int nbin_q;
    numtype **gvec, *gscl;
    int ngv;
    int *iq_start,*iq_end;
    int *vbegin,*vend;
    SQD(int nq,numtype qmax);
    ~SQD();
    void compute(int thread_id);
    void dump(const char *fname, bool append=false);
private:
    //quick sort
    void sortQ(int lo, int hi);
    //partition for quick sort
    int partitionQ(int lo, int hi);
    numtype **ssa,**cca;
    numtype *factor,*base;
    // int *counti;
};

// //tested does not work (?)
// //1 type mol only 
// class SQM
// {
// public: 
//     numtype *qq,**sq;
//     numtype *rr,*gr,*norm,dr,r_dr;
//     numtype **cr_intra, **cq_intra;
//     numtype **gr_intra, **sq_intra;
//     int nbin_q,nbin_r;
//     numtype **ftmx,**sincqr;
//     numtype **conv_ker;
//     int *iq_start,*iq_end;
//     SQM(int nq,int nr,numtype q_max, numtype rc);
//     ~SQM();
//     void compute();
//     void dump(const char *fname);
// };

// #define RMDF_IGNORE_SPECIAL

class RMDF
{
private:
    numtype gamma;
    int n_sq;
    numtype **ftmx,**potmx;
    numtype **sqex,**wt;
    numtype ***sqwt,***fwt,*qq,*rr;
    numtype **force_q;
    numtype **partialsq,**deltasq,**deltasq_accu;
    numtype **force_pr,**potential_pr;
    numtype **force_bs,**potential_bs;

    int nbin_r,npair,nbin_q;
    int *ibegin_r,*iend_r,*ibegin_q,*iend_q;
public:
    numtype **sq;
    // gamma=1 is DA/RMD; 0<=gamma<1 is FMIRL with learning rate propto gamma
    RMDF(int nexp, char **fexp, SQ *c_sq, numtype strength, numtype _gamma, LJTableCoulDSF *pair, NNPOTENTIAL_TBL *pair_nnp);
    ~RMDF();
    numtype compute_Qspace(int thread_id);
    void compute_Rspace(int thread_id);
    void compute_Potential(int thread_id);
    // note that the table structure of nnp potential is different from that of classical potential
    void compute_Rspace_NNP(int thread_id);
    void compute_Potential_NNP(int thread_id);
    void dumpSQ(const char *fname, bool append=false);
    void loadDeltaSQ(const char *fname);
};
