#pragma once
#include "environ.h"

#define NEGH_USE_INNERCUT

class NEIGHLIST
{
private:
    numtype cutsq;
    numtype **x_prev;
    numtype dx2;
    int capacity;
    // void refold();
public:
    // bins with max dist smaller than cutoff has this mask and does not check dist when building list
    // lower 24 bits for bin index (thus max bins is 255^3)
    // higher 8 bits: [0][-][-][include _ 3 ?][include_w/o_check _ 3 ?][include _ 2 ?][include_w/o_check _ 2 ?][include_w/o_check ?]
    static constexpr int mask_include_2=1<<26,mask_include_3=1<<28;
    static constexpr int mask_bin=0xFFFFFF,mask_info=0x7F000000;
#ifdef NEGH_USE_INNERCUT
    static constexpr int mask_no_check=1<<24,mask_no_check_2=1<<25,mask_no_check_3=1<<27;
#endif
    int thread_id;
    // numtype **x_local;
    // numtype **f_local;
    // numtype *q_local;
    // int *typ_local,*mol_id_local,*intra_id_local;
    int *local_list;
    int n_local; 
    
    int *num_neigh;
    int **nei_list;
    int **bin_list;
    int **special;
    unsigned int nbuild;
    // bin_l must be created using build_bin_list with cutoff=cutoff+skin, or null_ptr if no build is called [in this case, build via build_gr(lp)]
    NEIGHLIST(int thr, numtype cutoff, numtype skin, int cap, int **bin_l);
    ~NEIGHLIST();
    //build neigh list w/ special info
    template <bool full> int build();
    // //build for computing gr
    // void build_gr();
    // // build 2 lists simultainously; lp will be the one for pair
    // void build_gr(NEIGHLIST *lp);
    bool isvalid();
    static int** build_bin_list(numtype cutoff, int max_nbin, numtype cutoff2=0, numtype cutoff3=0);
};

