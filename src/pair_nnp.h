#pragma once
#include "tensorflow/c/c_api.h"
#include <iostream>
#include "environ.h"
#include "neighlist.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"

class TFCALL
{
private:
    int NumInputs;
    int NumOutputs;
    TF_Graph* Graph;
    TF_SessionOptions* SessionOpts;
    TF_Session* Session;
    TF_Output* Input;
    TF_Output* Output;
    TF_Status* Status;
public:
    TF_Tensor** InputValues;
    TF_Tensor** OutputValues;
    void **inputs;
    numtype **outputs;
    void evaluate();
    void clearOutput();
    void init(const char* saved_model_dir,int n_in, char **in_names, int64_t **dim_in, TF_DataType *dtype,int *index_in,int n_out, char **out_names,int *index_out);
    ~TFCALL();
};

class NNPOTENTIAL
{
private:
    // pair pot is U = \epsilon (-sr^6 + c8 sr^8 + c10 sr^10 + c12 sr^12) + qij/r
    // with unit L2 norm of c, sr=\sigma/r  
    numtype ***coef_p, // ep sg c8 c10 c12 qij
            ***coef_w, ***coef_d, ***coef_r;
    numtype ***vfc;
    numtype cutoff,cutsq;
public:
    class TFCALL *tfcall;
    inline static constexpr int n_descriptor=20;
    numtype **descriptor,**dedg;
    NNPOTENTIAL(numtype cf,const char* paramfile,const char* wtfile, BOND *bnd, ANGLE* agl, DIHEDRAL *dih);
    ~NNPOTENTIAL();
    void computeDescriptor(NEIGHLIST *list);
    void nnEval();
    void evalFinalize(LOCAL *local);
    template <bool ev> 
    void compute(NEIGHLIST *list, numtype *erg, numtype *viral);
};

class NNPOTENTIAL_TBL
{
private: 
    numtype ***tbl_desc, ***tbl_f_desc; // pair, 4*n, r
    numtype cutoff,cutsq;
public:
    numtype ***tbl_pair_base, ***tbl_f_pair_base; //this is the potential and force from pair part of nnp
    numtype ***tbl_pair, ***tbl_f_pair; //this allows the pair pot to be modified (eg. by da/rmd)
    int nbin_r;
    numtype dr,r_dr;
    class TFCALL *tfcall;
    inline static constexpr int n_descriptor=20;
    numtype **descriptor,**dedg;
    NNPOTENTIAL_TBL(numtype cf,int nbr,const char* paramfile,const char* wtfile, BOND *bnd, ANGLE* agl, DIHEDRAL *dih);
    ~NNPOTENTIAL_TBL();
    void computeDescriptor(NEIGHLIST *list);
    void allocateTable();
    void nnEval();
    void evalFinalize(LOCAL *local);
    template <bool ev> 
    void compute(NEIGHLIST *list, numtype *erg, numtype *viral);
};

// class NNP_TRAIN
// {
// public:
//     numtype ***coef_w, ***coef_d, ***coef_r;
//     numtype ****D_ijk;
//     numtype cutoff,cutsq;
//     int ntyp;
//     class TFCALL *tfcall;
//     numtype **descriptor, **dedg;
//     numtype ***grd_w,***grd_d,***grd_r;
//     NNP_TRAIN(numtype cf,const char* paramfile);
//     ~NNP_TRAIN();
//     void computeDescriptor(NEIGHLIST *list);
//     // evaluate loss & gradient of nn parameters
//     void nnEval();
//     // evaluate gradient of descriptor parameters
//     // before calling this, delta F must be stored in F
//     // note: this is currently incomplete: computing dF/d(coef) needs jacobian and it is difficult to do so.
//     void compute(NEIGHLIST *list,numtype d_e);
//     void reduce_grad();
// };

// controlls whether use x_mol as input or x
// #define NNP_USE_X_MOL

// #define NNP_TRAIN_TF_NORMALIZE_CHARGE
// this compute every thing in TF!
class NNP_TRAIN_TF
{
private:
    int64_t **dim;
    int *ndim;
    // list info are managed here because it is variable
    // int *i_all, *j_all, *im_all, *tp_all,capacity;
    int n_train;
#ifdef NNP_TRAIN_TF_NORMALIZE_CHARGE
    int npair;
    numtype* num_pair;
#endif 
public:
    static constexpr numtype ry_kcal_mol=313.7547370315279;
    char **fnames,**fnames_shuffled;
    int ***sym_group, ***sym_group_shuffed, n_sym, *grpsz, *symsz;
    numtype **x_temp, **f_temp;
    inline static constexpr int n_input=29,n_output=18, n_wts=15;
    // inline static constexpr int idx_x=n_wts, 
    //                             idx_i=n_wts+1, 
    //                             idx_j=n_wts+2,
    //                             idx_m=n_wts+3,
    //                             idx_t=n_wts+4;
    int num_neigh_local[ENVIRON::NUM_THREAD],num_neigh;
    class TFCALL *tfcall;
    int *sz_in;
    // note: the list here must be full list!
    void allocate_neilist();
    void gather_neilist(NEIGHLIST* list);
#ifndef NNP_USE_X_MOL
    int *inv_atom_id;
    void update_intra(int thread_id, BOND *bnd, ANGLE* agl, DIHEDRAL *dih);
#endif
    numtype nnEval(numtype e0); //return loss
    numtype cutoff;
    NNP_TRAIN_TF(const char* paramfile, const char* weightfile,const char* optzfile,numtype cf, BOND *bnd, ANGLE* agl, DIHEDRAL *dih);
    ~NNP_TRAIN_TF();
    numtype **wts,**acc1,**acc2;
    inline static constexpr numtype beta_1=.9F, beta_2=.999F,cbta_1=1-beta_1, cbta_2=1-beta_2,epsl=1e-7, l2=.00F;
    int n_update;
    int *iwt_begin,*iwt_end;
    void apply_gradient(numtype lr,int thread_id);
    void save_wts(const char* weightfile,const char* optzfile);
    // data loader
    // must call ENVIRON::unsort first
    // this also do data augmentation
    double load_data(const char* fname);
    int load_train_info(const char* fname_data, const char* fname_sym);
    };

class NNP_DEPLOY_TF
{
private:
    int64_t **dim;
    int *ndim;
    // list info are managed here because it is variable
    // int *i_all, *j_all, *im_all, *tp_all,capacity;
public:
    inline static constexpr int n_input=11;

    int num_neigh_local[ENVIRON::NUM_THREAD],num_neigh;
    class TFCALL *tfcall;
    int *sz_in;
    // note: the list here must be full list!
    void gather_neilist(NEIGHLIST* list);
    void allocate_neilist();
#ifndef NNP_USE_X_MOL
    int *inv_atom_id;
    void update_intra(int thread_id, BOND *bnd, ANGLE* agl, DIHEDRAL *dih);
#endif
    void nnEval();
    void evalFinalize(LOCAL *local);
    void compute_pair(NEIGHLIST* list, numtype *erg);
    numtype cutoff;
    NNP_DEPLOY_TF(const char* paramfile,numtype cf, BOND *bnd, ANGLE* agl, DIHEDRAL *dih);
    ~NNP_DEPLOY_TF();

};