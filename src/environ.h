#pragma once
//global macros come first

#define FLOAT_PRECESION

#ifdef FLOAT_PRECESION
using numtype = float;
#define FMT_NUMTYPE "%f"
#else
using numtype = double;
#define FMT_NUMTYPE "%lf"
#endif

#define EPSILON 1e-5F

// end global defs

#include "molecule.h"

class ENVIRON
{
private:
    // temp variable used for atom sorting
    inline static numtype **__x, **__v;
    inline static numtype **__f_prev;
    inline static numtype *__q, *__r_m;
    inline static int *__typ, *__atom_id;
    inline static int *__mol_id,*__intra_id;
    inline static int *__x_grd;
public:
    inline static numtype **x, **v;
    inline static numtype **f,**f_prev;
    inline static numtype *q, *r_m;
    inline static int *typ, *atom_id;
    inline static int *mol_id,*intra_id;


    //sorted by id, x in one molecule is in the same image (thus some may be ghost), while f is real.
    inline static numtype **x_mol;
    inline static numtype **f_mol;
    inline static int *typ_mol;

    inline static int natom, ntype, ntot, nghost;
    inline static int *typecount, **pairtype, npair;
    inline static numtype bl, r_bl, h_bl, m_bl;

    inline static class MOLECULE* moltype;
    inline static int nmol,nmoltype;
    // representative atom of molecules
    inline static int *mol_repre_id,*mol_type;

    inline static int *n_ghost_local;
    inline static numtype ***g_ghost_local,**g_ghost;
    inline static int **i_ghost_local,*i_ghost;

    inline static constexpr float ghost_capacity_frac=27, local_capacity_frac=8; // ghost_capacity/natoms, local_x_capacity*NUM_THREADS/natoms
    inline static constexpr numtype pi=3.141592653589793238462643383279502884197169399375105820974944592307816406286;
    inline static constexpr numtype two_pi=pi*2,four_pi=pi*4;

    /* 
    unit:
        length   -> A
        energy   -> kcal/mol
        force    -> kcal/mol/A
        time     -> fs
        velocity -> A/fs
        accel    -> A/fs^2
        charge   -> e

        electricK   = e^2/(4 pi ep0 A) -> kcal/mol
        accelF      = kcal/mol/A / D -> A/fs^2
        ekinF       = (A/fs)^2 * D / 2 -> kcal/mol
        kT_m_sqrt   = sqrt ( kB * K / D -> (A/fs)^2 )
        ek_T        = kcal/mol / (1.5 kB) -> K
    */

    inline static constexpr numtype electricK=332.06371, epsilon_0=1/(four_pi*electricK);
    inline static constexpr numtype accelF=4.184e-4;
    inline static constexpr numtype ekinF=1195.0286802753603;
    inline static constexpr numtype kT_m_sqrt=0.0009118367518929329;
    inline static constexpr numtype ek_T=335.47968899917714;
    inline static constexpr numtype kB_300K=0.5961612775922496;
    inline static constexpr numtype ev_GPa=6.9476954570553735;

#ifdef __APPLE__
    inline static constexpr int NUM_X=2, NUM_Y=2, NUM_Z=2;
#else
    inline static constexpr int NUM_X=3, NUM_Y=3, NUM_Z=3;
#endif
    inline static constexpr int NUM_THREAD=NUM_X*NUM_Y*NUM_Z;
    inline static int *natom_t;
    inline static int *x_thr,*__x_thr;
    inline static numtype t_rdx,t_rdy,t_rdz;

    // grid used for sorting atoms (including all ghosts)
    // used for building neigh list
    inline static int num_gx, num_gxsq, num_grid, num_gxm1;
    inline static int *grid_start,*grid_end;
    inline static int *grid_start_ghost,*grid_end_ghost;
    inline static int *x_grd,*x_grd_ghost;
    inline static numtype g_rdx, g_dx, g_offset;
    inline static int *natom_g;
    static void init_grid(numtype grid_length, numtype ghost_cutof);

    static void set_bl(numtype bl_in){bl=bl_in; r_bl=1/bl; h_bl=bl*.5F; m_bl=-bl;}
    static numtype distsq(numtype *x1, numtype *x2);
    static int tidyup_ghost();

    /* input file format:
    num_mol_types
    for each mol_type:    mol_file
    num_atom_type num_atom BL
    for each atom [must be in order of molecules!]:    mol_type typ mass charge x y z
    */
    static void initialize(const char *input_file);
    static void dump(const char *xyzfile, bool append);
    static void unsort(); // after this all ghosts will become invalid!
    static void sort();
    static void sort_ghost();
    static void verify_thread();
    static void verify_grid();
    static void initV(numtype tempera_rel_300,numtype fraction=2);
};

class LOCAL
{
public:
    static inline int ghost_capacity_local, local_capacity;
    int i_local_start, i_local_end, i_ghost_start, i_ghost_end;
    int thread_id;
    inline static numtype ghost_lim, m_ghost_lim; //=h_bl-ghost_cutoff;

    // data owned by thread
    // numtype **x_local;
    // numtype **f_local;
    // numtype *q_local;
    // int *typ_local,*mol_id_local,*intra_id_local;
    int n_local; // this != i_local_end-i_local_start

    // int fetch_atoms();

    static void set_ghost_lim(numtype ghost_cutoff){ghost_lim=ENVIRON::h_bl-ghost_cutoff; m_ghost_lim=-ghost_lim;}
    void generate_ghost();
    void update_ghost_and_info();
    template <bool tag> void update_ghost_pos();
    void refold_and_tag();
    template <typename TY> void clear_vector(TY **arr, int len);
    template <typename TY> void clear_vector_cs(TY *arr, int len); // continous storage
    template <typename TY> void subtract_vector(TY **arr, TY **arr2, TY **out, int len);
    template <typename TY> void subtract_vector_cs(TY *arr,TY *arr2,TY *out, int len); // continous storage
    LOCAL(int tid);
    ~LOCAL();
};
