#pragma once
#include "environ.h"

class MOLECULE
{
public:
    int n_atom,n_bond,n_angle,n_dihedral;
    int **bond, **angle, **dihedral;
    // numtype **bond_coef, **angle_coef, **dihedral_coef;
    int **special;
    inline static constexpr numtype special_lj[]={1,0,0,0.5}, special_coul[]={1,0,0,5/6.}; 
    MOLECULE();
    ~MOLECULE();
    void from_file(const char *fname);
};
