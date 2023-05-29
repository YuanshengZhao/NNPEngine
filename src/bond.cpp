#include "bond.h"
#include "memory.h"
#include "mathlib.h"
#include "util.h"
#include <iostream>


BOND::BOND(int n, int ns)
{
    nbtyp=n;
    nbonds=ns;
    create2DArray(bond,ns,3);
    create2DArray(coef,nbtyp,4);
    create1DArray(i_start,ENVIRON::NUM_THREAD);
    create1DArray(i_end,ENVIRON::NUM_THREAD);
    for(int i=0;i<nbtyp;++i) coef[i][0]=coef[i][1]=0;
}

BOND::~BOND()
{
    destroy2DArray(bond);
    destroy2DArray(coef);
    destroy1DArray(i_start);
    destroy1DArray(i_end);
}

void BOND::setParam(int i, numtype b0, numtype ka, numtype kb, numtype kc)
{
    if(i>=nbtyp) END_PROGRAM("bad i");
    coef[i][0]=b0;
    coef[i][1]=ka*2;
    coef[i][2]=kb;
    coef[i][3]=kc*2;
}

void BOND::setParam()
{
    for(int i=0;i<nbtyp;++i)
    {
        coef[i][1]*=2;
        coef[i][3]*=2;
    }
}

void BOND::autoGenerate()
{
    int ibond=0;
    MOLECULE *mol;
    int **bond_mol,nb_mol,*jb,*jbm;
    for(int i=0;i<ENVIRON::natom;++i)
    {
        if(ENVIRON::intra_id[i]) continue;
        mol=ENVIRON::moltype + ENVIRON::mol_type[ENVIRON::mol_id[i]];
        bond_mol=mol->bond;
        nb_mol=mol->n_bond;
        for(int j=0;j<nb_mol;++j)
        {
            (jb=bond[ibond])[0]=(jbm=bond_mol[j])[0];
            if(jb[0]<0 || jb[0]>=nbtyp) END_PROGRAM("auto generate error");
            jb[1]=jbm[1]+i;
            jb[2]=jbm[2]+i;
            ++ibond;
        }
    }
    if(ibond != nbonds) END_PROGRAM("auto generate error");
    int j;
    for(int i=0;i<ENVIRON::NUM_THREAD;++i)
    {
        i_start[i]=i*nbonds/ENVIRON::NUM_THREAD;
        // align mol with thread
        if( (j=i_start[i]) )
        {
            for(;j<nbonds;++j)
                if(ENVIRON::mol_id[bond[j-1][1]] != ENVIRON::mol_id[bond[j][1]]) break;
            i_start[i]=j;
        }
        // std::cerr<<i<<" "<<j<<"\n";
    }
    for(j=1;j<ENVIRON::NUM_THREAD;++j) i_end[j-1]=i_start[j];
    i_end[ENVIRON::NUM_THREAD-1]=nbonds;

}

template void BOND::compute<true>(int thread_id, numtype *erg, numtype *virial);
template void BOND::compute<false>(int thread_id, numtype *erg, numtype *virial);

template <bool ev> void BOND::compute(int thread_id, numtype *erg, numtype *virial)
{
    int iend=i_end[thread_id];
    numtype *xi,*xj,*icf,dx,dy,dz,rr,ff,dr,bdr;
    numtype *fi,*fj;
    int temp,*ptemp;
    for(int i=i_start[thread_id];i<iend;++i)
    {
        ptemp=bond[i];
        icf=coef[ptemp[0]];
        xi=ENVIRON::x_mol[temp=ptemp[1]];
        fi=ENVIRON::f_mol[temp];
        xj=ENVIRON::x_mol[temp=ptemp[2]];
        fj=ENVIRON::f_mol[temp];

        dx=xi[0]-xj[0]; dy=xi[1]-xj[1]; dz=xi[2]-xj[2];
        dr=(rr=std::sqrt(sqr(dx)+sqr(dy)+sqr(dz)))-icf[0];
        bdr=icf[2]-dr;
        ff=dr*(icf[3]+icf[1]*bdr*(bdr-dr))/rr;
        dx*=ff, dy*=ff, dz*=ff;
        fi[0]-=dx; fi[1]-=dy; fi[2]-=dz;
        fj[0]+=dx; fj[1]+=dy; fj[2]+=dz;

        if(ev)
        {
            *erg += 0.5F*sqr(dr)*(icf[1]*sqr(bdr)+icf[3]);
            *virial -= ff*rr*rr;
        }   
    }
}
