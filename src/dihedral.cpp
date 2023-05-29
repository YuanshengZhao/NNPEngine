#include "dihedral.h"
#include "memory.h"
#include "mathlib.h"
#include "util.h"
#include <iostream>


DIHEDRAL::DIHEDRAL(int n, int ns)
{
    ndtyp=n;
    ndihedrals=ns;
    create2DArray(dihedral,ns,5);
    create2DArray(coef,ndtyp,4);// for fourier up to cos (3 theta)
    create1DArray(i_start,ENVIRON::NUM_THREAD);
    create1DArray(i_end,ENVIRON::NUM_THREAD);
    for(int i=0;i<ndtyp;++i) coef[i][0]=coef[i][1]=coef[i][2]=coef[i][3]=0;
}

DIHEDRAL::~DIHEDRAL()
{
    destroy2DArray(dihedral);
    destroy2DArray(coef);
    destroy1DArray(i_start);
    destroy1DArray(i_end);
}

void DIHEDRAL::setParam(int i, numtype c0, numtype c1, numtype c2, numtype c3)
{
    // c0 + c1 cq + c2 c2q + c3 c3q
    if(i>=ndtyp) END_PROGRAM("bad i");
    coef[i][0]=c0-c2;
    coef[i][1]=c1-3*c3;
    coef[i][2]=c2*2 *2;
    coef[i][3]=c3*4 *3;
}

void DIHEDRAL::setParam()
{
    for(int i=0;i<ndtyp;++i)
    {
        coef[i][2]*=2;
        coef[i][3]*=3;
    }
}

void DIHEDRAL::autoGenerate()
{
    int idihedral=0;
    MOLECULE *mol;
    int **dihedral_mol,nd_mol,*jd,*jdm;
    for(int i=0;i<ENVIRON::natom;++i)
    {
        if(ENVIRON::intra_id[i]) continue;
        mol=ENVIRON::moltype + ENVIRON::mol_type[ENVIRON::mol_id[i]];
        dihedral_mol=mol->dihedral;
        nd_mol=mol->n_dihedral;
        for(int j=0;j<nd_mol;++j)
        {
            (jd=dihedral[idihedral])[0]=(jdm=dihedral_mol[j])[0];
            if(jd[0]<0 || jd[0]>=ndtyp) END_PROGRAM("auto generate error");
            jd[1]=jdm[1]+i;
            jd[2]=jdm[2]+i;
            jd[3]=jdm[3]+i;
            jd[4]=jdm[4]+i;
            ++idihedral;
        }
    }
    if(idihedral != ndihedrals) END_PROGRAM("auto generate error");
    int j;
    for(int i=0;i<ENVIRON::NUM_THREAD;++i)
    {
        i_start[i]=i*ndihedrals/ENVIRON::NUM_THREAD;
        // align mol with thread
        if( (j=i_start[i]) )
        {
            for(;j<ndihedrals;++j)
                if(ENVIRON::mol_id[dihedral[j-1][1]] != ENVIRON::mol_id[dihedral[j][1]]) break;
            i_start[i]=j;
        }
        // std::cerr<<i<<" "<<j<<"\n";
    }
    for(j=1;j<ENVIRON::NUM_THREAD;++j) i_end[j-1]=i_start[j];
    i_end[ENVIRON::NUM_THREAD-1]=ndihedrals;

}

template void DIHEDRAL::compute<true>(int thread_id, numtype *erg, numtype *virial);
template void DIHEDRAL::compute<false>(int thread_id, numtype *erg, numtype *virial);

template <bool ev> void DIHEDRAL::compute(int thread_id, numtype *erg, numtype *virial)
{
    int iend=i_end[thread_id];
    numtype *xi,*xj,*xk,*xl,*icf,dx1,dy1,dz1,dx2,dy2,dz2,dx3,dy3,dz3,r_rsq2,r_r1,r_r3,dot12,dot32,cq;
    numtype *fi,*fj,*fk,*fl;
    numtype f1x,f1y,f1z,f3x,f3y,f3z,f2x,f2y,f2z,ff;
    int temp,*ptemp;
    for(int i=i_start[thread_id];i<iend;++i)
    {
        ptemp=dihedral[i];
        icf=coef[ptemp[0]];
        xi=ENVIRON::x_mol[temp=ptemp[1]];
        fi=ENVIRON::f_mol[temp];
        xj=ENVIRON::x_mol[temp=ptemp[2]];
        fj=ENVIRON::f_mol[temp];
        xk=ENVIRON::x_mol[temp=ptemp[3]];
        fk=ENVIRON::f_mol[temp];
        xl=ENVIRON::x_mol[temp=ptemp[4]];
        fl=ENVIRON::f_mol[temp];

        dx1=xi[0]-xj[0]; dy1=xi[1]-xj[1]; dz1=xi[2]-xj[2];
        dx2=xk[0]-xj[0]; dy2=xk[1]-xj[1]; dz2=xk[2]-xj[2];
        dx3=xl[0]-xk[0]; dy3=xl[1]-xk[1]; dz3=xl[2]-xk[2];

        r_rsq2=1/(sqr(dx2)+sqr(dy2)+sqr(dz2));
        dot12=(dx1*dx2+dy1*dy2+dz1*dz2)*r_rsq2;
        dot32=(dx3*dx2+dy3*dy2+dz3*dz2)*r_rsq2;

        // projection
        dx1-=dx2*dot12; dy1-=dy2*dot12; dz1-=dz2*dot12; 
        dx3-=dx2*dot32; dy3-=dy2*dot32; dz3-=dz2*dot32; 
        r_r1=1/std::sqrt(sqr(dx1)+sqr(dy1)+sqr(dz1));
        r_r3=1/std::sqrt(sqr(dx3)+sqr(dy3)+sqr(dz3));
        dx1*=r_r1; dy1*=r_r1; dz1*=r_r1; 
        dx3*=r_r3; dy3*=r_r3; dz3*=r_r3; 
        cq=dx1*dx3+dy1*dy3+dz1*dz3;

        ff=(icf[3]*cq+icf[2])*cq+icf[1];
        r_r1*=ff; r_r3*=ff;

        f1x=r_r1*(cq*dx1-dx3); f1y=r_r1*(cq*dy1-dy3); f1z=r_r1*(cq*dz1-dz3); 
        f3x=r_r3*(cq*dx3-dx1); f3y=r_r3*(cq*dy3-dy1); f3z=r_r3*(cq*dz3-dz1); 
        f2x=f1x*dot12+f3x*dot32;
        f2y=f1y*dot12+f3y*dot32;
        f2z=f1z*dot12+f3z*dot32;

        fi[0]+=f1x;       fi[1]+=f1y;       fi[2]+=f1z;
        fl[0]+=f3x;       fl[1]+=f3y;       fl[2]+=f3z;
        fj[0]+=(f2x-f1x); fj[1]+=(f2y-f1y); fj[2]+=(f2z-f1z);
        fk[0]-=(f2x+f3x); fk[1]-=(f2y+f3y); fk[2]-=(f2z+f3z);

        if(ev)
        {
            // std::cerr<<cq<<" "<<std::acos(cq)/M_PI*180<<"\n";
            *erg += ((icf[3]/3*cq + icf[2]/2)*cq + icf[1])*cq + icf[0];
        }   
    }
}
