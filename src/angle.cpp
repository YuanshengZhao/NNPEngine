#include "angle.h"
#include "memory.h"
#include "mathlib.h"
#include "util.h"
// #include <iostream>

ANGLE::ANGLE(int n, int ns)
{
    natyp=n;
    nangles=ns;
    create2DArray(angle,ns,4);
    create2DArray(coef,natyp,4);
    create1DArray(i_start,ENVIRON::NUM_THREAD);
    create1DArray(i_end,ENVIRON::NUM_THREAD);
    for(int i=0;i<natyp;++i) coef[i][0]=coef[i][1]=0;
}

ANGLE::~ANGLE()
{
    destroy2DArray(angle);
    destroy2DArray(coef);
    destroy1DArray(i_start);
    destroy1DArray(i_end);
}

void ANGLE::setParam(int i, numtype theta_0, numtype ka, numtype kb, numtype kc)
{
    if(i>=natyp) END_PROGRAM("bad i");
    coef[i][0]=std::cos(theta_0*ENVIRON::pi/180);
    coef[i][1]=ka*2;
    coef[i][2]=kb;
    coef[i][3]=kc*2;
}

void ANGLE::setParam()
{
    for(int i=0;i<natyp;++i)
    {
        coef[i][1]*=2;
        coef[i][3]*=2;
    }
}

void ANGLE::autoGenerate()
{
    int iangle=0;
    MOLECULE *mol;
    int **angle_mol,na_mol,*ja,*jam;
    for(int i=0;i<ENVIRON::natom;++i)
    {
        if(ENVIRON::intra_id[i]) continue;
        mol=ENVIRON::moltype + ENVIRON::mol_type[ENVIRON::mol_id[i]];
        angle_mol=mol->angle;
        na_mol=mol->n_angle;
        for(int j=0;j<na_mol;++j)
        {
            (ja=angle[iangle])[0]=(jam=angle_mol[j])[0];
            if(ja[0]<0 || ja[0]>=natyp) END_PROGRAM("auto generate error");
            ja[1]=jam[1]+i;
            ja[2]=jam[2]+i;
            ja[3]=jam[3]+i;
            ++iangle;
        }
    }
    if(iangle != nangles) END_PROGRAM("auto generate error");
    int j;
    for(int i=0;i<ENVIRON::NUM_THREAD;++i)
    {
        i_start[i]=i*nangles/ENVIRON::NUM_THREAD;
        // align mol with thread
        if( (j=i_start[i]) )
        {
            for(;j<nangles;++j)
                if(ENVIRON::mol_id[angle[j-1][1]] != ENVIRON::mol_id[angle[j][1]]) break;
            i_start[i]=j;
        }
        // std::cerr<<i<<" "<<j<<"\n";
    }
    for(j=1;j<ENVIRON::NUM_THREAD;++j) i_end[j-1]=i_start[j];
    i_end[ENVIRON::NUM_THREAD-1]=nangles;

}

template void ANGLE::compute<true>(int thread_id, numtype *erg, numtype *virial);
template void ANGLE::compute<false>(int thread_id, numtype *erg, numtype *virial);

template <bool ev> void ANGLE::compute(int thread_id, numtype *erg, numtype *virial)
{
    int iend=i_end[thread_id];
    numtype *xi,*xj,*xk,*icf,dx1,dy1,dz1,dx2,dy2,dz2,r_r1,r_r2,fx,fy,fz,ff,fe,cq;
    numtype *fi,*fj,*fk;
    numtype dcq,b_dcq;
    int temp,*ptemp;
    for(int i=i_start[thread_id];i<iend;++i)
    {
        ptemp=angle[i];
        icf=coef[ptemp[0]];
        xi=ENVIRON::x_mol[temp=ptemp[1]];
        fi=ENVIRON::f_mol[temp];
        xj=ENVIRON::x_mol[temp=ptemp[2]];
        fj=ENVIRON::f_mol[temp];
        xk=ENVIRON::x_mol[temp=ptemp[3]];
        fk=ENVIRON::f_mol[temp];

        dx1=xi[0]-xj[0]; dy1=xi[1]-xj[1]; dz1=xi[2]-xj[2];
        dx2=xk[0]-xj[0]; dy2=xk[1]-xj[1]; dz2=xk[2]-xj[2];
        r_r1=1/std::sqrt(sqr(dx1)+sqr(dy1)+sqr(dz1));
        r_r2=1/std::sqrt(sqr(dx2)+sqr(dy2)+sqr(dz2));
        dx1*=r_r1; dy1*=r_r1; dz1*=r_r1;
        dx2*=r_r2; dy2*=r_r2; dz2*=r_r2;
        // r_r? stores inverse distance, dx? stores normalized dx
        //f1 = 2k (cos_theta - cq0) / |r1| * ( (r1_normalized.r2_normalized = cos_theta) r1_normalized - r2_normalized)
        dcq=(cq=dx1*dx2+dy1*dy2+dz1*dz2) - icf[0];
        b_dcq=icf[2]-dcq;
        fe=dcq*(icf[3]+icf[1]*b_dcq*(b_dcq-dcq));

        ff=fe*r_r1;
        fx=ff*(cq*dx1-dx2);
        fy=ff*(cq*dy1-dy2);
        fz=ff*(cq*dz1-dz2);
        fi[0]+=fx; fi[1]+=fy; fi[2]+=fz;
        fj[0]-=fx; fj[1]-=fy; fj[2]-=fz;

        ff=fe*r_r2;
        fx=ff*(dx1-cq*dx2);
        fy=ff*(dy1-cq*dy2);
        fz=ff*(dz1-cq*dz2);
        fk[0]-=fx; fk[1]-=fy; fk[2]-=fz;
        fj[0]+=fx; fj[1]+=fy; fj[2]+=fz;

        if(ev)
        {
            *erg += 0.5F*sqr(dcq)*(icf[1]*sqr(b_dcq)+icf[3]);
        }   
    }
}
