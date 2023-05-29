#include "verlet.h"
#include "environ.h"
#include "memory.h"
#include <cstring>
#include "mathlib.h"

VERLET::VERLET(int thread_id)
{
    int i_start=ENVIRON::natom*thread_id/ENVIRON::NUM_THREAD;
    i_num=ENVIRON::natom*(thread_id+1)/ENVIRON::NUM_THREAD - i_start;
    x_start=ENVIRON::x+i_start;
    v_start=ENVIRON::v+i_start;
    f_start=ENVIRON::f+i_start;
    r_m_start=ENVIRON::r_m+i_start;
    f_prev_start=ENVIRON::f_prev+i_start;
}

void VERLET::setDt(numtype ddt)
{
    dt=ddt;
    hdt=dt*.5F;
}

VERLET::~VERLET()
{
}

//warning: need to conver f to a before calling!
void VERLET::updateX()
{
    numtype *xx,*vv;
    numtype *ff;
    for(int i=0;i<i_num;++i)
    {
        xx=x_start[i];
        vv=v_start[i];
        ff=f_start[i];
        xx[0]+=(vv[0]+ff[0]*hdt)*dt;
        xx[1]+=(vv[1]+ff[1]*hdt)*dt;
        xx[2]+=(vv[2]+ff[2]*hdt)*dt;
    }
    memcpy(f_prev_start[0],f_start[0],i_num*3*sizeof(numtype));
}

//warning: need to conver f to a before calling!
void VERLET::updateV()
{
    numtype *vv;
    numtype *ff,*fp;
    numtype rm;
    for(int i=0;i<i_num;++i)
    {
        vv=v_start[i];
        ff=f_start[i];
        fp=f_prev_start[i];
        rm=r_m_start[i]*ENVIRON::accelF;
        vv[0]+=(fp[0]+(ff[0]*=rm))*hdt;
        vv[1]+=(fp[1]+(ff[1]*=rm))*hdt;
        vv[2]+=(fp[2]+(ff[2]*=rm))*hdt;
    }   
}
// void VERLET::f2a()
// {
//     numtype *ff;
//     numtype rm;
//     for(int i=0;i<i_num;++i)
//     {
//         ff=f_start[i];
//         rm=r_m_start[i]*ENVIRON::accelF;
//         ff[0]*=rm;
//         ff[1]*=rm;
//         ff[2]*=rm;
//     }   
// }
void VERLET::scaleV(numtype sc)
{
    numtype *vv;
    for(int i=0;i<i_num;++i)
    {
        vv=v_start[i];
        vv[0]*=sc;
        vv[1]*=sc;
        vv[2]*=sc;
    }   
}

numtype VERLET::computeKE()
{
    numtype ke=0;
    numtype *vv;
    for(int i=0;i<i_num;++i)
    {
        vv=v_start[i];
        ke+=(sqr(vv[0])+sqr(vv[1])+sqr(vv[2]))/r_m_start[i];
    }
    return ke*ENVIRON::ekinF;
}

