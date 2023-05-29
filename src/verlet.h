#pragma once
#include "environ.h"

class VERLET
{
private:
    inline static numtype dt,hdt;
    numtype **f_prev_start;
    numtype **x_start, **v_start, *r_m_start;
    numtype **f_start;
    int i_num;
public:
    VERLET(int thread_id);
    ~VERLET();
    //usage: updateX => calc force => convert f->a => updateV
    static void setDt(numtype ddt);
    void updateX();
    void updateV();
    // void f2a();
    void scaleV(numtype sc);
    numtype computeKE();
};
