#include "mathlib.h"
#include "environ.h"
#include <random>
#include <iostream>

std::mt19937 rand_mt19937;
std::normal_distribution<double> rand_normal(0,1);
std::gamma_distribution<double> rand_gamma(1,ENVIRON::kB_300K);
std::uniform_real_distribution<double> rand_uniform(0,1);

void random_set_seed(u_int32_t seed)
{
    if(!seed)
    {
        std::random_device rd;
        seed=rd();
    }
    std::cerr<<"random seed is "<<seed<<"\n";
    rand_mt19937.seed(seed);
}
double random_gaussian(double std_dev)
{
    return rand_normal(rand_mt19937)*std_dev;
}

double random_uniform()
{
    return rand_uniform(rand_mt19937);
}

void random_set_gamma_ndof(double ndof)
{
    std::gamma_distribution<double>::param_type param(ndof/2,ENVIRON::kB_300K);
    rand_gamma.param(param);
    // std::cerr<<"alpha "<<rand_gamma.alpha()<<"\n";
}
double random_gamma(double rel_temp_300)
{
    return rand_gamma(rand_mt19937)*rel_temp_300;
}

unsigned random_int(unsigned max)
{
    auto minv=rand_mt19937.max() % max;
    auto rnd=rand_mt19937();
    while(rnd<=minv) rnd=rand_mt19937();
    return rnd%max;
}

template void shuffle<char*>(char** arr, int sz);
template void shuffle<int*>(int** arr, int sz);
template <typename TY> void shuffle(TY* arr, int sz)
{
    int rnd;
    TY temp;
    for(int i=sz-1;i>0;--i)
    {
        rnd=random_int(i+1);
        temp=arr[rnd]; arr[rnd]=arr[i]; arr[i]=temp;
    }
}
