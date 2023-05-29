#pragma once
#include <cmath>

template <typename TY> inline TY sqr(TY __x) {return __x*__x;}
template <typename TY> inline TY cub(TY __x) {return __x*__x*__x;}
// template <typename TY> TY sphBessel0(TY __x) {return (__x<0.01F && __x>-0.01F)? 1-__x*__x/6 : std::sin(__x)/__x;}

void random_set_seed(u_int32_t seed=0);
double random_gaussian(double std_dev);
void random_set_gamma_ndof(double ndof);
double random_gamma(double rel_temp_300);
double random_uniform();
unsigned random_int(unsigned max);
template <typename TY> void shuffle(TY* arr, int sz);