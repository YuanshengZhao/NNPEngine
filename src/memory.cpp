#include <iostream>
#include "memory.h"

template double*   create1DArray<double>(double *&,   int);
template double**  create2DArray<double>(double **&,  int, int);
template double*** create3DArray<double>(double ***&, int, int, int);
template void destroy1DArray<double>(double *&);
template void destroy2DArray<double>(double **&);
template void destroy3DArray<double>(double ***&);

template int*   create1DArray<int>(int *&,   int);
template int**  create2DArray<int>(int **&,  int, int);
template int*** create3DArray<int>(int ***&, int, int, int);
template void destroy1DArray<int>(int *&);
template void destroy2DArray<int>(int **&);
template void destroy3DArray<int>(int ***&);

template int64_t**  create2DArray<int64_t>(int64_t **&,  int, int);
template void destroy2DArray<int64_t>(int64_t **&);
template char**  create2DArray<char>(char **&,  int, int);
template void destroy2DArray<char>(char **&);
template int*** create2DArray<int*>(int ***&, int, int);
template void destroy2DArray<int*>(int ***&);

template float*   create1DArray<float>(float *&,   int);
template float**  create2DArray<float>(float **&,  int, int);
template float*** create3DArray<float>(float ***&, int, int, int);
template void destroy1DArray<float>(float *&);
template void destroy2DArray<float>(float **&);
template void destroy3DArray<float>(float ***&);

template <typename TY>
TY* create1DArray(TY *&arr, int i)
{
    arr=new TY[i];
    return arr;
}

template <typename TY>
void destroy1DArray(TY *&arr)
{
    delete[] arr;
    arr=nullptr;
}

template <typename TY>
TY** create2DArray(TY **&arr, int i, int j)
{
    TY *dat=new TY[i*j];
    arr=new TY*[i];
    arr[0]=dat;
    for(int _i=1;_i<i;++_i) arr[_i]=arr[_i-1]+j;
    return arr;
}

template <typename TY>
void destroy2DArray(TY **&arr)
{
    delete[] arr[0];
    delete[] arr;
    arr=nullptr;
}


template <typename TY>
TY*** create3DArray(TY ***&arr, int i, int j, int k)
{
    TY *dat=new TY[i*j*k];
    TY **ptr=new TY*[i*j];
    arr=new TY**[i];
    ptr[0]=dat;
    int ij=i*j;
    for(int _i=1;_i<ij;++_i) ptr[_i]=ptr[_i-1]+k;
    arr[0]=ptr;
    for(int _i=1;_i<i;++_i) arr[_i]=arr[_i-1]+j;
    return arr;
}

template <typename TY>
void destroy3DArray(TY ***&arr)
{
    delete[] arr[0][0];
    delete[] arr[0];
    delete[] arr;
    arr=nullptr;
}