#pragma once

#define SZ_FBF 256
#include <iostream>

template <typename TY> void printMatrix(TY **mx,int m,int n,const char* info, char fmt=' ');
template <typename TY> void printMatrix_cs(TY *mx,int m,int n,const char* info, char fmt=' ');
template <typename TY> void writeArrayBin(TY *arr,int n,const char* fname);
char* non_empty_string(char *str);
void fgets_non_empty(FILE* fp, char *str);
void string_get_env(char *str);
#define END_PROGRAM(msg) terminate_program(__FILE__,__LINE__,msg)
void terminate_program(const char *file, const int line, const char *msg);
