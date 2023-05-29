#include "util.h"
#include <stdlib.h>
#include <mutex>
#include <cstring>

template void printMatrix<double>(double **mx,int m,int n,const char* info, char fmt);
template void printMatrix<float>(float **mx,int m,int n,const char* info, char fmt);
template void printMatrix<int>(int **mx,int m,int n,const char* info, char fmt);
template <typename TY> void printMatrix(TY **mx,int m,int n,const char* info, char fmt)
{
	std::cerr<<info<<"\n";
    switch (fmt)
    {
    case 'm':
        std::cerr<<'{'<<"\n";
    	for(int i=0;i<m;++i)
	    {
            std::cerr<<'{';
	    	for(int j=0;j<n;++j)
            {
	    		std::cerr<<mx[i][j];
                if (j+1<n) std::cerr<<',';
            }
            if(i+1<m) std::cerr<<"},\n";
            else std::cerr<<'}'<<"\n";
	    }
        std::cerr<<'}'<<"\n";
        break;
    case 'p':
        std::cerr<<'['<<"\n";
    	for(int i=0;i<m;++i)
	    {
            std::cerr<<'[';
	    	for(int j=0;j<n;++j)
            {
	    		std::cerr<<mx[i][j];
                if (j+1<n) std::cerr<<',';
            }
            if(i+1<m) std::cerr<<"],\n";
            else std::cerr<<']'<<"\n";
	    }
        std::cerr<<']'<<"\n";
        break;
    default:
    	for(int i=0;i<m;++i)
	    {
	    	for(int j=0;j<n;++j)
	    		std::cerr<<mx[i][j]<<' ';
	    	std::cerr<<"\n";
	    }
        break;
    }
	std::cerr<<"\n";
}

template void printMatrix_cs<double>(double *mx,int m,int n,const char* info, char fmt);
template void printMatrix_cs<float>(float *mx,int m,int n,const char* info, char fmt);
template void printMatrix_cs<int>(int *mx,int m,int n,const char* info, char fmt);
template <typename TY> void printMatrix_cs(TY *mx,int m,int n,const char* info, char fmt)
{
	std::cerr<<info<<"\n";
    switch (fmt)
    {
    case 'm':
        std::cerr<<'{'<<"\n";
    	for(int i=0;i<m;++i)
	    {
            std::cerr<<'{';
	    	for(int j=0;j<n;++j)
            {
	    		std::cerr<<mx[i*n+j];
                if (j+1<n) std::cerr<<',';
            }
            if(i+1<m) std::cerr<<"},\n";
            else std::cerr<<'}'<<"\n";
	    }
        std::cerr<<'}'<<"\n";
        break;
    case 'p':
        std::cerr<<'['<<"\n";
    	for(int i=0;i<m;++i)
	    {
            std::cerr<<'[';
	    	for(int j=0;j<n;++j)
            {
	    		std::cerr<<mx[i*n+j];
                if (j+1<n) std::cerr<<',';
            }
            if(i+1<m) std::cerr<<"],\n";
            else std::cerr<<']'<<"\n";
	    }
        std::cerr<<']'<<"\n";
        break;
    default:
    	for(int i=0;i<m;++i)
	    {
	    	for(int j=0;j<n;++j)
	    		std::cerr<<mx[i*n+j]<<' ';
	    	std::cerr<<"\n";
	    }
        break;
    }
	std::cerr<<"\n";
}

template void writeArrayBin<double>(double *arr,int n,const char* fname);
template void writeArrayBin<float> (float *arr,int n,const char* fname);
template void writeArrayBin<int> (int *arr,int n,const char* fname);
template <typename TY> void writeArrayBin(TY *arr,int n,const char* fname)
{
    FILE *fp=fopen(fname,"wb");
    fwrite(arr,sizeof(TY),n,fp);
    fclose(fp);
}


void terminate_program(const char *file, const int line, const char *msg)
{
    static std::mutex print_locker;
    print_locker.lock();
    std::cerr<<file<<":"<<line<<" -> "<<msg<<"\n";
    // print_locker.unlock();
    exit(EXIT_FAILURE);
}

char* non_empty_string(char *str)
{
    char c;
    while((c=*str))
    {
        if(!c) return nullptr;
        if(c!=' ' && c!='\n' && c!='\r' && c!='\t' ) return str;
        ++str;
    }
    return nullptr;
}

void fgets_non_empty(FILE* fp, char *str)
{
    while(! feof(fp))
    {
        fgets(str,SZ_FBF,fp);
        if(non_empty_string(str)) return;
    }
}

void string_get_env(char *str)
{
    int ps=0;
    char c;
    while(true) //find first '$'
    {
        if(! (c=str[ps]) ) return;
        if(c!='$') ++ps;
        else
        {
            str+=ps;
            break;
        }
    }

    char tempstr[SZ_FBF],envv[SZ_FBF],*gtev;
    memccpy(tempstr,str,0,SZ_FBF);
    ps=0;
    int n_write=0,n_env=0;
    enum {SGE_NON_VAR,SGE_VAR} status=SGE_NON_VAR;
    while(true)
    {
        c=tempstr[ps++];
        if(status==SGE_VAR)
        {
            switch(c)
            {
            case '}':
            case '\0':
                envv[n_env]='\0';
                n_env=0;
                if(!(gtev=std::getenv(envv))) END_PROGRAM("getenv failed");
                while((str[n_write++]=gtev[n_env++]));
                if(!c) goto cleanup;
                --n_write;
                status=SGE_NON_VAR;
                break;
            case '{':
                break;
            default:
                envv[n_env++]=c;
                break;
            }
        }
        else
        {
            switch(c)
            {
            case '$':
                status=SGE_VAR;
                n_env=0;
                break;
            case '\0':
                str[n_write]='\0';
                goto cleanup;
                break;
            default:
                str[n_write++]=c;
                break;
            }
        }
    }
cleanup:
    if(n_write>=SZ_FBF) END_PROGRAM("string overflow");
}