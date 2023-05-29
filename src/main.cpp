#include "run.h"
#include "util.h"
#include <iostream>
#include <cstring>
#include "mathlib.h"

int main(int argc, const char *argv[])
{
    std::cerr<<"(C) YZ, compiled at " __TIME__ " " __DATE__ ".\n";
    std::cout<<std::scientific;
    std::cerr<<std::scientific;
    // std::cout.precision(8);
    // std::cerr.precision(8);
    if(argc<2) END_PROGRAM("invalid arg");

    random_set_seed();

    if(!strcmp(argv[1],"classical")) return classical(argc-1,argv+1);
    if(!strcmp(argv[1],"nnp_train")) return nnp_train(argc-1,argv+1);
    if(!strcmp(argv[1],"nnp_run")) return nnp_run(argc-1,argv+1);

    END_PROGRAM("unknown arg");
}
