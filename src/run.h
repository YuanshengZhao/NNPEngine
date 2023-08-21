#include "environ.h"
#define NEIGH_SZ 2048

int classical(int argc, const char *argv[]);
int nnp_train_fixedBL(int argc, const char *argv[]);
int nnp_train_variableBL(int argc, const char *argv[]);
int nnp_run(int argc, const char *argv[]);

class BL_MANAGER
{
private:
    inline static constexpr int lst_capa=16;
    int cur_lst;
    int **lists[lst_capa];
    numtype bls[lst_capa];
    numtype grid_length, ghost_cutof, blst_cutof;
public:
    ~BL_MANAGER();
    BL_MANAGER(numtype gl,numtype gc,numtype bc);
    int** setBL(numtype bl);
};
