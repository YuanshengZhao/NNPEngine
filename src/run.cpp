#include "environ.h"
#include "neighlist.h"
#include "pair.h"
#include "pair_nnp.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "verlet.h"
#include "mathlib.h"
#include "memory.h"
#include "grsq.h"
#include "util.h"
#include "run.h"
#include <iostream>
#include <iomanip>
#ifdef __APPLE__
#include </usr/local/opt/libomp/include/omp.h>
#else
#include <omp.h>
#endif

#define BIN_NEIGH 1024
#define THERMO_ANDERSEN_FRAC 0.1

/*
input file format: 
    note: any number of blank rows are allowed
    note: can append extra items in each row for annotation (except for blank rows)
    note: use ${} for getting env variables (parenthes is mandatory)
    ***************************
    initial_configuration file
    pair_cutoff skin gr_cutoff
    pair.n_table pair.pot_file
    bond.ntyp
    b0 k
    ...
    angle.ntyp
    theta_0 ka kb kc
    ...
    dihedral.ntyp
    c0 c1 c2 c3
    ...
    gr.nbin_r (must match pair.n_table) sq.nbin_q sq.q_max
    rmdf.strength rmdf.gamma rmdf.sp_exp
    rmdf.initia_weight
    temp_start temp_end
    total_steps log_interval scale_interval [gr_interval gr_interval_long]
    output
    ***************************
*/


int classical(int argc, const char *argv[])
{
    if(argc!=2) END_PROGRAM("invalid arg");

    char file_buf[SZ_FBF],str_buf[SZ_FBF],str_buf2[SZ_FBF],*strbufs[]={str_buf,str_buf2};
    FILE *fp=fopen(argv[1],"r");
    int tpint,tpint2,tpint3,tpint4;
    numtype tpflt,tpflt2,tpflt3,tpflt4;

    std::cerr<<"NUM_THREAD = "<<ENVIRON::NUM_THREAD<<"\n";

    // initialization 
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s",str_buf) != 1) END_PROGRAM("read error");
    ENVIRON::initialize(str_buf);

    random_set_gamma_ndof(ENVIRON::natom*3);

    // setup cutoff & potential
    numtype pair_cutoff=10.,skin=1.,gr_cutoff=19.;
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,FMT_NUMTYPE " " FMT_NUMTYPE " " FMT_NUMTYPE, &pair_cutoff, &skin, &gr_cutoff) != 3) END_PROGRAM("read error");
    std::cerr<<"cutoff "<<pair_cutoff<<" "<<gr_cutoff<<" skin "<<skin<<"\n";
    const numtype comm_cutoff=skin+std::max(gr_cutoff,pair_cutoff);

    LOCAL::set_ghost_lim(comm_cutoff);
    ENVIRON::init_grid(0.5*(skin+pair_cutoff),comm_cutoff);

    int **bin_list_pair=NEIGHLIST::build_bin_list(pair_cutoff+skin,126);
    int **bin_list_gr=NEIGHLIST::build_bin_list(gr_cutoff+skin,BIN_NEIGH);
    // int **bin_list_gr=NEIGHLIST::build_bin_list(gr_cutoff+skin,650,pair_cutoff+skin);

    // LJCutCoulDSF pair(ENVIRON::ntype,pair_cutoff,.0);
    // pair.setParam(0,0,0.1094,3.39967);
    // pair.setParam(1,1,0.0157,2.47135);
    // pair.setParam(2,2,0.0000,0.00000);
    // pair.setParam(3,3,0.2104,3.06647);
    // pair.mixParam('A');

    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%d %s",&tpint,str_buf)!=2)  END_PROGRAM("read error");
    std::cerr<<"table "<<tpint<<" cols\n";
    LJTableCoulDSF pair(pair_cutoff,tpint);
    pair.loadPotential(0,str_buf);
#ifdef RMDF_IGNORE_SPECIAL
    pair.loadPotential(2);
#else
    pair.loadPotential(1);
#endif

    //compute n_bounds
    MOLECULE *imol;
    tpint2=tpint3=tpint4=0;
    for(int im=0;im<ENVIRON::nmol;++im)
    {   
        imol=ENVIRON::moltype+ENVIRON::mol_type[im];
        tpint2+= imol->n_bond;
        tpint3+= imol->n_angle;
        tpint4+= imol->n_dihedral;
    }


    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%d",&tpint)!=1)  END_PROGRAM("read error");
    BOND bond(tpint,tpint2);
    for(int i=0;i<tpint;++i)
    {
        fgets_non_empty(fp,file_buf); string_get_env(file_buf);
        if(sscanf(file_buf,FMT_NUMTYPE " " FMT_NUMTYPE,&tpflt,&tpflt2)!=2)  END_PROGRAM("read error");
        bond.setParam(i,tpflt,0,0,tpflt2);
        std::cerr<<"bond "<<i<<" "<<tpflt<<" "<<tpflt2<<"\n";
    }
    bond.autoGenerate();

    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%d",&tpint)!=1)  END_PROGRAM("read error");
    ANGLE angle(tpint,tpint3);
    for(int i=0;i<tpint;++i)
    {
        fgets_non_empty(fp,file_buf); string_get_env(file_buf);
        if(sscanf(file_buf,FMT_NUMTYPE " " FMT_NUMTYPE " " FMT_NUMTYPE " " FMT_NUMTYPE,&tpflt,&tpflt2,&tpflt3,&tpflt4)!=4)  END_PROGRAM("read error");
        angle.setParam(i,tpflt,tpflt2,tpflt3,tpflt4);
        std::cerr<<"angle "<<i<<" "<<tpflt<<" "<<tpflt2<<" "<<tpflt3<<" "<<tpflt4<<"\n";
    }
    angle.autoGenerate();

    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%d",&tpint)!=1)  END_PROGRAM("read error");
    DIHEDRAL dihedral(tpint,tpint4);
    for(int i=0;i<tpint;++i)
    {
        fgets_non_empty(fp,file_buf); string_get_env(file_buf);
        if(sscanf(file_buf,FMT_NUMTYPE " " FMT_NUMTYPE " " FMT_NUMTYPE " " FMT_NUMTYPE,&tpflt,&tpflt2,&tpflt3,&tpflt4)!=4)  END_PROGRAM("read error");
        dihedral.setParam(i,tpflt,tpflt2,tpflt3,tpflt4);
        std::cerr<<"dihedral "<<i<<" "<<tpflt<<" "<<tpflt2<<" "<<tpflt3<<" "<<tpflt4<<"\n";
    }
    dihedral.autoGenerate();

    //grsq
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%d %d " FMT_NUMTYPE,&tpint,&tpint2,&tpflt)!=3)  END_PROGRAM("read error");
    GR gr(tpint,gr_cutoff,pair_cutoff);
    SQ sq(tpint2,tpflt,&gr);
    numtype gamma;
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,FMT_NUMTYPE " " FMT_NUMTYPE,&tpflt,&gamma)!=2)  END_PROGRAM("read error");
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%d %n",&tpint,&tpint3)!=1)  END_PROGRAM("read error");
    if(tpint>2) END_PROGRAM("max n_sq is 2");
    for(int i=0;i<tpint;++i)
    {
        if(sscanf(file_buf+tpint3,"%s %n",strbufs[i],&tpint2)!=1)  END_PROGRAM("read error");
        tpint3+=tpint2;
    }
    RMDF rmdf(tpint,strbufs,&sq,tpflt,gamma,&pair,nullptr);
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s",str_buf) != 1) END_PROGRAM("read error");
    rmdf.loadDeltaSQ(str_buf);

    numtype erg,virial,ekin,temperature,press,erg_exp;
    bool is_list_valid;
#ifdef THERMO_ANDERSEN_FRAC
    std::cerr<<"Andersen thermo: "<<THERMO_ANDERSEN_FRAC<<"\n";
#else
    std::cerr<<"CSVR thermo\n";
    numtype ke_scale;
#endif
    numtype temp_start,temp_end;
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,FMT_NUMTYPE " " FMT_NUMTYPE,&temp_start,&temp_end) != 2) END_PROGRAM("read error");
    std::cerr<<"temp "<<temp_start<<" -> "<<temp_end<<"\n";
    temp_start/=300; temp_end/=300;
    int total_steps, log_interval, scale_interval, gr_interval=-1, gr_interval_long=-1;
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%d %d %d %d %d",&total_steps,&log_interval,&scale_interval,&gr_interval,&gr_interval_long) < 3) END_PROGRAM("read error");
    gr_interval=gr_interval<0? total_steps:gr_interval; gr_interval_long=gr_interval_long<0? total_steps:gr_interval_long; 
    if((total_steps%log_interval) || (total_steps%scale_interval)) END_PROGRAM("bad steps");
    std::cerr<<"MD steps: total "<<total_steps<<"; log "<<log_interval<<"; scale "<<scale_interval<<"; gr "<<gr_interval<<"; long "<<gr_interval_long<<"\n";

    ENVIRON::initV(temp_start);
    VERLET::setDt(-1); // first set to -1 to undo first update X

    // enter parallel region
    std::cerr<<std::setw(7)<<"md_step"
            <<std::setw(18)<<"temp [K]"
            <<std::setw(18)<<"press [GPa]"
            <<std::setw(18)<<"erg [kcal/mol]"
            <<std::setw(18)<<"dif [kcal/mol]"
            // <<std::setw(18)<<"etot [kcal/mol]"
            <<std::setw(10)<<"nbuild"
            <<"\n";
    #pragma omp parallel num_threads(ENVIRON::NUM_THREAD)
    {
        if(omp_get_num_threads()!=ENVIRON::NUM_THREAD) END_PROGRAM("omp failed to start");
        bool f_log;
        int comm_me=omp_get_thread_num();
        LOCAL local(comm_me);

        NEIGHLIST list_pair(comm_me,pair_cutoff,skin,NEIGH_SZ,bin_list_pair);
        // NEIGHLIST list_pair(comm_me,pair_cutoff,skin,NEIGH_SZ,nullptr);
        // NEIGHLIST list_gr(comm_me,gr_cutoff,skin,NEIGH_SZ,bin_list_gr);

        VERLET verlet(comm_me);
        numtype erg_local=INFINITY, virial_local=INFINITY, ekin_local=INFINITY,erg_exp_local=INFINITY;
        bool is_list_valid_local;

        // init force
        local.clear_vector(ENVIRON::f,3);
        #pragma omp barrier
        verlet.updateX();
        #pragma omp barrier
        #pragma omp single
        {
            VERLET::setDt(1);
        }
        
        // start MD simulation
        for(int md_step=0;md_step<=total_steps;++md_step)
        {
            f_log = !(md_step%log_interval);
            // step 1: update X
            verlet.updateX();

            //step 2_1: check list and rebuild if necessary
            #pragma omp single
            {
                is_list_valid=true;
            }
            is_list_valid_local=list_pair.isvalid();
            #pragma omp critical
            {
                is_list_valid = is_list_valid && is_list_valid_local;
            }
            #pragma omp barrier
            if(! is_list_valid)
            {
                local.refold_and_tag();               
                #pragma omp barrier
                #pragma omp single
                {
                    ENVIRON::sort();
                }
                local.generate_ghost();
                #pragma omp barrier
                #pragma omp single
                {
                    ENVIRON::tidyup_ghost();
                }
                local.update_ghost_pos<true>();
                #pragma omp barrier
                #pragma omp single
                {
                    ENVIRON::sort_ghost();
                }
                local.update_ghost_and_info();
                #pragma omp barrier

                // #pragma omp single
                // {
                //     ENVIRON::verify_thread();
                //     ENVIRON::verify_grid();
                // }

                list_pair.build<false>();
                // list_gr.build<false>();
                // list_gr.build_gr(&list_pair);
            }
            else
            {
                local.update_ghost_pos<false>();
            }

            // compute gr update DA & pot
            if(!(md_step%gr_interval))
            {
                #pragma omp barrier
                if(md_step%gr_interval_long) gr.tally<true>(&list_pair);
                else gr.tally(bin_list_gr,&local);
                #pragma omp barrier
                gr.reduce_local(comm_me);
                #pragma omp barrier
                sq.compute(comm_me);
                erg_exp_local=rmdf.compute_Qspace(comm_me);
                #pragma omp barrier
                rmdf.compute_Rspace(comm_me);
                if(f_log && gamma!=1) rmdf.compute_Potential(comm_me);
            }

            //step 2.2: compute force
            local.clear_vector_cs(*ENVIRON::f,3);
            #pragma omp barrier
            if(f_log)
            {
                erg_local=virial_local=0;
                bond.compute<true>(comm_me,&erg_local,&virial_local);
                #pragma omp barrier
                angle.compute<true>(comm_me,&erg_local,&virial_local);
                #pragma omp barrier
                dihedral.compute<true>(comm_me,&erg_local,&virial_local);
                #pragma omp barrier
#ifdef RMDF_IGNORE_SPECIAL
                pair.compute<true,true>(&list_pair,&erg_local,&virial_local);
#else
                pair.compute<true,false>(&list_pair,&erg_local,&virial_local);
#endif
                #pragma omp single
                {
                    erg=virial=0;
                    erg_exp=0;
                }
                #pragma omp critical
                {
                    erg+=erg_local; virial+=virial_local;
                    erg_exp+=erg_exp_local;
                }
            }
            else
            {
                bond.compute<false>(comm_me,nullptr,nullptr);
                #pragma omp barrier
                angle.compute<false>(comm_me,nullptr,nullptr);
                #pragma omp barrier
                dihedral.compute<false>(comm_me,nullptr,nullptr);
                #pragma omp barrier
#ifdef RMDF_IGNORE_SPECIAL
                pair.compute<false,true>(&list_pair,nullptr,nullptr);
#else
                pair.compute<false,false>(&list_pair,nullptr,nullptr);
#endif
            }

            //step 3: update V
            #pragma omp barrier
            verlet.updateV();

            // apply thermostat
            if(!(md_step%scale_interval))
            {
                ekin_local=verlet.computeKE();
                #pragma omp single
                {
                    ekin=0;
                }
                #pragma omp critical
                {
                    ekin+=ekin_local;
                }
#ifdef THERMO_ANDERSEN_FRAC
                #pragma omp single
                {
                    ENVIRON::initV(temp_start+(temp_end-temp_start)*md_step/total_steps,THERMO_ANDERSEN_FRAC);
                }
#else
                #pragma omp barrier
                #pragma omp single
                {
                    ke_scale=std::sqrt(random_gamma(temp_start+(temp_end-temp_start)*md_step/total_steps)/ekin);
                }
                verlet.scaleV(ke_scale);
#endif
            }

            // log result
            if(f_log)
            {
                #pragma omp single
                {
                    temperature=ekin/ENVIRON::natom*ENVIRON::ek_T;
                    press=(virial + ekin*2) /3*ENVIRON::ev_GPa/cub(ENVIRON::bl) ;
                    std::cout<<std::setw(7)<<md_step
                            <<std::setw(18)<<temperature
                            <<std::setw(18)<<press
                            <<std::setw(18)<<erg
                            <<std::setw(18)<<erg_exp
                            <<std::setw(10)<<list_pair.nbuild
                            // <<std::setw(18)<<(erg+ekin+erg_exp)
                            // <<std::setw(18)<<(erg+ekin)
                            <<"\n"; 
                }
            }
        }
    }
    std::cerr<<"n_ghost "<<ENVIRON::nghost<<"\n";

    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s %d",str_buf, &tpint4) != 2) END_PROGRAM("read error");
    if(*str_buf != '*') 
    {
        std::cerr<<"xyz = "<<str_buf<<"; append = "<<(bool)tpint4<<"\n";
        ENVIRON::dump(str_buf,tpint4);
    }

    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s %d",str_buf, &tpint4) != 2) END_PROGRAM("read error");
    if(*str_buf != '*') 
    {
        std::cerr<<"gr = "<<str_buf<<"; append = "<<(bool)tpint4<<"\n";
        gr.dump(str_buf,tpint4);
    }
    
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s %d",str_buf, &tpint4) != 2) END_PROGRAM("read error");
    if(*str_buf != '*') 
    {
        std::cerr<<"sq = "<<str_buf<<"; append = "<<(bool)tpint4<<"\n";
        sq.dump(str_buf,tpint4);
    }

    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s %d " FMT_NUMTYPE " %d",str_buf, &tpint4, &tpflt, &tpint) != 4) END_PROGRAM("read error");
    if(*str_buf != '*') 
    {
        std::cerr<<"sqd = "<<str_buf<<"; append = "<<(bool)tpint4<<"\n";
        SQD sqd(tpint,tpflt);
        #pragma omp parallel num_threads(ENVIRON::NUM_THREAD)
        {
            sqd.compute(omp_get_thread_num());
        }
        sqd.dump(str_buf,tpint4);
    }

    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s %d",str_buf, &tpint4) != 2) END_PROGRAM("read error");
    if(*str_buf != '*') 
    {
        std::cerr<<"sqx = "<<str_buf<<"; append = "<<(bool)tpint4<<"\n";
        rmdf.dumpSQ(str_buf,tpint4);
    }

    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s %d",str_buf, &tpint4) != 2) END_PROGRAM("read error");
    if(*str_buf != '*') 
    {
        std::cerr<<"force = "<<str_buf<<"; append = "<<(bool)tpint4<<"\n";
#ifdef RMDF_IGNORE_SPECIAL
        pair.dumpForce(str_buf,true,tpint4);
#else
        pair.dumpForce(str_buf,false,tpint4);
#endif
    }

    fclose(fp);

    return 0;
}

int nnp_train(int argc, const char *argv[])
{
    if(argc!=2) END_PROGRAM("invalid arg");

    char file_buf[SZ_FBF],str_buf[SZ_FBF],str_buf2[SZ_FBF],str_buf3[SZ_FBF];
    FILE *fp=fopen(argv[1],"r");
    int tpint,tpint2,tpint3,tpint4;

    std::cerr<<"NUM_THREAD = "<<ENVIRON::NUM_THREAD<<"\n";

    // initialization 
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s",str_buf) != 1) END_PROGRAM("read error");
    ENVIRON::initialize(str_buf);

    random_set_gamma_ndof(ENVIRON::natom*3);

    // setup cutoff & potential
    numtype pair_cutoff=8.,skin=0;
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,FMT_NUMTYPE " ", &pair_cutoff) != 1) END_PROGRAM("read error");
    std::cerr<<"cutoff "<<pair_cutoff<<" skin "<<skin<<"\n";
    const numtype comm_cutoff=skin+pair_cutoff;

    LOCAL::set_ghost_lim(comm_cutoff);
    ENVIRON::init_grid(0.5*(skin+pair_cutoff),comm_cutoff);

    int **bin_list_pair=NEIGHLIST::build_bin_list(pair_cutoff+skin,126);

    //compute n_bounds
    MOLECULE *imol;
    tpint2=tpint3=tpint4=0;
    for(int im=0;im<ENVIRON::nmol;++im)
    {   
        imol=ENVIRON::moltype+ENVIRON::mol_type[im];
        tpint2+= imol->n_bond;
        tpint3+= imol->n_angle;
        tpint4+= imol->n_dihedral;
    }

    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%d",&tpint)!=1)  END_PROGRAM("read error");
    BOND bond(tpint,tpint2);
    bond.autoGenerate();

    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%d",&tpint)!=1)  END_PROGRAM("read error");
    ANGLE angle(tpint,tpint3);
    angle.autoGenerate();

    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%d",&tpint)!=1)  END_PROGRAM("read error");
    DIHEDRAL dihedral(tpint,tpint4);
    dihedral.autoGenerate();

    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s",str_buf) != 1) END_PROGRAM("read error");
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s",str_buf2) != 1) END_PROGRAM("read error");
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s",str_buf3) != 1) END_PROGRAM("read error");
    std::cerr<<"NNP: "<<str_buf<<"\n";
    std::cerr<<"WGT: "<<str_buf2<<"\n";
    std::cerr<<"OPT: "<<str_buf3<<"\n";
    NNP_TRAIN_TF nnp(str_buf,str_buf2,str_buf3,pair_cutoff,&bond,&angle,&dihedral);

    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s",str_buf) != 1) END_PROGRAM("read error");
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s",str_buf2) != 1) END_PROGRAM("read error");
    std::cerr<<"DAT: "<<str_buf<<"\n";
    int n_dat=nnp.load_train_info(str_buf,str_buf2);

    float erg_true=0, loss=0, lr0, lr, alpha;
    int num_epochs;
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%f %f %f %d",&lr0,&lr,&alpha,&num_epochs) != 4) END_PROGRAM("read error");
    std::cerr<<"lr = "<<lr0<<" | "<<lr<<"; alpha = "<<alpha<<"\n"<<"epochs = "<<num_epochs<<"\n";

    double erg_true_mean=0, erg_pred_mean=0;

    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s",str_buf) != 1) END_PROGRAM("read error");
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s",str_buf2) != 1) END_PROGRAM("read error");
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s",str_buf3) != 1) END_PROGRAM("read error");
    std::cerr<<"LOS: "<<str_buf<<"\n";
    std::cerr<<"ERG: "<<str_buf2<<"\n";
    std::cerr<<"FRC: "<<str_buf3<<"\n";
    FILE *fp_erg,*fp_frc,*fp_los=fopen(str_buf,"ab");


    #pragma omp parallel num_threads(ENVIRON::NUM_THREAD)
    {
        if(omp_get_num_threads()!=ENVIRON::NUM_THREAD) END_PROGRAM("omp failed to start");
        int comm_me=omp_get_thread_num();
        LOCAL local(comm_me);

        NEIGHLIST list_pair(comm_me,pair_cutoff,skin,NEIGH_SZ,bin_list_pair);

        // init force
        #pragma omp barrier
        // first epoch is for pre-training: alpha is set to zero and erg diff is accumulated
        // on the begining of the second epoch, the bias of final layer is updated
        for(int epoch=0; epoch<num_epochs;++epoch)
        {
            #pragma omp single
            {
                if(epoch==1)
                {
                    erg_pred_mean/=n_dat;
                    erg_true_mean/=n_dat;
                    *(float*)nnp.tfcall->inputs[nnp.n_wts-1]-=erg_pred_mean/(ENVIRON::nmol);
                    *(float*)nnp.tfcall->inputs[nnp.n_wts+13]=alpha/ENVIRON::natom;
                    lr0=lr;
                }
                if(epoch==num_epochs-1)
                {
                    fp_erg=fopen(str_buf2,"wb");
                    fp_frc=fopen(str_buf3,"wb");
                }
                loss=0;
                shuffle(nnp.fnames_shuffled,n_dat);
            }
            for(int idat=0; idat<n_dat; ++idat)
            {
                #pragma omp single
                {
                    ENVIRON::unsort();
                    if(epoch)
                    {
                        erg_true=nnp.ry_kcal_mol*(nnp.load_data(nnp.fnames_shuffled[idat])-erg_true_mean);
                    }
                    else
                    {
                        erg_true_mean+=nnp.load_data(nnp.fnames_shuffled[idat]);
                    }
                    // printMatrix(ENVIRON::f_mol,14,3,"f",'p');
                }
                local.refold_and_tag();               
                #pragma omp barrier
                #pragma omp single
                {
                    ENVIRON::sort();
                }
                local.generate_ghost();
                #pragma omp barrier
                #pragma omp single
                {
                    ENVIRON::tidyup_ghost();
                }
                local.update_ghost_pos<true>();
                #pragma omp barrier
                #pragma omp single
                {
                    ENVIRON::sort_ghost();
                }
                local.update_ghost_and_info();
                #pragma omp barrier
                nnp.num_neigh_local[comm_me]=list_pair.build<true>();
                #pragma omp barrier
                #pragma omp single
                {
                    nnp.allocate_neilist();
                }
                nnp.gather_neilist(&list_pair);
#ifndef NNP_USE_X_MOL
                #pragma omp barrier
                nnp.update_intra(comm_me,&bond,&angle,&dihedral);
#endif
                #pragma omp barrier
                #pragma omp single
                {
                    loss+=nnp.nnEval(erg_true);
                    if(!epoch) erg_pred_mean+=**(nnp.tfcall->outputs+(nnp.n_output-2));
                    if(epoch==num_epochs-1)
                    {
                        fwrite(&erg_true,sizeof(float),1,fp_erg);
                        fwrite(*(nnp.tfcall->outputs+(nnp.n_output-2)),sizeof(float),1,fp_erg);
                        if(idat%100 == 0)
                        {
                            fwrite(*(nnp.tfcall->inputs+(nnp.n_input-3)),sizeof(float),ENVIRON::natom*3,fp_frc);
                            fwrite(*(nnp.tfcall->outputs+(nnp.n_output-1)),sizeof(float),ENVIRON::natom*3,fp_frc);
                        }
                    }
                }
                nnp.apply_gradient(lr0,comm_me);
                #pragma omp barrier
                #pragma omp single
                {
                    // for(int jj=nnp.n_wts;jj<nnp.n_wts+8;++jj)
                    // {
                    //     std::cerr<<jj<<" "<<nnp.sz_in[jj]<<" -> ";
                    //     for(int kk=0;kk<20;++kk)
                    //         std::cerr<<((int*)nnp.tfcall->inputs[jj])[kk]<<" ";
                    //     std::cerr<<"\n";
                    // }
                    // for(int jj=0;jj<28;++jj) std::cerr<<ENVIRON::atom_id[jj]<<" ";
                    // for(int jj=0;jj<28;++jj) std::cerr<<nnp.inv_atom_id[jj]<<" ";
                    // std::cerr<<**(nnp.tfcall->outputs+(nnp.n_output-2))<<" "<<loss<<"\n";
                    nnp.tfcall->clearOutput();
                }
            }
            #pragma omp single
            {
                loss/=(n_dat*ENVIRON::natom);
                std::cout<<"iter "<<nnp.n_update<<": loss = "<<(loss)<<"\n";
                if(epoch) fwrite(&loss,sizeof(float),1,fp_los);
            }
        }
    }

    fclose(fp_los);
    fclose(fp_erg);
    fclose(fp_frc);

    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s",str_buf2) != 1) END_PROGRAM("read error");
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s",str_buf3) != 1) END_PROGRAM("read error");
    std::cerr<<"WGT: "<<str_buf2<<"\n";
    std::cerr<<"OPT: "<<str_buf3<<"\n";
    nnp.save_wts(str_buf2,str_buf3);

    return 0;
}

// #define TF_DESCRIPTOR

int nnp_run(int argc, const char *argv[])
{
    if(argc!=2) END_PROGRAM("invalid arg");

    char file_buf[SZ_FBF],str_buf[SZ_FBF],str_buf2[SZ_FBF],*strbufs[]={str_buf,str_buf2};
    FILE *fp=fopen(argv[1],"r");
    int tpint,tpint2,tpint3,tpint4;
    numtype tpflt;

    std::cerr<<"NUM_THREAD = "<<ENVIRON::NUM_THREAD<<"\n";

    // initialization 
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s",str_buf) != 1) END_PROGRAM("read error");
    ENVIRON::initialize(str_buf);

    random_set_gamma_ndof(ENVIRON::natom*3);

    // setup cutoff & potential
    numtype pair_cutoff=8.,skin=1.,gr_cutoff=9.;
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,FMT_NUMTYPE " " FMT_NUMTYPE " " FMT_NUMTYPE, &pair_cutoff, &skin, &gr_cutoff) != 3) END_PROGRAM("read error");
    std::cerr<<"cutoff "<<pair_cutoff<<" "<<gr_cutoff<<" skin "<<skin<<"\n";
    const numtype comm_cutoff=skin+std::max(gr_cutoff,pair_cutoff);

    LOCAL::set_ghost_lim(comm_cutoff);
    ENVIRON::init_grid(0.5*(skin+pair_cutoff),comm_cutoff);

    int **bin_list_pair=NEIGHLIST::build_bin_list(pair_cutoff+skin,126);
    int **bin_list_gr=NEIGHLIST::build_bin_list(gr_cutoff+skin,BIN_NEIGH);

    //compute n_bounds
    MOLECULE *imol;
    tpint2=tpint3=tpint4=0;
    for(int im=0;im<ENVIRON::nmol;++im)
    {   
        imol=ENVIRON::moltype+ENVIRON::mol_type[im];
        tpint2+= imol->n_bond;
        tpint3+= imol->n_angle;
        tpint4+= imol->n_dihedral;
    }
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%d",&tpint)!=1)  END_PROGRAM("read error");
    BOND bond(tpint,tpint2);
    bond.autoGenerate();

    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%d",&tpint)!=1)  END_PROGRAM("read error");
    ANGLE angle(tpint,tpint3);
    angle.autoGenerate();

    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%d",&tpint)!=1)  END_PROGRAM("read error");
    DIHEDRAL dihedral(tpint,tpint4);
    dihedral.autoGenerate();

    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s",str_buf) != 1) END_PROGRAM("read error");
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s",str_buf2) != 1) END_PROGRAM("read error");
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%d",&tpint) != 1) END_PROGRAM("read error");

#ifdef TF_DESCRIPTOR
    NNP_DEPLOY_TF nnp(str_buf,pair_cutoff,&bond,&angle,&dihedral);
    std::cerr<<"NNP: "<<str_buf<<"\n";
#else
    // NNPOTENTIAL nnp(pair_cutoff,str_buf,str_buf2,&bond,&angle,&dihedral);
    NNPOTENTIAL_TBL nnp(pair_cutoff,tpint,str_buf,str_buf2,&bond,&angle,&dihedral);
    std::cerr<<"NNP: "<<str_buf<<"\n";
    std::cerr<<"WGT: "<<str_buf2<<"\n";
#endif

    //grsq
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%d %d " FMT_NUMTYPE,&tpint,&tpint2,&tpflt)!=3)  END_PROGRAM("read error");
    GR gr(tpint,gr_cutoff,pair_cutoff);
    SQ sq(tpint2,tpflt,&gr);
    numtype gamma;
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,FMT_NUMTYPE " " FMT_NUMTYPE,&tpflt,&gamma)!=2)  END_PROGRAM("read error");
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%d %n",&tpint,&tpint3)!=1)  END_PROGRAM("read error");
    if(tpint>2) END_PROGRAM("max n_sq is 2");
    for(int i=0;i<tpint;++i)
    {
        if(sscanf(file_buf+tpint3,"%s %n",strbufs[i],&tpint2)!=1)  END_PROGRAM("read error");
        tpint3+=tpint2;
    }
#ifndef TF_DESCRIPTOR
    nnp.allocateTable();
    RMDF rmdf(tpint,strbufs,&sq,tpflt,gamma,nullptr,&nnp);
#endif
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s",str_buf) != 1) END_PROGRAM("read error");
#ifndef TF_DESCRIPTOR
    rmdf.loadDeltaSQ(str_buf);
#endif

    numtype erg=INFINITY,virial=INFINITY,ekin=INFINITY,temperature=INFINITY,press=INFINITY,erg_exp=INFINITY;
    bool is_list_valid=false;
#ifdef THERMO_ANDERSEN_FRAC
    std::cerr<<"Andersen thermo: "<<THERMO_ANDERSEN_FRAC<<"\n";
#else
    std::cerr<<"CSVR thermo\n";
    numtype ke_scale;
#endif
    numtype temp_start,temp_end;
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,FMT_NUMTYPE " " FMT_NUMTYPE,&temp_start,&temp_end) != 2) END_PROGRAM("read error");
    std::cerr<<"temp "<<temp_start<<" -> "<<temp_end<<"\n";
    temp_start/=300; temp_end/=300;
    int total_steps=10000, log_interval=1000, scale_interval=10, gr_interval=-1, gr_interval_long=-1;
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%d %d %d %d %d",&total_steps,&log_interval,&scale_interval,&gr_interval,&gr_interval_long) < 3) END_PROGRAM("read error");
    gr_interval=gr_interval<0? total_steps:gr_interval; gr_interval_long=gr_interval_long<0? total_steps:gr_interval_long; 
    if((total_steps%log_interval) || (total_steps%scale_interval)) END_PROGRAM("bad steps");
    std::cerr<<"MD steps: total "<<total_steps<<"; log "<<log_interval<<"; scale "<<scale_interval<<"; gr "<<gr_interval<<"; long "<<gr_interval_long<<"\n";

    ENVIRON::initV(temp_start);
    // setup MD info
    VERLET::setDt(-1);

    std::cerr<<std::setw(7)<<"md_step"
            <<std::setw(18)<<"temp [K]"
            <<std::setw(18)<<"press [GPa]"
            <<std::setw(18)<<"erg [kcal/mol]"
            <<std::setw(18)<<"dif [kcal/mol]"
            // <<std::setw(18)<<"etot [kcal/mol]"
            <<std::setw(10)<<"nbuild"
            <<"\n";
    #pragma omp parallel num_threads(ENVIRON::NUM_THREAD)
    {
        if(omp_get_num_threads()!=ENVIRON::NUM_THREAD) END_PROGRAM("omp failed to start");
        bool f_log;
        int comm_me=omp_get_thread_num();
        LOCAL local(comm_me);

        NEIGHLIST list_pair(comm_me,pair_cutoff,skin,NEIGH_SZ,bin_list_pair);

        VERLET verlet(comm_me);
        numtype erg_local=INFINITY;
#ifndef TF_DESCRIPTOR
        numtype virial_local=INFINITY;
        numtype erg_exp_local=INFINITY;
#endif
        numtype ekin_local=INFINITY;
        bool is_list_valid_local;

        // init force
        local.clear_vector(ENVIRON::f,3);
        #pragma omp barrier
        verlet.updateX();
        #pragma omp barrier
        #pragma omp single
        {
            VERLET::setDt(1);
        }
        // start MD simulation
        for(int md_step=0;md_step<=total_steps;++md_step)
        {
            f_log = !(md_step%log_interval);
            // step 1: update X
            verlet.updateX();

            //step 2_1: check list and rebuild if necessary
            #pragma omp single
            {
                is_list_valid=true;
            }
            is_list_valid_local=list_pair.isvalid();
            #pragma omp critical
            {
                is_list_valid = is_list_valid && is_list_valid_local;
            }
            #pragma omp barrier
            if(! is_list_valid)
            {
                local.refold_and_tag();               
                #pragma omp barrier
                #pragma omp single
                {
                    ENVIRON::sort();
                }
                local.generate_ghost();
                #pragma omp barrier
                #pragma omp single
                {
                    ENVIRON::tidyup_ghost();
                }
                local.update_ghost_pos<true>();
                #pragma omp barrier
                #pragma omp single
                {
                    ENVIRON::sort_ghost();
                }
                local.update_ghost_and_info();
                #pragma omp barrier
                // #pragma omp single
                // {
                //     ENVIRON::verify_thread();
                //     ENVIRON::verify_grid();
                // }
#ifdef TF_DESCRIPTOR
                nnp.num_neigh_local[comm_me]=list_pair.build<true>();
                #pragma omp barrier
                #pragma omp single
                {
                    nnp.allocate_neilist();
                }
                nnp.gather_neilist(&list_pair);
#ifndef NNP_USE_X_MOL
                #pragma omp barrier
                nnp.update_intra(comm_me,&bond,&angle,&dihedral);
#endif
#else
                list_pair.build<false>();
#endif
                // list_gr.build<false>();
                // list_gr.build_gr(&list_pair);
            }
#ifndef TF_DESCRIPTOR
            else
            {
                local.update_ghost_pos<false>();
            }
#endif

            // compute gr update DA & pot
            if(!(md_step%gr_interval))
            {
                #pragma omp barrier
                if(md_step%gr_interval_long) gr.tally<true>(&list_pair);
                else gr.tally(bin_list_gr,&local);
                #pragma omp barrier
                gr.reduce_local(comm_me);
                #pragma omp barrier
                sq.compute(comm_me);
#ifndef TF_DESCRIPTOR
                erg_exp_local=rmdf.compute_Qspace(comm_me);
                #pragma omp barrier
                rmdf.compute_Rspace_NNP(comm_me);
                if(f_log && gamma!=1) rmdf.compute_Potential_NNP(comm_me);
#endif
            }
            
            //step 2.2: compute force
            local.clear_vector_cs(*ENVIRON::f,3);
            #pragma omp barrier
            if(f_log)
            {
#ifdef TF_DESCRIPTOR
                nnp.compute_pair(&list_pair,&erg_local);
                #pragma omp barrier
                #pragma omp single
                {
                    nnp.nnEval();
                    erg=**(nnp.tfcall->outputs);
                }
                nnp.evalFinalize(&local);
                #pragma omp barrier
                #pragma omp single
                {
                    nnp.tfcall->clearOutput();
                }
#else
                local.clear_vector_cs(*nnp.descriptor,nnp.n_descriptor);
                erg_local=virial_local=0;
                #pragma omp barrier
                nnp.computeDescriptor(&list_pair);
                dihedral.compute<true>(comm_me,&erg_local,&virial_local);
                #pragma omp barrier
                angle.compute<true>(comm_me,&erg_local,&virial_local);
                #pragma omp single
                {
                    nnp.nnEval();
                }
                nnp.evalFinalize(&local);
                #pragma omp barrier
                bond.compute<true>(comm_me,&erg_local,&virial_local);
                #pragma omp barrier
                nnp.compute<true>(&list_pair,&erg_local,&virial_local);
                #pragma omp barrier
                #pragma omp single
                {
                    erg=**(nnp.tfcall->outputs);
                    erg_exp=virial=0;
                    nnp.tfcall->clearOutput();
                }
                #pragma omp critical
                {
                    erg+=erg_local;
                    // std::cerr<<comm_me<<" "<<erg_local<<"\n";
                    virial+=virial_local;
                    erg_exp+=erg_exp_local;
                }
#endif
            }
            else
            {
#ifdef TF_DESCRIPTOR
                nnp.compute_pair(&list_pair,&erg_local);
                #pragma omp barrier
                #pragma omp single
                {
                    nnp.nnEval();
                    // erg=**(nnp.tfcall->outputs);
                }
                nnp.evalFinalize(&local);
                #pragma omp barrier
                #pragma omp single
                {
                    nnp.tfcall->clearOutput();
                }
#else
                local.clear_vector_cs(*nnp.descriptor,nnp.n_descriptor);
                #pragma omp barrier
                nnp.computeDescriptor(&list_pair);
                dihedral.compute<false>(comm_me,nullptr,nullptr);
                #pragma omp barrier
                angle.compute<false>(comm_me,nullptr,nullptr);
                #pragma omp single
                {
                    nnp.nnEval();
                }
                nnp.evalFinalize(&local);
                #pragma omp barrier
                nnp.compute<false>(&list_pair,nullptr,nullptr);
                #pragma omp barrier
                bond.compute<false>(comm_me,nullptr,nullptr);
                #pragma omp single
                {
                    nnp.tfcall->clearOutput();
                }
#endif
            }

            //step 3: update V
            #pragma omp barrier
            verlet.updateV();

            // apply thermostat
            if(!(md_step%scale_interval))
            {
                ekin_local=verlet.computeKE();
                #pragma omp single
                {
                    ekin=0;
                }
                #pragma omp critical
                {
                    ekin+=ekin_local;
                }
#ifdef THERMO_ANDERSEN_FRAC
                #pragma omp single
                {
                    ENVIRON::initV(temp_start+(temp_end-temp_start)*md_step/total_steps,THERMO_ANDERSEN_FRAC);
                }
#else
                #pragma omp barrier
                #pragma omp single
                {
                    ke_scale=std::sqrt(random_gamma(temp_start+(temp_end-temp_start)*md_step/total_steps)/ekin);
                }
                verlet.scaleV(ke_scale);
#endif
            }

            // log result
            if(f_log)
            {
                #pragma omp single
                {
                    temperature=ekin/ENVIRON::natom*ENVIRON::ek_T;
                    press=(virial + ekin*2) /3*ENVIRON::ev_GPa/cub(ENVIRON::bl) ;
                    std::cout<<std::setw(7)<<md_step
                            <<std::setw(18)<<temperature
                            <<std::setw(18)<<press
                            <<std::setw(18)<<erg
                            <<std::setw(18)<<erg_exp
                            <<std::setw(10)<<list_pair.nbuild
                            // <<std::setw(18)<<(erg+ekin)
                            <<"\n"; 
                }
            }
        }
    }
    std::cerr<<"n_ghost "<<ENVIRON::nghost<<"\n";
    
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s %d",str_buf, &tpint4) != 2) END_PROGRAM("read error");
    if(*str_buf != '*') 
    {
        std::cerr<<"xyz = "<<str_buf<<"; append = "<<(bool)tpint4<<"\n";
        ENVIRON::dump(str_buf,tpint4);
    }

    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s %d",str_buf, &tpint4) != 2) END_PROGRAM("read error");
    if(*str_buf != '*') 
    {
        std::cerr<<"gr = "<<str_buf<<"; append = "<<(bool)tpint4<<"\n";
        gr.dump(str_buf,tpint4);
    }
    
    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s %d",str_buf, &tpint4) != 2) END_PROGRAM("read error");
    if(*str_buf != '*') 
    {
        std::cerr<<"sq = "<<str_buf<<"; append = "<<(bool)tpint4<<"\n";
        sq.dump(str_buf,tpint4);
    }

    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s %d " FMT_NUMTYPE " %d",str_buf, &tpint4, &tpflt, &tpint) != 4) END_PROGRAM("read error");
    if(*str_buf != '*') 
    {
        std::cerr<<"sqd = "<<str_buf<<"; append = "<<(bool)tpint4<<"\n";
        SQD sqd(tpint,tpflt);
        #pragma omp parallel num_threads(ENVIRON::NUM_THREAD)
        {
            sqd.compute(omp_get_thread_num());
        }
        sqd.dump(str_buf,tpint4);
    }

    fgets_non_empty(fp,file_buf); string_get_env(file_buf);
    if(sscanf(file_buf,"%s %d",str_buf, &tpint4) != 2) END_PROGRAM("read error");
    if(*str_buf != '*') 
    {
        std::cerr<<"sqx = "<<str_buf<<"; append = "<<(bool)tpint4<<"\n";
        rmdf.dumpSQ(str_buf,tpint4);
    }
    
    return 0;
}


