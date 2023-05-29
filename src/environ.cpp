#include "environ.h"
#include "memory.h"
#include "mathlib.h"
#include "util.h"
#include <iostream>
#include <cstring>

// #define CHECK_BINNING

numtype ENVIRON::distsq(numtype *x1, numtype *x2)
{
    numtype dx,rsq=0;
    for (int i=0;i<3;++i)
    {
        dx=std::fabs(x1[i]-x2[i]);
        dx-= static_cast<int>(dx*r_bl + 0.5)*bl;
        rsq+=dx*dx;
    }
    return rsq;
}

void ENVIRON::initialize(const char *input_file)
{
    char buf[SZ_FBF],buf2[SZ_FBF];
    std::cerr<<"reading "<<input_file<<"\n";
    FILE *fp=fopen(input_file,"r");

    // molecule type
    fgets_non_empty(fp,buf);
    if(sscanf(buf,"%d %d",&nmoltype,&nmol)!=2) END_PROGRAM("read error");
    std::cerr<<"nmol = "<<nmol<<", "<<nmoltype<<" type(s)\n";
    moltype=new MOLECULE[nmoltype];
    for(int i=0;i<nmoltype;++i)
    {
        fgets_non_empty(fp,buf);
        sscanf(buf,"%s",buf2);
        moltype[i].from_file(buf2);
    }

    //sys info
    fgets_non_empty(fp,buf);
    if(sscanf(buf,"%d %d " FMT_NUMTYPE ,&ntype, &natom, &bl)!=3) END_PROGRAM("read error");
    std::cerr<<"natom = "<<(ntot=natom)<<", "<<ntype<<" type(s)\n";
    set_bl(bl);
    std::cerr<<"bl = "<<bl<<"\n";

    LOCAL::ghost_capacity_local=ghost_capacity_frac*natom/NUM_THREAD*local_capacity_frac;
    LOCAL::local_capacity=natom*local_capacity_frac/NUM_THREAD;
    int tot_capa=natom+NUM_THREAD*LOCAL::ghost_capacity_local;
    int mcp=std::max(natom,tot_capa-natom);

    //these three use different size becuase it is also used for ghost sorting
    create1DArray(__typ,mcp);
    create2DArray(__v,mcp,3);

    create1DArray(__q,natom);
    create1DArray(__r_m,natom);
    create2DArray(__x,natom,3);
    create2DArray(__f_prev,natom,3);
    create1DArray(__atom_id,natom);
    create1DArray(__mol_id,natom);
    create1DArray(__intra_id,natom);
    create1DArray(__x_grd,natom);

    create1DArray(typ,tot_capa);
    create1DArray(q,tot_capa);
    create1DArray(r_m,tot_capa);
    create2DArray(x,tot_capa,3);
    create2DArray(v,natom,3);
    create2DArray(f,natom,3);
    create2DArray(f_prev,natom,3);
    create1DArray(atom_id,tot_capa);
    create1DArray(mol_id,tot_capa);
    create1DArray(intra_id,tot_capa);

    x_mol=new numtype*[natom];
    f_mol=new numtype*[natom];
    typ_mol=new int[natom];

    create1DArray(x_thr,tot_capa); //ghost atoms is asigned to thread -1
    for(int i=natom;i<tot_capa;++i) x_thr[i]=-1;

    create1DArray(__x_thr,natom);
    create1DArray(x_grd,natom);
    create1DArray(x_grd_ghost,NUM_THREAD*LOCAL::ghost_capacity_local);

    create1DArray(mol_repre_id,nmol);
    create1DArray(mol_type,nmol);

    create1DArray(n_ghost_local,NUM_THREAD);
    create3DArray(g_ghost_local,NUM_THREAD,LOCAL::ghost_capacity_local,3);
    create2DArray(i_ghost_local,NUM_THREAD,LOCAL::ghost_capacity_local);
    g_ghost=g_ghost_local[0];
    i_ghost=i_ghost_local[0];

    create1DArray(natom_t,NUM_THREAD);
    create1DArray(natom_g,num_grid);

    t_rdx=NUM_X/bl;t_rdy=NUM_Y/bl;t_rdz=NUM_Z/bl;

    int intra=0,imol=0,mt=0,mt_prev=0;
    for(int idd=0;idd<natom;++idd)
    {
        fgets_non_empty(fp,buf);
        if(sscanf(buf,"%d %d " FMT_NUMTYPE " "  FMT_NUMTYPE " "  FMT_NUMTYPE " "  FMT_NUMTYPE " "  FMT_NUMTYPE,
                    &mt,typ+idd,r_m+idd,q+idd,x[idd],x[idd]+1,x[idd]+2) != 7) END_PROGRAM("read error");
        if(typ[idd]>=ntype || typ[idd]<0) END_PROGRAM("read error");
        atom_id[idd]=idd;
        intra_id[idd]=intra;
        mol_id[idd]=imol;
        if(!intra) 
        {
            mol_repre_id[imol]=idd;
            mol_type[imol]=mt;
        }
        else if(mt!=mt_prev)  END_PROGRAM("read error");
        r_m[idd]=1/r_m[idd];

        if(moltype[mt].n_atom == ++intra) {++imol; intra=0;}
        mt_prev=mt;
    }
    if(imol!=nmol) END_PROGRAM("read error");
    
    fclose(fp);

    memcpy(x_mol,x,natom*sizeof(numtype*));
    memcpy(f_mol,f,natom*sizeof(numtype*));
    memcpy(typ_mol,typ,natom*sizeof(int));

    create1DArray(typecount,ntype);
    create2DArray(pairtype,ntype,ntype);
    npair=0;
    for(int i=0;i<ntype;++i) 
    {
        typecount[i]=0;
        for(int j=i;j<ntype;++j)
            pairtype[i][j]=pairtype[j][i]=npair++;
    }
    for(int i=0;i<natom;++i) ++typecount[typ[i]];

}

void ENVIRON::init_grid(numtype grid_length, numtype ghost_cutof)
{
    g_rdx=1/(g_dx=grid_length);
    g_offset=ghost_cutof+h_bl;
    num_gx=std::ceil((g_offset*2)*g_rdx);
    std::cerr<<"grid dx = "<<g_dx<<"; n = "<<num_gx<<"\n";
    num_grid=num_gx*(num_gxsq=num_gx*num_gx);
    num_gxm1=num_gx-1;
    create1DArray(grid_start,num_grid);
    create1DArray(grid_end,num_grid);
    create1DArray(grid_start_ghost,num_grid);
    create1DArray(grid_end_ghost,num_grid);
}

int ENVIRON::tidyup_ghost()
{
    nghost=n_ghost_local[0];
    int ng;
    int *igl;
    numtype** ggl;
    for(int i=1;i<NUM_THREAD;++i)
    {
        ng=n_ghost_local[i];
        igl=i_ghost_local[i];
        ggl=g_ghost_local[i];
        for(int j=0;j<ng;++j)
        {
            i_ghost[nghost]=igl[j];
            memcpy(g_ghost[nghost],ggl[j],3*sizeof(numtype));
            ++nghost;
        }
    }
    return ntot=nghost+natom;
}

void ENVIRON::dump(const char *xyzfile, bool append)
{
    FILE *fp=fopen(xyzfile,append?"a":"w");
    // int na=all? ntot : natom;
    int na=natom;
    fprintf(fp,"%d\nx y z w %.15e %.15e\n", na, -h_bl, h_bl);
    for(int i=0;i<na;++i)
        fprintf(fp,"T%d %.15e %.15e %.15e\n", typ_mol[i], x_mol[i][0],x_mol[i][1],x_mol[i][2]);
        // fprintf(fp,"%s %.15e %.15e %.15e %d\n", elms[typ[i]], x[i][0],x[i][1],x[i][2],x_grd[i]);
    fclose(fp);
}

void ENVIRON::initV(numtype tempera_rel_300,numtype fraction)
{
    tempera_rel_300*=300;
    numtype *vv, fc;
    for(int i=0;i<natom;++i)
    {
        if(random_uniform()<fraction)
        {
            vv=v[i];
            fc=std::sqrt(tempera_rel_300*r_m[i])*ENVIRON::kT_m_sqrt;
            vv[0]=random_gaussian(fc);
            vv[1]=random_gaussian(fc);
            vv[2]=random_gaussian(fc);
        }
    }   
}

LOCAL::LOCAL(int tid)
{
    thread_id=tid;
    i_local_start=ENVIRON::natom*thread_id/ENVIRON::NUM_THREAD;
    i_local_end=ENVIRON::natom*(thread_id+1)/ENVIRON::NUM_THREAD;

    // int tot_capa=ENVIRON::natom+ENVIRON::NUM_THREAD*ghost_capacity_local;
    // create1DArray(q_local,tot_capa);
    // create1DArray(typ_local,tot_capa);
    // x_local= new numtype*[tot_capa];
    // f_local= new numtype*[local_capacity];
    // create1DArray(mol_id_local,tot_capa);
    // create1DArray(intra_id_local,tot_capa);

    // memcpy(x_local,ENVIRON::x,tot_capa*sizeof(numtype*));
}

LOCAL::~LOCAL()
{
    // destroy1DArray(q_local);
    // destroy1DArray(typ_local);
    // destroy1DArray(mol_id_local);
    // destroy1DArray(intra_id_local);/
    // delete[] x_local;
    // delete[] f_local;
}

void LOCAL::refold_and_tag()
{
    numtype *xl;
    int _tix,_tiy,_tiz;
    for(int i=i_local_start;i<i_local_end;++i)
    {
        xl=ENVIRON::x[i];

        _tix=(ENVIRON::h_bl + (xl[0]-=std::nearbyint(xl[0]*ENVIRON::r_bl)*ENVIRON::bl))*ENVIRON::t_rdx;
        _tiy=(ENVIRON::h_bl + (xl[1]-=std::nearbyint(xl[1]*ENVIRON::r_bl)*ENVIRON::bl))*ENVIRON::t_rdy;
        _tiz=(ENVIRON::h_bl + (xl[2]-=std::nearbyint(xl[2]*ENVIRON::r_bl)*ENVIRON::bl))*ENVIRON::t_rdz;
#ifdef CHECK_BINNING
        if(_tix<0) {_tix=0;std::cerr<<"w";} else if (_tix>=ENVIRON::NUM_X){_tix=ENVIRON::NUM_X-1;std::cerr<<"w";}
        if(_tiy<0) {_tiy=0;std::cerr<<"w";} else if (_tiy>=ENVIRON::NUM_Y){_tiy=ENVIRON::NUM_Y-1;std::cerr<<"w";}
        if(_tiz<0) {_tiz=0;std::cerr<<"w";} else if (_tiz>=ENVIRON::NUM_Z){_tiz=ENVIRON::NUM_Z-1;std::cerr<<"w";}
#endif
        ENVIRON::x_thr[i]=_tix*(ENVIRON::NUM_Y*ENVIRON::NUM_Z)+_tiy*(ENVIRON::NUM_Z)+_tiz;

        _tix=(ENVIRON::g_offset + xl[0])*ENVIRON::g_rdx;
        _tiy=(ENVIRON::g_offset + xl[1])*ENVIRON::g_rdx;
        _tiz=(ENVIRON::g_offset + xl[2])*ENVIRON::g_rdx;
#ifdef CHECK_BINNING
        if(_tix<0) {_tix=0;std::cerr<<"g";} else if (_tix>=ENVIRON::num_gx){_tix=ENVIRON::num_gxm1;std::cerr<<"g";}
        if(_tiy<0) {_tiy=0;std::cerr<<"g";} else if (_tiy>=ENVIRON::num_gx){_tiy=ENVIRON::num_gxm1;std::cerr<<"g";}
        if(_tiz<0) {_tiz=0;std::cerr<<"g";} else if (_tiz>=ENVIRON::num_gx){_tiz=ENVIRON::num_gxm1;std::cerr<<"g";}
#endif
        ENVIRON::x_grd[i]=_tix*ENVIRON::num_gxsq+_tiy*ENVIRON::num_gx+_tiz;
    }
}
void ENVIRON::unsort()
{
    memcpy(__typ,typ,natom*sizeof(int));
    memcpy(__x_thr,x_thr,natom*sizeof(int));
    memcpy(__q,q,    natom*sizeof(numtype));
    memcpy(__r_m,r_m,natom*sizeof(numtype));
    memcpy(__atom_id,atom_id,natom*sizeof(int));
    memcpy(__mol_id,mol_id,natom*sizeof(int));
    memcpy(__intra_id,intra_id,natom*sizeof(int));
    memcpy(__x_grd,x_grd,natom*sizeof(int));
    memcpy(__x[0],x[0],natom*3*sizeof(numtype));
    memcpy(__v[0],v[0],natom*3*sizeof(numtype));
    memcpy(__f_prev[0],f_prev[0],natom*3*sizeof(numtype));

    int sqq;
    for(int i=0;i<natom;++i)
    {
        sqq=__atom_id[i];
        x_grd[sqq]=__x_grd[i];
        typ[sqq]=__typ[i];
        x_thr[sqq]=__x_thr[i];
        q[sqq]=__q[i];
        r_m[sqq]=__r_m[i];
        // atom_id[sqq]=__atom_id[i];
        mol_id[sqq]=__mol_id[i];
        intra_id[sqq]=__intra_id[i];
        memcpy(x[sqq],__x[i],3*sizeof(numtype));
        memcpy(v[sqq],__v[i],3*sizeof(numtype));
        memcpy(f_prev[sqq],__f_prev[i],3*sizeof(numtype));
    }

    for(int i=0;i<natom;++i)
    {
        atom_id[i]=i;
        f_mol[i]=f[i];
        x_mol[i]=x[i];
    }
}

// using id as secondary index may be better.
void ENVIRON::sort()
{
    for(int i=0;i<num_grid;++i) grid_start[i]=0;
    for(int i=0;i<natom;++i)
    {
        ++grid_start[x_grd[i]];
    }

    // accumulate
    grid_end[0]=grid_start[0]; grid_start[0]=0;
    for(int i=1;i<num_grid;++i) 
    {
        grid_end[i]=grid_end[i-1]+grid_start[i];
        grid_start[i]=grid_end[i-1];
    }

    memcpy(__typ,typ,natom*sizeof(int));
    memcpy(__x_thr,x_thr,natom*sizeof(int));
    memcpy(__q,q,    natom*sizeof(numtype));
    memcpy(__r_m,r_m,natom*sizeof(numtype));
    memcpy(__atom_id,atom_id,natom*sizeof(int));
    memcpy(__mol_id,mol_id,natom*sizeof(int));
    memcpy(__intra_id,intra_id,natom*sizeof(int));
    memcpy(__x_grd,x_grd,natom*sizeof(int));
    memcpy(__x[0],x[0],natom*3*sizeof(numtype));
    memcpy(__v[0],v[0],natom*3*sizeof(numtype));
    memcpy(__f_prev[0],f_prev[0],natom*3*sizeof(numtype));

    int sqq;
    for(int i=0;i<natom;++i)
    {
        sqq=grid_start[__x_grd[i]]++;
        x_grd[sqq]=__x_grd[i];
        typ[sqq]=__typ[i];
        x_thr[sqq]=__x_thr[i];
        q[sqq]=__q[i];
        r_m[sqq]=__r_m[i];
        atom_id[sqq]=__atom_id[i];
        mol_id[sqq]=__mol_id[i];
        intra_id[sqq]=__intra_id[i];
        memcpy(x[sqq],__x[i],3*sizeof(numtype));
        memcpy(v[sqq],__v[i],3*sizeof(numtype));
        memcpy(f_prev[sqq],__f_prev[i],3*sizeof(numtype));
    }

    // restore grid_start
    grid_start[0]=0;
    for(int i=1;i<num_grid;++i)
    {
        // if(grid_start[i] != grid_end[i]) END_PROGRAM("sort error");
        grid_start[i]=grid_end[i-1];
    }

    for(int i=0;i<natom;++i)
    {
        // typ_mol[atom_id[i]]=typ[i]; this is not needed because it does nothing
        f_mol[atom_id[i]]=f[i];
        x_mol[atom_id[i]]=x[i];
    }
}

void ENVIRON::sort_ghost()
{
    for(int i=0;i<num_grid;++i) grid_start_ghost[i]=0;
    for(int i=0;i<nghost;++i)
    {
        ++grid_start_ghost[x_grd_ghost[i]];
    }

    // accumulate
    grid_end_ghost[0]=grid_start_ghost[0]; grid_start_ghost[0]=0;
    for(int i=1;i<num_grid;++i) 
    {
        grid_end_ghost[i]=grid_end_ghost[i-1]+grid_start_ghost[i];
        grid_start_ghost[i]=grid_end_ghost[i-1];
    }

    memcpy(__typ,i_ghost,nghost*sizeof(int));
    memcpy(__v[0],g_ghost[0],nghost*3*sizeof(numtype));

    int sqq;
    for(int i=0;i<nghost;++i)
    {
        sqq=grid_start_ghost[x_grd_ghost[i]]++;
        i_ghost[sqq]=__typ[i];
        memcpy(g_ghost[sqq],__v[i],3*sizeof(numtype));
    }

    // restore grid_start
    grid_start_ghost[0]=natom;
    for(int i=1;i<num_grid;++i)
    {
        // if(grid_start_ghost[i] != grid_end_ghost[i]) END_PROGRAM("sort_ghost error");
        grid_start_ghost[i]=(grid_end_ghost[i-1]+=natom);
    }
    grid_end_ghost[num_grid-1]=ntot;
}

void LOCAL::generate_ghost()
{
    int n_ghost_local=0;
    numtype *xl;
    int xs,xe,ys,ye,zs,ze,_ix,_iy,_iz;
    numtype _x,_y,_z;
    int *i_ghost_local=ENVIRON::i_ghost_local[thread_id];
    numtype **g_ghost_local=ENVIRON::g_ghost_local[thread_id];

    for(int i=i_local_start;i!=i_local_end;++i)
    {
        xl=ENVIRON::x[i];
        _x=xl[0];
        _y=xl[1];
        _z=xl[2];
        xs=_x>ghost_lim?-1:0; xe=_x<m_ghost_lim? 1:0;
        ys=_y>ghost_lim?-1:0; ye=_y<m_ghost_lim? 1:0;
        zs=_z>ghost_lim?-1:0; ze=_z<m_ghost_lim? 1:0;
        
        if(xs!=xe || ys!=ye || zs!=ze)
        {
            for(_ix=xs;_ix<=xe;++_ix)
            {
                for(_iy=ys;_iy<=ye;++_iy)
                {
                    for(_iz=zs;_iz<=ze;++_iz)
                    {
                        if(_ix==0 && _iy==0 && _iz==0) continue;
                        i_ghost_local[n_ghost_local]=i;
                        xl=g_ghost_local[n_ghost_local];
                        xl[0]=(_ix==0 ? 0 : (_ix>0? ENVIRON::bl : ENVIRON::m_bl));
                        xl[1]=(_iy==0 ? 0 : (_iy>0? ENVIRON::bl : ENVIRON::m_bl));
                        xl[2]=(_iz==0 ? 0 : (_iz>0? ENVIRON::bl : ENVIRON::m_bl));
                        ++n_ghost_local;
                    }
                }
            }
        }
    }
    if(ghost_capacity_local<=n_ghost_local) END_PROGRAM("ghost overflow");
    ENVIRON::n_ghost_local[thread_id]=n_ghost_local;
}

void LOCAL::update_ghost_and_info()
{
    numtype *xglt,*xxl,*ggl;    
    int itt,iigs;
    for(int i=i_ghost_start;i<i_ghost_end;++i)
    {
        ENVIRON::q[iigs=i+ENVIRON::natom]=ENVIRON::q[itt=ENVIRON::i_ghost[i]];
        ENVIRON::typ[iigs]=ENVIRON::typ[itt];
        ENVIRON::mol_id[iigs]=ENVIRON::mol_id[itt];
        ENVIRON::intra_id[iigs]=ENVIRON::intra_id[itt];
        ENVIRON::atom_id[iigs]=ENVIRON::atom_id[itt];
        
        xglt=ENVIRON::x[itt];
        xxl=ENVIRON::x[iigs];
        ggl=ENVIRON::g_ghost[i];
        xxl[0]=xglt[0]+ggl[0]; xxl[1]=xglt[1]+ggl[1]; xxl[2]=xglt[2]+ggl[2];

        if(ENVIRON::intra_id[itt])
        {
            // std::cerr<<itt<<" "<<ENVIRON::atom_id[itt]<<"\n";
            xglt=ENVIRON::x_mol[ENVIRON::mol_repre_id[ENVIRON::mol_id[itt]]];
            if(std::fabs(xxl[0]-xglt[0])<ENVIRON::h_bl && 
                std::fabs(xxl[1]-xglt[1])<ENVIRON::h_bl &&
                std::fabs(xxl[2]-xglt[2])<ENVIRON::h_bl)
                ENVIRON::x_mol[ENVIRON::atom_id[itt]]=xxl;
        }
    }


    // for(int i=i_local_start;i<i_local_end;++i)
    // {
    //     if(ENVIRON::intra_id[i])
    //     {
    //         xxl=ENVIRON::x[i];
    //         xglt=ENVIRON::x_mol[ENVIRON::mol_repre_id[ENVIRON::mol_id[i]]];
    //         if(std::fabs(xxl[0]-xglt[0])<ENVIRON::h_bl && 
    //             std::fabs(xxl[1]-xglt[1])<ENVIRON::h_bl &&
    //             std::fabs(xxl[2]-xglt[2])<ENVIRON::h_bl)
    //             ENVIRON::x_mol[ENVIRON::atom_id[i]]=xxl;
    //     }
    // }
}
template void LOCAL::update_ghost_pos<true>();
template void LOCAL::update_ghost_pos<false>();
template <bool tag> void LOCAL::update_ghost_pos()
{
    if(tag)
    {
        i_ghost_start = ENVIRON::nghost*thread_id/ENVIRON::NUM_THREAD;
        i_ghost_end   = ENVIRON::nghost*(thread_id+1)/ENVIRON::NUM_THREAD;
    }

    numtype *xglt,*xxl,*ggl;
    int _tix,_tiy,_tiz;
    for(int i=i_ghost_start;i<i_ghost_end;++i)
    {
        xglt=ENVIRON::x[ENVIRON::i_ghost[i]];
        xxl=ENVIRON::x[i+ENVIRON::natom];
        ggl=ENVIRON::g_ghost[i];
        if(tag)
        {
            _tix=(ENVIRON::g_offset+(xxl[0]=xglt[0]+ggl[0]))*ENVIRON::g_rdx;
            _tiy=(ENVIRON::g_offset+(xxl[1]=xglt[1]+ggl[1]))*ENVIRON::g_rdx;
            _tiz=(ENVIRON::g_offset+(xxl[2]=xglt[2]+ggl[2]))*ENVIRON::g_rdx;
#ifdef CHECK_BINNING
            if(_tix<0) {_tix=0;std::cerr<<"G";} else if (_tix>=ENVIRON::num_gx){_tix=ENVIRON::num_gxm1;std::cerr<<"G";}
            if(_tiy<0) {_tiy=0;std::cerr<<"G";} else if (_tiy>=ENVIRON::num_gx){_tiy=ENVIRON::num_gxm1;std::cerr<<"G";}
            if(_tiz<0) {_tiz=0;std::cerr<<"G";} else if (_tiz>=ENVIRON::num_gx){_tiz=ENVIRON::num_gxm1;std::cerr<<"G";}
#endif
            ENVIRON::x_grd_ghost[i]=_tix*ENVIRON::num_gxsq+_tiy*ENVIRON::num_gx+_tiz;
        }
        else
        {
            xxl[0]=xglt[0]+ggl[0];
            xxl[1]=xglt[1]+ggl[1];
            xxl[2]=xglt[2]+ggl[2];
        }
    }
}

template void LOCAL::clear_vector<float>(float **arr, int len);
template void LOCAL::clear_vector<double>(double **arr, int len);
template <typename TY> void LOCAL::clear_vector(TY **arr, int len)
{
    TY *ff;
    for(int i=i_local_start;i<i_local_end;++i)
    {
        ff=arr[i];
        for(int j=0;j<len;++j) ff[j]=0;
    }
}
template void LOCAL::subtract_vector<float>(float **arr,float **arr2,float **out, int len);
template void LOCAL::subtract_vector<double>(double **arr,double **arr2,double **out, int len);
template <typename TY> void LOCAL::subtract_vector(TY **arr,TY **arr2,TY **out, int len)
{
    TY *ff,*ff2,*ffo;
    for(int i=i_local_start;i<i_local_end;++i)
    {
        ff=arr[i];ff2=arr2[i];ffo=out[i];
        for(int j=0;j<len;++j) ffo[j]=ff[j]-ff2[j];
    }
}

template void LOCAL::clear_vector_cs<float>(float *arr, int len);
template void LOCAL::clear_vector_cs<double>(double *arr, int len);
template <typename TY> void LOCAL::clear_vector_cs(TY *arr, int len)
{
    int st=i_local_start*len,ed=i_local_end*len;
    for(int i=st;i<ed;++i) arr[i]=0;
}

template void LOCAL::subtract_vector_cs<float>(float *arr,float *arr2,float *out, int len);
template void LOCAL::subtract_vector_cs<double>(double *arr,double *arr2,double *out, int len);
template <typename TY> void LOCAL::subtract_vector_cs(TY *arr,TY *arr2,TY *out, int len)
{
    int st=i_local_start*len,ed=i_local_end*len;
    for(int i=st;i<ed;++i) out[i]=arr[i]-arr2[i];
}

// int LOCAL::fetch_atoms()
// {
//     n_local=0;
//     int cl=0,cnl=ENVIRON::nghost;
//     memcpy(q_local+ENVIRON::natom,ENVIRON::q+ENVIRON::natom,              cnl*sizeof(numtype));
//     memcpy(typ_local+ENVIRON::natom,ENVIRON::typ+ENVIRON::natom,          cnl*sizeof(int));
//     memcpy(mol_id_local+ENVIRON::natom,ENVIRON::mol_id+ENVIRON::natom,    cnl*sizeof(int));
//     memcpy(intra_id_local+ENVIRON::natom,ENVIRON::intra_id+ENVIRON::natom,cnl*sizeof(int));
//     // memcpy(q_local,ENVIRON::q,              ENVIRON::ntot*sizeof(numtype));
//     // memcpy(typ_local,ENVIRON::typ,          ENVIRON::ntot*sizeof(int));
//     // memcpy(mol_id_local,ENVIRON::mol_id,    ENVIRON::ntot*sizeof(int));
//     // memcpy(intra_id_local,ENVIRON::intra_id,ENVIRON::ntot*sizeof(int));

//     for(int i=0; i<ENVIRON::natom; ++i)
//     {
//         // if(i>=i_local_start && i<i_local_end)
//         if(ENVIRON::x_thr[i] == thread_id)
//             ++n_local;
//     }
//     // std::cerr<<thread_id<<" mmm "<<n_local<<"\n";
//     if(LOCAL::local_capacity<=(cnl=n_local)) END_PROGRAM("n_local overflow");

//     for(int i=0; i<ENVIRON::natom; ++i)
//     {
//         // if(i>=i_local_start && i<i_local_end)
//         if(ENVIRON::x_thr[i] == thread_id)
//         {
//             x_local[cl]=ENVIRON::x[i];
//             f_local[cl]=ENVIRON::f[i];
//             q_local[cl]=ENVIRON::q[i];
//             mol_id_local[cl]=ENVIRON::mol_id[i];
//             intra_id_local[cl]=ENVIRON::intra_id[i];
//             typ_local[cl++]=ENVIRON::typ[i];
//         }
//         else
//         {
//             q_local[cnl]=ENVIRON::q[i];
//             typ_local[cnl]=ENVIRON::typ[i];
//             mol_id_local[cnl]=ENVIRON::mol_id[i];
//             intra_id_local[cnl]=ENVIRON::intra_id[i];
//             x_local[cnl++]=ENVIRON::x[i];
//         }
//     }
//     return n_local;
// }

void ENVIRON::verify_thread()
{
    numtype *xl;
    int _tix,_tiy,_tiz;
    for(int i=0;i<natom;++i)
    {
        xl=ENVIRON::x[i];
        _tix=(ENVIRON::h_bl + xl[0])*ENVIRON::t_rdx;
        _tiy=(ENVIRON::h_bl + xl[1])*ENVIRON::t_rdy;
        _tiz=(ENVIRON::h_bl + xl[2])*ENVIRON::t_rdz;
#ifdef CHECK_BINNING
        if(_tix<0) {_tix=0;std::cerr<<"w";} else if (_tix>=ENVIRON::NUM_X){_tix=ENVIRON::NUM_X-1;std::cerr<<"w";}
        if(_tiy<0) {_tiy=0;std::cerr<<"w";} else if (_tiy>=ENVIRON::NUM_Y){_tiy=ENVIRON::NUM_Y-1;std::cerr<<"w";}
        if(_tiz<0) {_tiz=0;std::cerr<<"w";} else if (_tiz>=ENVIRON::NUM_Z){_tiz=ENVIRON::NUM_Z-1;std::cerr<<"w";}
#endif
        if(ENVIRON::x_thr[i]!=_tix*(ENVIRON::NUM_Y*ENVIRON::NUM_Z)+_tiy*(ENVIRON::NUM_Z)+_tiz)
            END_PROGRAM("verify_thread fail");
    }
    for(int i=natom;i<ntot;++i)
    {
        if(ENVIRON::x_thr[i]!=-1) END_PROGRAM("verify_thread fail");
    }

    std::cerr<<"verify_thread pass\n";
}

void ENVIRON::verify_grid()
{
    int jend;
    numtype *xl;
    int _tix,_tiy,_tiz;
    if(grid_start[0]!=0 || grid_start_ghost[0]!=natom) END_PROGRAM("verify_grid fail");
    if(grid_end[num_grid-1]!=natom || grid_end_ghost[num_grid-1]!=ntot) END_PROGRAM("verify_grid fail");
    for(int i=1;i<num_grid;++i)
    {
        if(grid_start[i]!=grid_end[i-1] || grid_start_ghost[i]!=grid_end_ghost[i-1] ) END_PROGRAM("verify_grid fail");
    }
    for(int i=0;i<num_grid;++i)
    {
        jend=grid_end[i];
        for(int j=grid_start[i];j<jend;++j)
        {
            xl=ENVIRON::x[j];
            _tix=(ENVIRON::g_offset + xl[0])*ENVIRON::g_rdx;
            _tiy=(ENVIRON::g_offset + xl[1])*ENVIRON::g_rdx;
            _tiz=(ENVIRON::g_offset + xl[2])*ENVIRON::g_rdx;
#ifdef CHECK_BINNING
            if(_tix<0) {_tix=0;std::cerr<<"g";} else if (_tix>=ENVIRON::num_gx){_tix=ENVIRON::num_gxm1;std::cerr<<"g";}
            if(_tiy<0) {_tiy=0;std::cerr<<"g";} else if (_tiy>=ENVIRON::num_gx){_tiy=ENVIRON::num_gxm1;std::cerr<<"g";}
            if(_tiz<0) {_tiz=0;std::cerr<<"g";} else if (_tiz>=ENVIRON::num_gx){_tiz=ENVIRON::num_gxm1;std::cerr<<"g";}
#endif
            if(ENVIRON::x_grd[j]!=i || ENVIRON::x_grd[j]!=_tix*ENVIRON::num_gxsq+_tiy*ENVIRON::num_gx+_tiz) END_PROGRAM("verify_grid fail");
        }

        
        // std::cerr<<i<<" "<<grid_start_ghost[i]<<" "<<grid_end_ghost[i]<<"\n";
        jend=grid_end_ghost[i];
        for(int j=grid_start_ghost[i];j<jend;++j)
        {
            xl=ENVIRON::x[j];
            _tix=(ENVIRON::g_offset + xl[0])*ENVIRON::g_rdx;
            _tiy=(ENVIRON::g_offset + xl[1])*ENVIRON::g_rdx;
            _tiz=(ENVIRON::g_offset + xl[2])*ENVIRON::g_rdx;
#ifdef CHECK_BINNING
            if(_tix<0) {_tix=0;std::cerr<<"g";} else if (_tix>=ENVIRON::num_gx){_tix=ENVIRON::num_gxm1;std::cerr<<"g";}
            if(_tiy<0) {_tiy=0;std::cerr<<"g";} else if (_tiy>=ENVIRON::num_gx){_tiy=ENVIRON::num_gxm1;std::cerr<<"g";}
            if(_tiz<0) {_tiz=0;std::cerr<<"g";} else if (_tiz>=ENVIRON::num_gx){_tiz=ENVIRON::num_gxm1;std::cerr<<"g";}
#endif
            // std::cerr<<j<<" "<<xl[0]<<" "<<xl[1]<<" "<<xl[2]<<" "<<_tix*ENVIRON::num_gxsq+_tiy*ENVIRON::num_gx+_tiz<<" "<<i<<"\n";
            if(i!=_tix*ENVIRON::num_gxsq+_tiy*ENVIRON::num_gx+_tiz) END_PROGRAM("verify_grid fail");
        }
    }

    std::cerr<<"verify_grid pass\n";
}
