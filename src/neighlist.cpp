#include "neighlist.h"
#include "environ.h"
#include "memory.h"
#include "mathlib.h"
#include "util.h"
#include <cstring>
#include <iostream>


template int NEIGHLIST::build<true>();
template int NEIGHLIST::build<false>();

NEIGHLIST::NEIGHLIST(int thr, numtype cutoff, numtype skin, int cap, int **bin_l) :
nbuild(0)
{
    thread_id=thr;
    capacity=cap;
    create1DArray(local_list,LOCAL::local_capacity);
    create1DArray(num_neigh,LOCAL::local_capacity);
    create2DArray(x_prev,LOCAL::local_capacity,3);
    create2DArray(nei_list,LOCAL::local_capacity,capacity);
    create2DArray(special,LOCAL::local_capacity,capacity);
    // q_local=local->q_local;
    // typ_local=local->typ_local;
    // x_local=local->x_local;
    // f_local=local->f_local;
    // mol_id_local=local->mol_id_local;
    // intra_id_local=local->intra_id_local;
    local_list[0]=0; n_local=1; x_prev[0][0]=INFINITY; //this makes first call to is_valid returns false
    cutsq=sqr(cutoff+skin);
    dx2=sqr(skin/2);
    bin_list=bin_l;
}

int** NEIGHLIST::build_bin_list(numtype cutoff, int max_nbin, numtype cutoff2, numtype cutoff3)
{
    int **bin_list;
    create2DArray(bin_list,ENVIRON::num_grid,max_nbin+1);
    const int max_d=std::ceil(sqr(cutoff*ENVIRON::g_rdx)),
        max_d2=std::ceil(sqr(cutoff2*ENVIRON::g_rdx)),
        max_d3=std::ceil(sqr(cutoff3*ENVIRON::g_rdx));
    if(max_d2>max_d || max_d3>max_d2) END_PROGRAM("bad cutoff");
    std::cerr<<"bin_list max_di^2 = "<<max_d<<" "<<max_d2<<" "<<max_d3<<"\n";
    int n_i,n_j,*ilist,num_i;
    int mx,my,mz,tpp; //min dx
    int n_l,n_m;
    int index;
#ifdef NEGH_USE_INNERCUT
    int mxx,mxy,mxz; //max dx
#endif
#ifdef NEGH_USE_INNERCUT
#endif
    for(int _i=0;_i<ENVIRON::num_gx;++_i) 
    {
        n_i=_i*ENVIRON::num_gxsq;
        for(int _j=0;_j<ENVIRON::num_gx;++_j) 
        {
            n_j=n_i+_j*ENVIRON::num_gx;
            for(int _k=0;_k<ENVIRON::num_gx;++_k)
            {
                ilist=bin_list[n_j+_k];
                num_i=0;

                for(int _l=0;_l<ENVIRON::num_gx;++_l)
                {
                    if(_l==_i) // no skip if equal
                    {
                        mx=0;
#ifdef NEGH_USE_INNERCUT
                        mxx=1;
#endif
                    }
                    else
                    {
                        tpp=_l>_i? _l-_i : _i-_l;
                        if( (mx=sqr(tpp-1))>=max_d ) continue;
#ifdef NEGH_USE_INNERCUT
                        mxx=sqr(tpp+1);
#endif
                    }
                    n_l=_l*ENVIRON::num_gxsq;
                    for(int _m=0;_m<ENVIRON::num_gx;++_m)
                    {
                        if(_m==_j) // no skip if equal
                        {
                            my=mx;
#ifdef NEGH_USE_INNERCUT
                            mxy=mxx+1;
#endif
                        }
                        else
                        {
                            tpp=_m>_j? _m-_j : _j-_m;
                            if( (my=mx+sqr(tpp-1))>=max_d ) continue;
#ifdef NEGH_USE_INNERCUT
                            mxy=mxx+sqr(tpp+1);
#endif
                        }
                        n_m=n_l+_m*ENVIRON::num_gx;
                        for(int _n=0;_n<ENVIRON::num_gx;++_n)
                        {
                            if(_n==_k) // no skip if equal
                            {
                                mz=my;
#ifdef NEGH_USE_INNERCUT
                                mxz=mxy+1;
#endif
                            }
                            else
                            {
                                tpp=_n>_k? _n-_k : _k-_n;
                                if( (mz=my+sqr(tpp-1))>=max_d ) continue;
#ifdef NEGH_USE_INNERCUT
                                mxz=mxy+sqr(tpp+1);
#endif
                            }
                            index= n_m+_n;
                            if(mz<max_d2)
                            {
                                index |= mask_include_2;
                                if(mz<max_d3) index |= mask_include_3;
                            }
#ifdef NEGH_USE_INNERCUT
                            if(mxz<max_d)
                            {
                                index |= mask_no_check;
                                if(mxz<max_d2)
                                {
                                    index |= mask_no_check_2;
                                    if(mxz<max_d3) index |= mask_no_check_3;
                                }
                            }
#endif
                            ilist[num_i++]=index;
                        }
                    }
                }
                if(num_i>= max_nbin) END_PROGRAM("bin_list overflow");
                ilist[num_i]=-1;
            }
        }
    }

    // int i_,j_,k_,bb,j;
    // for(int i=0;i<ENVIRON::num_grid;++i)
    // {
    //     i_=i/ENVIRON::num_gxsq;
    //     j_=i%ENVIRON::num_gxsq;
    //     k_=j_%ENVIRON::num_gx;
    //     j_/=ENVIRON::num_gx;
    //     std::cerr<<(i_*ENVIRON::num_gxsq+j_*ENVIRON::num_gx+k_)<<"<"<<i_<<","<<j_<<","<<k_<<"> == ";
    //     ilist=bin_list[i];
    //     j=0;
    //     while(true)
    //     {
    //         if((bb=ilist[j])<0) break;
    //         i_=bb/ENVIRON::num_gxsq;
    //         j_=bb%ENVIRON::num_gxsq;
    //         k_=j_%ENVIRON::num_gx;
    //         j_/=ENVIRON::num_gx;
    //         std::cerr<<j<<"<"<<i_<<","<<j_<<","<<k_<<"> ";
    //         ++j;
    //     }
    //     std::cerr<<"\n"<<"\n";
    // }

    return bin_list;
}

NEIGHLIST::~NEIGHLIST()
{
    destroy1DArray(num_neigh);
    destroy2DArray(x_prev);
    destroy2DArray(nei_list);
    destroy2DArray(special);
}

bool NEIGHLIST::isvalid()
{
    numtype *xx,*xp;
    for(int i=0;i<n_local;++i)
    {
        xx=ENVIRON::x[local_list[i]];xp=x_prev[i];
        if (sqr(xx[0]-xp[0])+sqr(xx[1]-xp[1])+sqr(xx[2]-xp[2]) > dx2) return false;
    }
    return true;
}


template <bool full> int NEIGHLIST::build()
{
    int num_neigh_total=0;
    numtype *xx,*yy,_x,_y,_z;
    int icount, *ilist=nei_list[0],mi;
    int *ispl=special[0],*mspl;
    MOLECULE *mml;
    int *ib_list,cbin,jj_end,jj_begin,bi;
#ifdef NEGH_USE_INNERCUT
    int bmask;
#endif
    ++nbuild;
    n_local=0;
    for(int _i=0;_i<ENVIRON::natom;++_i)
    {
        if (ENVIRON::x_thr[_i]!=thread_id) continue;
        local_list[n_local]=_i;
        nei_list[n_local]=ilist;
        special[n_local]=ispl;
        xx=ENVIRON::x[_i];
        yy=x_prev[n_local];
        yy[0]=_x=xx[0]; yy[1]=_y=xx[1]; yy[2]=_z=xx[2];

        mi=ENVIRON::mol_id[_i];
        mml=ENVIRON::moltype+ENVIRON::mol_type[mi];
        mspl=mml->special[ENVIRON::intra_id[_i]];

        icount=0;

        ib_list=bin_list[ENVIRON::x_grd[_i]];
        
        // real atoms
        bi=0;
        while((cbin=ib_list[bi++])>=0)
        {
#ifdef NEGH_USE_INNERCUT
            bmask=cbin & mask_no_check;
            cbin &= mask_bin;
#endif
            if((jj_end=ENVIRON::grid_end[cbin]) <= (jj_begin=ENVIRON::grid_start[cbin])) continue;
#ifdef NEGH_USE_INNERCUT
            if(bmask)
            {
                for(int j = jj_begin; j<jj_end; ++j)
                {
                    if(full) {if(_i==j) continue;}
                    else {if(j<=_i && ENVIRON::x_thr[j]==thread_id) continue;}

                    if(mi != ENVIRON::mol_id[j]) ispl[icount]=0;
                    else
                    {
                        ispl[icount]=mspl[ENVIRON::intra_id[j]];
                    }
                    ilist[icount++]=j;

                    // yy=ENVIRON::x[j];
                    // if((sqr(_x-yy[0])+sqr(_y-yy[1])+sqr(_z-yy[2]))>cutsq)
                    // {
                    //     END_PROGRAM("bin mismatch");
                    // }
                }
            }
            else
            {
#endif
                for(int j = jj_begin; j<jj_end; ++j)
                {
                    if(full) {if(_i==j) continue;}
                    else {if(j<=_i && ENVIRON::x_thr[j]==thread_id) continue;}
                    yy=ENVIRON::x[j];
                    if((sqr(_x-yy[0])+sqr(_y-yy[1])+sqr(_z-yy[2]))<cutsq)
                    {
                        if(mi != ENVIRON::mol_id[j]) ispl[icount]=0;
                        else
                        {
                            ispl[icount]=mspl[ENVIRON::intra_id[j]];
                        }
                        ilist[icount++]=j;
                    }
                }
#ifdef NEGH_USE_INNERCUT
            }
#endif
        }

        // ghost atoms
        bi=0;
        while((cbin=ib_list[bi++])>=0)
        {
#ifdef NEGH_USE_INNERCUT
            bmask=cbin & mask_no_check;
            cbin &= mask_bin;
#endif
            if((jj_end=ENVIRON::grid_end_ghost[cbin]) <= (jj_begin=ENVIRON::grid_start_ghost[cbin])) continue;
#ifdef NEGH_USE_INNERCUT
            if(bmask)
            {
                for(int j = jj_begin; j<jj_end; ++j)
                {
                    if(mi != ENVIRON::mol_id[j]) ispl[icount]=0;
                    else
                    {
                        ispl[icount]=mspl[ENVIRON::intra_id[j]];
                    }
                    ilist[icount++]=j;

                    // yy=ENVIRON::x[j];
                    // if((sqr(_x-yy[0])+sqr(_y-yy[1])+sqr(_z-yy[2]))>cutsq)
                    // {
                    //     END_PROGRAM("bin mismatch");
                    // }
                }
            }
            else
            {
#endif
                for(int j = jj_begin; j<jj_end; ++j)
                {
                    yy=ENVIRON::x[j];
                    if((sqr(_x-yy[0])+sqr(_y-yy[1])+sqr(_z-yy[2]))<cutsq)
                    {
                        if(mi != ENVIRON::mol_id[j]) ispl[icount]=0;
                        else
                        {
                            ispl[icount]=mspl[ENVIRON::intra_id[j]];
                        }
                        ilist[icount++]=j;
                    }
                }
#ifdef NEGH_USE_INNERCUT
            }
#endif
        }

        ilist+=(num_neigh[n_local++]=icount);
        ispl+=icount;
        num_neigh_total+=icount;
        if(icount>=capacity) END_PROGRAM("nei_list overflow");
    }
    if(LOCAL::local_capacity<=n_local) END_PROGRAM("n_local overflow");
    return num_neigh_total;
}

// void NEIGHLIST::build_gr()
// {
//     numtype *xx,*yy,_x,_y,_z;
//     int icount, *ilist=nei_list[0];
//     int *ib_list,cbin,jj_end,jj_begin,bi;

// #ifdef NEGH_USE_INNERCUT
//     int bmask;
// #endif
//     ++nbuild;
//     n_local=0;
//     for(int _i=0;_i<ENVIRON::natom;++_i)
//     {
//         if (ENVIRON::x_thr[_i]!=thread_id) continue;
//         local_list[n_local]=_i;
//         nei_list[n_local]=ilist;
//         xx=ENVIRON::x[_i];
//         yy=x_prev[n_local];
//         yy[0]=_x=xx[0]; yy[1]=_y=xx[1]; yy[2]=_z=xx[2];

//         icount=0;

//         ib_list=bin_list[ENVIRON::x_grd[_i]];
        
//         // real atoms
//         bi=0;
//         while((cbin=ib_list[bi++])>=0)
//         {
// #ifdef NEGH_USE_INNERCUT
//             bmask=cbin & mask_no_check;
//             cbin &= mask_bin;
// #endif
//             if((jj_end=ENVIRON::grid_end[cbin]) <= (jj_begin=std::max(_i+1,ENVIRON::grid_start[cbin]))) continue;
// #ifdef NEGH_USE_INNERCUT
//             if(bmask)
//             {
//                 for(int j = jj_begin; j<jj_end; ++j)
//                 {
//                     ilist[icount++]=j;
//                 }
//             }
//             else
//             {
// #endif
//                 for(int j = jj_begin; j<jj_end; ++j)
//                 {
//                     yy=ENVIRON::x[j];
//                     if((sqr(_x-yy[0])+sqr(_y-yy[1])+sqr(_z-yy[2]))<cutsq)
//                     {
//                         ilist[icount++]=j;
//                     }
//                 }
// #ifdef NEGH_USE_INNERCUT
//             }
// #endif
//         }

//         // ghost atoms
//         bi=0;
//         while((cbin=ib_list[bi++])>=0)
//         {
// #ifdef NEGH_USE_INNERCUT
//             bmask=cbin & mask_no_check;
//             cbin &= mask_bin;
// #endif
//             if((jj_end=ENVIRON::grid_end_ghost[cbin]) <= (jj_begin=ENVIRON::grid_start_ghost[cbin])) continue;
// #ifdef NEGH_USE_INNERCUT
//             if(bmask)
//             {
//                 for(int j = jj_begin; j<jj_end; ++j)
//                 {
//                     ilist[icount++]=j;
//                 }
//             }
//             else
//             {
// #endif
//                 for(int j = jj_begin; j<jj_end; ++j)
//                 {
//                     yy=ENVIRON::x[j];
//                     if((sqr(_x-yy[0])+sqr(_y-yy[1])+sqr(_z-yy[2]))<cutsq)
//                     {
//                         ilist[icount++]=j;
//                     }
//                 }
// #ifdef NEGH_USE_INNERCUT
//             }
// #endif
//         }

//         ilist+=(num_neigh[n_local++]=icount);
//         if(icount>=capacity) END_PROGRAM("nei_list overflow");
//     }
//     if(LOCAL::local_capacity<=n_local) END_PROGRAM("n_local overflow"); 
// }

// void NEIGHLIST::build_gr(NEIGHLIST *lp)
// {
//     numtype *xx,*yy,_x,_y,_z,rsq;
//     int icount, *ilist=nei_list[0],mi;
//     int *ib_list,cbin,jj_end,jj_begin,bi;

//     int icount_lp, **nei_list_lp=lp->nei_list, *ilist_lp=nei_list_lp[0],*local_list_lp=lp->local_list,*num_neigh_lp=lp->num_neigh,capacity_lp=lp->capacity;
//     int **special_lp=lp->special, *ispl_lp=special_lp[0],*mspl;
//     numtype cutsq_lp=lp->cutsq;
//     MOLECULE *mml;

//     int bmask;

//     ++nbuild; ++lp->nbuild;
//     n_local=0;
//     for(int _i=0;_i<ENVIRON::natom;++_i)
//     {
//         if (ENVIRON::x_thr[_i]!=thread_id) continue;
//         local_list_lp[n_local]=local_list[n_local]=_i;
//         nei_list[n_local]=ilist; nei_list_lp[n_local]=ilist_lp;
//         special_lp[n_local]=ispl_lp;
//         xx=ENVIRON::x[_i];
//         yy=x_prev[n_local];
//         yy[0]=_x=xx[0]; yy[1]=_y=xx[1]; yy[2]=_z=xx[2];

//         mi=ENVIRON::mol_id[_i];
//         mml=ENVIRON::moltype+ENVIRON::mol_type[mi];
//         mspl=mml->special[ENVIRON::intra_id[_i]];

//         icount=icount_lp=0;

//         ib_list=bin_list[ENVIRON::x_grd[_i]];
        
//         // real atoms
//         bi=0;
//         while((cbin=ib_list[bi++])>=0)
//         {
//             bmask=cbin & mask_info;
//             cbin &= mask_bin;
//             if((jj_end=ENVIRON::grid_end[cbin]) <= (jj_begin=ENVIRON::grid_start[cbin])) continue;

//             switch (bmask)
//             {
//             case mask_include_2:                   // 2 check, 1 check
//                 for(int j = jj_begin; j<jj_end; ++j)
//                 {
//                     if(j<=_i && ENVIRON::x_thr[j]==thread_id) continue;
//                     yy=ENVIRON::x[j];
//                     if((rsq=sqr(_x-yy[0])+sqr(_y-yy[1])+sqr(_z-yy[2]))<cutsq)
//                     {
//                         if(_i<j) ilist[icount++]=j;
//                         if(rsq<cutsq_lp)
//                         {
//                             if(mi != ENVIRON::mol_id[j]) ispl_lp[icount_lp]=0;
//                             else
//                             {
//                                 ispl_lp[icount_lp]=mspl[ENVIRON::intra_id[j]];
//                             }
//                             ilist_lp[icount_lp++]=j;
//                         }
//                     }
//                 }
//                 break;
//             case 0:                                // 2 exclude, 1 check 
//                 for(int j = std::max(_i+1,jj_begin); j<jj_end; ++j)
//                 {
//                     yy=ENVIRON::x[j];
//                     if((rsq=sqr(_x-yy[0])+sqr(_y-yy[1])+sqr(_z-yy[2]))<cutsq)
//                     {
//                         ilist[icount++]=j;
//                     }
//                 }
//                 break;
// #ifdef NEGH_USE_INNERCUT
//             case mask_no_check:                    // 2 exclude, 1 no check
//                 for(int j = std::max(_i+1,jj_begin); j<jj_end; ++j)
//                 {
//                     ilist[icount++]=j;
//                 }
//                 break;
//             case mask_include_2 | mask_no_check:   //2 check, 1 no check
//                 for(int j = jj_begin; j<jj_end; ++j)
//                 {
//                     if(j<=_i && ENVIRON::x_thr[j]==thread_id) continue;
//                     if(_i<j) ilist[icount++]=j;
//                     yy=ENVIRON::x[j];
//                     if((sqr(_x-yy[0])+sqr(_y-yy[1])+sqr(_z-yy[2]))<cutsq_lp)
//                     {
//                         if(mi != ENVIRON::mol_id[j]) ispl_lp[icount_lp]=0;
//                         else
//                         {
//                             ispl_lp[icount_lp]=mspl[ENVIRON::intra_id[j]];
//                         }
//                         ilist_lp[icount_lp++]=j;
//                     }
//                 }
//                 break;
//             case mask_include_2 | mask_no_check_2 | mask_no_check: // 2 no check, 1 no check
//                 for(int j = jj_begin; j<jj_end; ++j)
//                 {
//                     if(j<=_i && ENVIRON::x_thr[j]==thread_id) continue;
//                     if(_i<j) ilist[icount++]=j;
//                     if(mi != ENVIRON::mol_id[j]) ispl_lp[icount_lp]=0;
//                     else
//                     {
//                         ispl_lp[icount_lp]=mspl[ENVIRON::intra_id[j]];
//                     }
//                     ilist_lp[icount_lp++]=j;
//                 }
//                 break;
// #endif
//             default:
//                 std::cerr<<(bmask>>24)<<"\n";
//                 END_PROGRAM("bad switch");
//                 break;
//             }
//         }

//         // ghost atoms
//         bi=0;
//         while((cbin=ib_list[bi++])>=0)
//         {
//             bmask=cbin & mask_info;
//             cbin &= mask_bin;
//             if((jj_end=ENVIRON::grid_end_ghost[cbin]) <= (jj_begin=ENVIRON::grid_start_ghost[cbin])) continue;

//             switch (bmask)
//             {
//             case mask_include_2:                   // 2 check, 1 check
//                 for(int j = jj_begin; j<jj_end; ++j)
//                 {
//                     yy=ENVIRON::x[j];
//                     if((rsq=sqr(_x-yy[0])+sqr(_y-yy[1])+sqr(_z-yy[2]))<cutsq)
//                     {
//                         if(_i<j) ilist[icount++]=j;
//                         if(rsq<cutsq_lp)
//                         {
//                             if(mi != ENVIRON::mol_id[j]) ispl_lp[icount_lp]=0;
//                             else
//                             {
//                                 ispl_lp[icount_lp]=mspl[ENVIRON::intra_id[j]];
//                             }
//                             ilist_lp[icount_lp++]=j;
//                         }
//                     }
//                 }
//                 break;
//             case 0:                                // 2 exclude, 1 check 
//                 for(int j = jj_begin; j<jj_end; ++j)
//                 {
//                     yy=ENVIRON::x[j];
//                     if((rsq=sqr(_x-yy[0])+sqr(_y-yy[1])+sqr(_z-yy[2]))<cutsq)
//                     {
//                         ilist[icount++]=j;
//                     }
//                 }
//                 break;
// #ifdef NEGH_USE_INNERCUT
//             case mask_no_check:                    // 2 exclude, 1 no check
//                 for(int j = jj_begin; j<jj_end; ++j)
//                 {
//                     ilist[icount++]=j;
//                 }
//                 break;
//             case mask_include_2 | mask_no_check:   //2 check, 1 no check
//                 for(int j = jj_begin; j<jj_end; ++j)
//                 {
//                     ilist[icount++]=j;
//                     yy=ENVIRON::x[j];
//                     if((sqr(_x-yy[0])+sqr(_y-yy[1])+sqr(_z-yy[2]))<cutsq_lp)
//                     {
//                         if(mi != ENVIRON::mol_id[j]) ispl_lp[icount_lp]=0;
//                         else
//                         {
//                             ispl_lp[icount_lp]=mspl[ENVIRON::intra_id[j]];
//                         }
//                         ilist_lp[icount_lp++]=j;
//                     }
//                 }
//                 break;
//             case mask_include_2 | mask_no_check_2 | mask_no_check: // 2 no check, 1 no check
//                 for(int j = jj_begin; j<jj_end; ++j)
//                 {
//                     ilist[icount++]=j;
//                     if(mi != ENVIRON::mol_id[j]) ispl_lp[icount_lp]=0;
//                     else
//                     {
//                         ispl_lp[icount_lp]=mspl[ENVIRON::intra_id[j]];
//                     }
//                     ilist_lp[icount_lp++]=j;
//                 }
//                 break;
// #endif
//             default:
//                 std::cerr<<(bmask>>24)<<"\n";
//                 END_PROGRAM("bad switch");
//                 break;
//             }
//         }

//         ilist+=(num_neigh[n_local]=icount); ilist_lp+=(num_neigh_lp[n_local++]=icount_lp);
//         ispl_lp+=icount_lp;
//         if(icount>=capacity || icount_lp>=capacity_lp) END_PROGRAM("nei_list overflow");
//     }
//     if(LOCAL::local_capacity<=n_local) END_PROGRAM("n_local overflow");
//     lp->n_local=n_local;
// }