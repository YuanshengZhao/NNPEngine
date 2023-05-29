#include "grsq.h"
#include "memory.h"
#include "mathlib.h"
#include "util.h"
#include <iostream>
#include <cstring>

// #define GRSQ_USE_BLAS

#ifdef GRSQ_USE_BLAS
#ifdef __APPLE__
#include </usr/local/Cellar/openblas/0.3.21/include/cblas.h>
#else
#include <cblas.h>
#endif

#ifdef FLOAT_PRECESION
#define cblas_matmul cblas_sgemm
#else
#define cblas_matmul cblas_dgemm
#endif

#endif

#define MAXACC 100

#define ONE_THIRD (numtype)0.3333333333333333333333333333333333333333
GR::GR(int nbi, numtype r_max, numtype r_max_inner)
{
    if(r_max_inner<=0)
    {
        r_max=r_max<=0? ENVIRON::h_bl : r_max;
        nbin_r_inner=nbin_r=nbi;
        r_maxsq_inner=r_maxsq=sqr(r_max);;
        dr=r_max/nbi;
        r_dr=nbi/r_max;
    }
    else
    {
        nbin_r_inner=nbi;
        dr=r_max_inner/nbi;
        r_dr=nbi/r_max_inner;
        r_maxsq_inner=sqr(r_max_inner);
        if ((r_max=r_max<=0? ENVIRON::h_bl : r_max)<r_max_inner) END_PROGRAM("r_max_inner larger than r_max");
        nbin_r=r_max*r_dr;
        r_maxsq=sqr(nbin_r*dr);
    }

    create1DArray(rr,nbin_r);
    create2DArray(gr,ENVIRON::npair,nbin_r);
    create2DArray(norm_gr,ENVIRON::npair,nbin_r);
    create3DArray(gr_local,ENVIRON::NUM_THREAD,ENVIRON::npair,nbin_r);

    for(int i=0;i<nbin_r;++i) rr[i]=(i+0.5F)*dr;

    numtype normg,rc_dr3=cub(ENVIRON::bl*r_dr)/(ENVIRON::four_pi),*pnorm;

    int ipair=0;
    for(int i=0;i<ENVIRON::ntype;++i)
    {
        for(int j=i;j<ENVIRON::ntype;++j)
        {
            normg=rc_dr3/(ENVIRON::typecount[i]*(i==j? (ENVIRON::typecount[j]-1) : 2*ENVIRON::typecount[j]));
            pnorm=norm_gr[ipair++];
            for(int k=0;k<nbin_r;++k)
            {
                pnorm[k]=normg/(k*(k+1)+ONE_THIRD);
            }
        }
    }

    create1DArray(n_start,ENVIRON::NUM_THREAD);
    create1DArray(n_end,ENVIRON::NUM_THREAD);
    int np=nbin_r*ENVIRON::npair;
    for(int i=1;i<ENVIRON::NUM_THREAD;++i)
    {
        n_end[i-1]=n_start[i]=i*np/ENVIRON::NUM_THREAD; //view gr as id array when reducing
    }
    n_start[0]=0; n_end[ENVIRON::NUM_THREAD-1]=np;
}

GR::~GR()
{
    destroy1DArray(rr);
    destroy2DArray(gr);
    destroy2DArray(norm_gr);
    destroy3DArray(gr_local);
    destroy1DArray(n_start);
    destroy1DArray(n_end);
}

template void GR::tally<true> (NEIGHLIST *list);
template void GR::tally<false> (NEIGHLIST *list);
template <bool inner> void GR::tally (NEIGHLIST *list)
{
    numtype thisrmaxsq=inner? r_maxsq_inner : r_maxsq;
    int thisnbin=inner? nbin_r_inner : nbin_r;

    int thread_id=list->thread_id;
    int *pgr,**mygr=gr_local[thread_id];
    for(int p=0;p<ENVIRON::npair;++p)
    {
        pgr=mygr[p];
        for(int i=0;i<thisnbin;++i) pgr[i]=0;
    }


    numtype *xi,*xj;
    int *ipair;
    int n_local=list->n_local, **nlist=list->nei_list,*num_nei=list->num_neigh;
    int *local_list=list->local_list;
    int inum,*ilist,ii,jj,bin;
    numtype rsq, _xi,_yi,_zi;

    for(int i=0;i<n_local;++i)
    {
        ii=local_list[i];
        xi=ENVIRON::x[ii];
        _xi=xi[0];_yi=xi[1];_zi=xi[2];
        ipair=ENVIRON::pairtype[ENVIRON::typ[ii]];

        ilist=nlist[i];
        inum=num_nei[i];

        for(int j=0; j<inum; ++j)
        {
            jj=ilist[j];
            xj=ENVIRON::x[jj];
            if((rsq=sqr(_xi-xj[0])+sqr(_yi-xj[1])+sqr(_zi-xj[2]))<thisrmaxsq && (bin=std::sqrt(rsq)*r_dr)<thisnbin)
            {
                if(inner)
                {
                    if(ENVIRON::x_thr[jj]==thread_id)
                        mygr[ipair[ENVIRON::typ[jj]]][bin]+=2;
                    else
                        ++mygr[ipair[ENVIRON::typ[jj]]][bin];
                }
                else
                {
                    if(jj<ENVIRON::natom)
                        mygr[ipair[ENVIRON::typ[jj]]][bin]+=2;
                    else
                        ++mygr[ipair[ENVIRON::typ[jj]]][bin];

                }
            }
        }
    }
}

void GR::tally(int **bin_list, LOCAL *local)
{
    int *pgr,**mygr=gr_local[local->thread_id],_iend=local->i_local_end;
    for(int p=0;p<ENVIRON::npair;++p)
    {
        pgr=mygr[p];
        for(int i=0;i<nbin_r;++i) pgr[i]=0;
    }

    numtype *xx,*yy,_xi,_yi,_zi;
    int *ib_list,cbin,jj_end,jj_begin,bi;
    int bin, *ipair;
    numtype rsq;

    for(int _i=local->i_local_start;_i<_iend;++_i)
    {
        xx=ENVIRON::x[_i];
        _xi=xx[0];_yi=xx[1];_zi=xx[2];
        ipair=ENVIRON::pairtype[ENVIRON::typ[_i]];

        ib_list=bin_list[ENVIRON::x_grd[_i]];
        
        // real atoms
        bi=0;
        while((cbin=ib_list[bi++])>=0)
        {
            cbin &= NEIGHLIST::mask_bin;
            if((jj_end=ENVIRON::grid_end[cbin]) <= (jj_begin=std::max(_i+1,ENVIRON::grid_start[cbin]))) continue;
            for(int j = jj_begin; j<jj_end; ++j)
            {
                yy=ENVIRON::x[j];
                if((rsq=sqr(_xi-yy[0])+sqr(_yi-yy[1])+sqr(_zi-yy[2]))<r_maxsq && (bin=std::sqrt(rsq)*r_dr)<nbin_r)
                {
                    mygr[ipair[ENVIRON::typ[j]]][bin]+=2;
                }
            }
        }

        // ghost atoms
        bi=0;
        while((cbin=ib_list[bi++])>=0)
        {
            cbin &= NEIGHLIST::mask_bin;
            if((jj_end=ENVIRON::grid_end_ghost[cbin]) <= (jj_begin=ENVIRON::grid_start_ghost[cbin])) continue;
            for(int j = jj_begin; j<jj_end; ++j)
            {
                yy=ENVIRON::x[j];
                if((rsq=sqr(_xi-yy[0])+sqr(_yi-yy[1])+sqr(_zi-yy[2]))<r_maxsq && (bin=std::sqrt(rsq)*r_dr)<nbin_r)
                {
                    ++mygr[ipair[ENVIRON::typ[j]]][bin];
                }
            }
        }
    }
}

void GR::reduce_local(int thread_id)
{
    int iend=n_end[thread_id],istart=n_start[thread_id];
    numtype *gra=gr[0],*grn=norm_gr[0];
    int *grl;
    for(int i=istart;i<iend;++i) gra[i]=0;
    for(int thr=0;thr<ENVIRON::NUM_THREAD;++thr)
    {
        grl=gr_local[thr][0];
        for(int i=istart;i<iend;++i) gra[i]+=grl[i];
    }
    for(int i=istart;i<iend;++i) gra[i]=gra[i]*grn[i]-1;
}

void GR::dump(const char *fname, bool append)
{
    FILE *fp;
    int num=1;
    numtype factor,cofactor,gri;
    if(append && (fp=fopen(fname,"r")))
    {
        fscanf(fp,"# %d ",&num);
        num=std::min(num+1,MAXACC);
        cofactor=1-( factor=((numtype)1)/(num) );
        std::cerr<<"gr_acc = "<<num<<"\n";
        for(int i=0;i<nbin_r;++i)
        {
            fscanf(fp,FMT_NUMTYPE " ",&gri);
            if(std::fabs(gri-rr[i])>EPSILON) std::cerr<<"warning: large diff: "<<gri<<" "<<rr[i]<<"\n";
            for(int p=0;p<ENVIRON::npair;++p)
            {
                fscanf(fp,FMT_NUMTYPE " ",&gri);
                gr[p][i]=gr[p][i]*factor+cofactor*gri;
            }
        }
        fclose(fp);
    }

    fp=fopen(fname,"w");
    fprintf(fp,"# %d\n",num);
    for(int i=0;i<nbin_r;++i)
    {
        fprintf(fp,"%.15e",rr[i]);
        for(int p=0;p<ENVIRON::npair;++p)
            // fprintf(fp," %d",gr_local[0][p][i]);
            fprintf(fp," %.15e",gr[p][i]);
        fprintf(fp,"\n");
    }
    fclose(fp);
}

SQ::SQ(int nq,numtype qmax, GR *ggr)
{
#ifdef GRSQ_USE_BLAS
    openblas_set_num_threads(1);
#endif
    q_max=qmax;
    gr=ggr;
    nbin_q=nq;
    create1DArray(qq,nbin_q);
    create2DArray(sq,ENVIRON::npair,nbin_q);
    nbin_r=gr->nbin_r;
    create2DArray(ftmx,nbin_q,nbin_r);
    numtype dq=q_max/nbin_q,qi;
    numtype *rr=gr->rr,*ift;
    numtype fourPiRhoDr=ENVIRON::four_pi*ENVIRON::natom/cub(ENVIRON::bl)*(gr->dr);
    for(int i=0;i<nbin_q;++i)
    {
        qq[i]=qi=(i+.5F)*dq;
        ift=ftmx[i];
        for(int j=0;j<nbin_r;++j)
            ift[j]=std::sin(qi*rr[j])*fourPiRhoDr*rr[j]/qi;
    }

    create1DArray(iq_start,ENVIRON::NUM_THREAD);
    create1DArray(iq_end,ENVIRON::NUM_THREAD);
    for(int i=1;i<ENVIRON::NUM_THREAD;++i)
    {
        iq_end[i-1]=iq_start[i]=i*nbin_q/ENVIRON::NUM_THREAD;
    }
    iq_start[0]=0; iq_end[ENVIRON::NUM_THREAD-1]=nbin_q;
}

SQ::~SQ()
{
    destroy1DArray(qq);
    destroy2DArray(sq);
    destroy2DArray(ftmx);
    destroy1DArray(iq_start);
    destroy1DArray(iq_end);
}

void SQ::compute(int thread_id)
{
    int qend=iq_end[thread_id],qstart=iq_start[thread_id];
#ifdef GRSQ_USE_BLAS
    cblas_matmul(CblasRowMajor,CblasNoTrans,CblasTrans,ENVIRON::npair,qend-qstart,nbin_r,1,gr->gr[0],nbin_r,ftmx[qstart],nbin_r,0,sq[0]+qstart,nbin_q);
#else
    numtype **grs=gr->gr,*igr,*isq,*ift,temp;
    for(int i=0;i<ENVIRON::npair;++i)
    {
        igr=grs[i];
        isq=sq[i];
        for(int iq=qstart;iq<qend;++iq)
        {
            temp=0;
            ift=ftmx[iq];
            for(int ir=0;ir<nbin_r;++ir)
            {
                temp+=ift[ir]*igr[ir];
            }
            isq[iq]=temp;
        }
    }
#endif
}

void SQ::dump(const char *fname, bool append)
{
    FILE *fp;
    int num=1;
    numtype factor,cofactor,sqi;
    if(append && (fp=fopen(fname,"r")))
    {
        fscanf(fp,"# %d ",&num);
        num=std::min(num+1,MAXACC);
        cofactor=1-( factor=((numtype)1)/(num) );
        std::cerr<<"sq_acc = "<<num<<"\n";
        for(int i=0;i<nbin_q;++i)
        {
            fscanf(fp,FMT_NUMTYPE " ",&sqi);
            if(std::fabs(sqi-qq[i])>EPSILON) std::cerr<<"warning: large diff: "<<sqi<<" "<<qq[i]<<"\n";
            for(int p=0;p<ENVIRON::npair;++p)
            {
                fscanf(fp,FMT_NUMTYPE " ",&sqi);
                sq[p][i]=sq[p][i]*factor+cofactor*sqi;
            }
        }
        fclose(fp);
    }

    fp=fopen(fname,"w");
    fprintf(fp,"# %d\n",num);
    for(int i=0;i<nbin_q;++i)
    {
        fprintf(fp,"%.15e",qq[i]);
        for(int p=0;p<ENVIRON::npair;++p)
            // fprintf(fp," %d",gr_local[0][p][i]);
            fprintf(fp," %.15e",sq[p][i]);
        fprintf(fp,"\n");
    }
    fclose(fp);
}


SQD::SQD(int nq,numtype qmax)
{
    nbin_q=nq;
    q_max=qmax;
    ENVIRON::npair=ENVIRON::ntype*(ENVIRON::ntype+1)/2;
    create1DArray(qq,nbin_q);
    // create1DArray(counti,nbin_q);
    numtype dq=qmax/nq;
    create2DArray(sq,nbin_q,ENVIRON::npair);

    int i2,i2j2,i2j2k2,j2,k2;
    numtype g=ENVIRON::two_pi/ENVIRON::bl,gx,gy,gz;
    int kcutoff=q_max/g;
    int totalq=(cub(2*kcutoff+1)-1)/2;
    create2DArray(gvec,totalq,3);
    create1DArray(gscl,totalq);
    // create2DArray(gsq,totalq,ENVIRON::npair);

    int cuti2=(int)sqr(q_max/g);
    ngv=0;

    for (int i=-kcutoff; i<=kcutoff; ++i)
    {
        i2=i*i;
        gx=g*i;
        for (int j=-kcutoff; j<=kcutoff; ++j)
        {
            i2j2=i2+(j2=j*j);
            gy=g*j;
            for (int k=1; k<=kcutoff; ++k)
            {
                i2j2k2=i2j2+(k2=k*k);
                // if ((i2>16 && (j2>16 || k2>16)) || (j2>16 && k2>16)) continue;
                if (i2j2k2>cuti2) continue;

                gz=g*k;
                gvec[ngv][0]=gx; gvec[ngv][1]=gy; gvec[ngv][2]=gz;
                gscl[ngv++]=std::sqrt((numtype)i2j2k2)*g;
            }
        }
    }

    for (int i=-kcutoff; i<=kcutoff; ++i)
    {
        i2=i*i;
        gx=g*i;
        for (int j=1; j<=kcutoff; ++j)
        {
            i2j2=i2+(j2=j*j);
            // if (i2>16 && j2>16) continue;
            if (i2j2>cuti2) continue;
            gy=g*j;
            gvec[ngv][0]=gx; gvec[ngv][1]=gy; gvec[ngv][2]=0;
            gscl[ngv++]=std::sqrt((numtype)i2j2)*g;
        }
    }

    for (int i=1; i<=kcutoff; ++i)
    {
        i2=i*i;
        if (i2>cuti2) continue;
        gx=g*i;
        gvec[ngv][0]=gx; gvec[ngv][1]=0; gvec[ngv][2]=0;
        gscl[ngv++]=std::sqrt((numtype)i2)*g;
    }
    sortQ(0,ngv);
    std::cerr<<"SQDIRECT::SQDIRECT "<<ngv<<" G vecs\n";
    // for(int i=0;i<ngv;++i)
    //     std::cerr<<i<<" "<<gvec[i][0]<<" "<<gvec[i][1]<<" "<<gvec[i][2]<<" "<<
    //         gscl[i]<<" "<<(std::sqrt(sqr(gvec[i][0])+sqr(gvec[i][1])+sqr(gvec[i][2])))<<"\n";

    create1DArray(iq_start,ENVIRON::NUM_THREAD);
    create1DArray(iq_end,ENVIRON::NUM_THREAD);
    create1DArray(vend,nbin_q);
    create1DArray(vbegin,nbin_q);

    vbegin[0]=i2=0;
    for(int i=1;i<nbin_q;++i)
    {
        gx=i*dq;
        gy=0;
        while (i2<ngv && (gz=gscl[i2])<gx)
        {
            gy+=gz;
            ++i2;
        }
        vbegin[i]=vend[i-1]=i2;
        qq[i-1]=gy;
    }
    gy=0;
    while (i2<ngv)
    {
        gy+=gscl[i2];
        ++i2;
    }
    qq[nbin_q-1]=gy;
    vend[nbin_q-1]=ngv;
    for(int i=0;i<nbin_q;++i) qq[i]/=(vend[i]-vbegin[i]);

    iq_start[0]=j2=0;
    for(int i=1;i<ENVIRON::NUM_THREAD;++i)
    {
        i2=i*ngv/ENVIRON::NUM_THREAD;
        while(vbegin[j2]<=i2) ++j2;
        iq_end[i-1]=iq_start[i]=j2-1;
        // std::cerr<<i<<" "<<(j2-1)<<"\n";
    }
    iq_end[ENVIRON::NUM_THREAD-1]=nbin_q;

    // for(int i=1;i<nbin_q;++i) std::cerr<<i<<" "<<(vbegin[i])<<"\n";

    create2DArray(ssa,ENVIRON::NUM_THREAD,ENVIRON::ntype);
    create2DArray(cca,ENVIRON::NUM_THREAD,ENVIRON::ntype);
    create1DArray(factor,ENVIRON::npair);
    create1DArray(base,ENVIRON::npair);

    int ipair=0;

    for(int i=0;i<ENVIRON::ntype;++i)
    {
        for(int j=i;j<ENVIRON::ntype;++j)
        {
            factor[ipair]=((numtype)ENVIRON::natom)/(ENVIRON::typecount[i]*ENVIRON::typecount[j]);
            base[ipair]= i==j? ((numtype)ENVIRON::natom)/(ENVIRON::typecount[j]) : 0;
            ++ipair;
        }
    }

    
}

SQD::~SQD()
{
    destroy1DArray(qq);
    destroy2DArray(sq);
    destroy2DArray(gvec);
    destroy1DArray(gscl);
    destroy1DArray(iq_start);
    destroy1DArray(iq_end);
    destroy1DArray(vend);
    destroy1DArray(vbegin);
    destroy2DArray(ssa);
    destroy2DArray(cca);
    destroy1DArray(factor);
    destroy1DArray(base);
}

void SQD::sortQ(int lo, int hi)
{
    if (lo+1>=hi || lo<0)  return;
    int p=partitionQ(lo,hi); 
    sortQ(lo,p);
    sortQ(p+1,hi); 
}
int SQD::partitionQ(int lo, int hi)
{
    //use median-of-3 pivot
    int i=(lo+hi)/2;
    numtype temp,*tp,*ti;
    --hi;
    if (gscl[i]<gscl[lo])
    {
        temp=gscl[lo]; gscl[lo]=gscl[i]; gscl[i]=temp;
        tp=gvec[lo]; ti=gvec[i];
        temp=tp[0]; tp[0]=ti[0]; ti[0]=temp;
        temp=tp[1]; tp[1]=ti[1]; ti[1]=temp;
        temp=tp[2]; tp[2]=ti[2]; ti[2]=temp;
    }
    if (gscl[hi]<gscl[lo])
    { 
        temp=gscl[lo]; gscl[lo]=gscl[hi]; gscl[hi]=temp;
        tp=gvec[lo]; ti=gvec[hi];
        temp=tp[0]; tp[0]=ti[0]; ti[0]=temp;
        temp=tp[1]; tp[1]=ti[1]; ti[1]=temp;
        temp=tp[2]; tp[2]=ti[2]; ti[2]=temp;
    }
    if (gscl[i]<gscl[hi])
    {
        temp=gscl[i]; gscl[i]=gscl[hi]; gscl[hi]=temp;
        tp=gvec[hi]; ti=gvec[i];
        temp=tp[0]; tp[0]=ti[0]; ti[0]=temp;
        temp=tp[1]; tp[1]=ti[1]; ti[1]=temp;
        temp=tp[2]; tp[2]=ti[2]; ti[2]=temp;
    }

    double pivot=gscl[hi];

    i=lo-1;
    for(int j=lo;j<hi;++j)
    {
        if (gscl[j]<=pivot)
        {
            ++i;
            temp=gscl[i]; gscl[i]=gscl[j]; gscl[j]=temp;
            tp=gvec[j]; ti=gvec[i];
            temp=tp[0]; tp[0]=ti[0]; ti[0]=temp;
            temp=tp[1]; tp[1]=ti[1]; ti[1]=temp;
            temp=tp[2]; tp[2]=ti[2]; ti[2]=temp;
        }
    }
    ++i;
    temp=gscl[i]; gscl[i]=gscl[hi]; gscl[hi]=temp;
    tp=gvec[hi]; ti=gvec[i];
    temp=tp[0]; tp[0]=ti[0]; ti[0]=temp;
    temp=tp[1]; tp[1]=ti[1]; ti[1]=temp;
    temp=tp[2]; tp[2]=ti[2]; ti[2]=temp;
    return i;
}

void SQD::compute(int thread_id)
{
    numtype *ss=ssa[thread_id],*cc=cca[thread_id];
    int nend=iq_end[thread_id],qend;
    numtype qx,qy,qz,*xi,*qi,kr,*spi;
    int ti;
    for(int ibin=iq_start[thread_id];ibin<nend;++ibin)
    {
        qend=vend[ibin];
        spi=sq[ibin];
        for(int i=0;i<ENVIRON::npair;++i) spi[i]=0;
        for(int iq=vbegin[ibin];iq<qend;++iq)
        {
            qi=gvec[iq];
            qx=qi[0]; qy=qi[1]; qz=qi[2]; 
            for(int i=0;i<ENVIRON::ntype;++i) ss[i]=cc[i]=0;
            for(int i=0;i<ENVIRON::natom;++i)
            {
                xi=ENVIRON::x[i];
                ti=ENVIRON::typ[i];
                kr=(qx*xi[0]+qy*xi[1]+qz*xi[2]);
                ss[ti]+=std::sin(kr);
                cc[ti]+=std::cos(kr);
            }
            ti=0;
            for(int i=0;i<ENVIRON::ntype;++i)
            {
                for(int j=i;j<ENVIRON::ntype;++j)
                {
                    spi[ti++]+=ss[i]*ss[j]+cc[i]*cc[j];
                }
            }
        }
        for(int i=0;i<ENVIRON::npair;++i) spi[i]=spi[i]/(qend-vbegin[ibin])*factor[i]-base[i];
    }
}

void SQD::dump(const char *fname, bool append)
{
    FILE *fp;
    int num=1;
    numtype factor,cofactor,sqi;
    if(append && (fp=fopen(fname,"r")))
    {
        fscanf(fp,"# %d ",&num);
        num=std::min(num+1,MAXACC);
        cofactor=1-( factor=((numtype)1)/(num) );
        std::cerr<<"sq_acc = "<<num<<"\n";
        for(int i=0;i<nbin_q;++i)
        {
            if (vbegin[i]==vend[i]) continue;
            fscanf(fp,FMT_NUMTYPE " ",&sqi);
            if(std::fabs(sqi-qq[i])>EPSILON) std::cerr<<"warning: large diff: "<<sqi<<" "<<qq[i]<<"\n";
            for(int p=0;p<ENVIRON::npair;++p)
            {
                fscanf(fp,FMT_NUMTYPE " ",&sqi);
                sq[i][p]=sq[i][p]*factor+cofactor*sqi;
            }
        }
        fclose(fp);
    }

    fp=fopen(fname,"w");
    fprintf(fp,"# %d\n",num);
    for(int i=0;i<nbin_q;++i)
    {
        if (vbegin[i]==vend[i]) continue;
        fprintf(fp,"%.15e",qq[i]);
        for(int p=0;p<ENVIRON::npair;++p)
            fprintf(fp," %.15e",sq[i][p]);
        fprintf(fp,"\n");
    }
    fclose(fp);
}

/*
// SQM::SQM(int nq,int nr,numtype q_max, numtype rc)
// {
//     nbin_q=nq;
//     nbin_r=nr;
//     ENVIRON::npair=ENVIRON::ntype*(ENVIRON::ntype+1)/2;
//     numtype r_max=ENVIRON::h_bl;
//     dr=r_max/nbin_r; r_dr=nbin_r/r_max;

//     create1DArray(qq,nbin_q);
//     create2DArray(sq,ENVIRON::npair+1,nbin_q);
//     create1DArray(rr,nbin_r);
//     create1DArray(gr,std::max(nbin_r,nbin_q));
//     create1DArray(norm,nbin_r);
//     numtype normg=2*cub(ENVIRON::bl*r_dr)/(ENVIRON::four_pi*ENVIRON::nmol*(ENVIRON::nmol-1));
//     for(int k=0;k<nbin_r;++k)
//     {
//         rr[k]=(k+.5F)*dr;
//         norm[k]=normg/(k*(k+1)+ONE_THIRD);
//     }
//     create2DArray(sincqr,nbin_q,nbin_r);
//     create2DArray(cr_intra,ENVIRON::ntype,nbin_r);
//     create2DArray(cq_intra,ENVIRON::ntype,nbin_q);
//     create2DArray(gr_intra,ENVIRON::npair,nbin_r);
//     create2DArray(sq_intra,ENVIRON::npair,nbin_q);

//     create2DArray(ftmx,nbin_q,nbin_r);
//     numtype dq=q_max/nbin_q,qi,ss;
//     numtype *ift,qr,*isq;
//     numtype fourPiRhoDr=ENVIRON::four_pi*ENVIRON::nmol/cub(ENVIRON::bl)*(dr);
//     for(int i=0;i<nbin_q;++i)
//     {
//         qq[i]=qi=(i+.5F)*dq;
//         ift=ftmx[i];
//         isq=sincqr[i];
//         for(int j=0;j<nbin_r;++j)
//         {
//             qr=qi*rr[j];
//             ss=std::sin(qr);

//             ift[j]=ss*fourPiRhoDr*rr[j]/qi;
//             isq[j]=ss/qr;
//         }
//     }

//     create2DArray(conv_ker,nbin_q,nbin_q);
//     qr=-rc/ENVIRON::pi*dq;
//     for(int i=0;i<nbin_q;++i)
//     {
//         qi=qq[i];
//         ift=conv_ker[i];
//         for(int j=0;j<nbin_q;++j)
//         {
//             ss=qq[j];
//             ift[j]=qr*ss/qi*(sphBessel0(rc*(qi-ss))-sphBessel0(rc*(qi+ss)));
//         }
//         ift[i]+=1;
//     }

// }

// SQM::~SQM()
// {

// }

// void SQM::compute()
// {
//     numtype rsq,*xi,*xj,cutsq=sqr(ENVIRON::h_bl);
//     int bin;

//     if(ENVIRON::nmoltype!=1) END_PROGRAM("multiple moltypes not supported");
//     int nat_mol=ENVIRON::moltype->n_atom,jmax,bs=0,*ipair,tpj;
//     numtype _x,_y,_z;

//     //intra part
//     for(int ti=0;ti<ENVIRON::ntype;++ti)
//     {
//         xi=cr_intra[ti];
//         for(int i=0;i<nbin_r;++i) xi[i]=0;
//     }
//     for(int ti=0;ti<ENVIRON::npair;++ti)
//     {
//         xi=gr_intra[ti];
//         for(int i=0;i<nbin_r;++i) xi[i]=0;
//     }

//     for(int im=0;im<ENVIRON::nmol;++im)
//     {
//         jmax=nat_mol+bs;
//         //i=bs
//         xi=ENVIRON::x_mol[bs];
//         _x=xi[0];_y=xi[1];_z=xi[2];
//         ipair=ENVIRON::pairtype[ENVIRON::typ_mol[bs]];
//         for(int j=bs+1;j<jmax;++j)
//         {
//             xj=ENVIRON::x_mol[j];
//             if((rsq=sqr(_x-xj[0])+sqr(_y-xj[1])+sqr(_z-xj[2]))<cutsq && (bin=std::sqrt(rsq)*r_dr)<nbin_r)
//             {
//                 ++cr_intra[tpj=ENVIRON::typ_mol[j]][bin];
//                 ++gr_intra[ipair[tpj]][bin];
//             }
//         }
//         //bs<i<jmax
//         for(int i=bs+1;i<jmax;++i)
//         {
//             xi=ENVIRON::x_mol[i];
//             _x=xi[0];_y=xi[1];_z=xi[2];
//             ipair=ENVIRON::pairtype[ENVIRON::typ_mol[i]];
//             for(int j=i+1;j<jmax;++j)
//             {
//                 xj=ENVIRON::x_mol[j];
//                 if((rsq=sqr(_x-xj[0])+sqr(_y-xj[1])+sqr(_z-xj[2]))<cutsq && (bin=std::sqrt(rsq)*r_dr)<nbin_r)
//                 {
//                     ++gr_intra[ipair[ENVIRON::typ_mol[j]]][bin];
//                 }
//             }
//         }
//         bs+=nat_mol;
//     }
    
//     numtype temp,*ift;
//     for(int ti=0;ti<ENVIRON::ntype;++ti)
//     {
//         xi=cr_intra[ti];
//         for(int i=0;i<nbin_r;++i) xi[i]/=ENVIRON::nmol;
//         xj=cq_intra[ti];
//         for(int iq=0;iq<nbin_q;++iq)
//         {
//             temp=0;
//             ift=sincqr[iq];
//             for(int ir=0;ir<nbin_r;++ir)
//             {
//                 temp+=ift[ir]*xi[ir];
//             }
//             xj[iq]=temp;
//         }
//     }
//     xj=cq_intra[ENVIRON::typ_mol[0]];
//     for(int i=0;i<nbin_r;++i) xj[i]+=1;

//     for(int ti=0;ti<ENVIRON::npair;++ti)
//     {
//         xi=gr_intra[ti];
//         for(int i=0;i<nbin_r;++i) xi[i]/=ENVIRON::nmol;
//         xj=sq_intra[ti];
//         for(int iq=0;iq<nbin_q;++iq)
//         {
//             temp=0;
//             ift=sincqr[iq];
//             for(int ir=0;ir<nbin_r;++ir)
//             {
//                 temp+=ift[ir]*xi[ir];
//             }
//             xj[iq]=temp;
//         }
//     }
//     tpj=0;
//     for(int ti=0;ti<ENVIRON::ntype;++ti)
//     {
//         xj=sq_intra[tpj];
//         for(int ir=0;ir<nbin_q;++ir)
//         {
//             xj[ir]*=2;
//         }
//         tpj+=(ENVIRON::ntype-ti);
//     }

//     // gr of mol
//     for(int i=0;i<nbin_r;++i) gr[i]=0;
//     for(int i=0;i<ENVIRON::nmol;++i)
//     {
//         xi=ENVIRON::x_mol[ENVIRON::mol_repre_id[i]];
//         for(int j=i+1;j<ENVIRON::nmol;++j)
//         {
//             xj=ENVIRON::x_mol[ENVIRON::mol_repre_id[j]];
//             if((rsq=ENVIRON::distsq(xi,xj))<cutsq && (bin=std::sqrt(rsq)*r_dr)<nbin_r)
//             {
//                 ++gr[bin];
//             }
//             // if(rsq<1)
//             // {
//             //     std::cerr<<i<<" "<<j<<" "<<xi[0]<<" "<<xi[1]<<" "<<xi[2]<<" "<<xj[0]<<" "<<xj[1]<<" "<<xj[2]<<"\n";
//             // }
//         }
//     }
//     for(int i=0;i<nbin_r;++i)
//         gr[i]=gr[i]*norm[i]-1;
//         // gr[i]= rr[i]<10? 0 :  gr[i]*norm[i]-1;

//     xi=sq[0];
//     for(int iq=0;iq<nbin_q;++iq)
//     {
//         temp=0;
//         ift=ftmx[iq];
//         for(int ir=0;ir<nbin_r;++ir)
//         {
//             temp+=ift[ir]*gr[ir];
//         }
//         xi[iq]=temp;
//     }

//     bin=1;
//     numtype *cqi,*cqj;
//     for(int ti=0;ti<ENVIRON::ntype;++ti)
//     {
//         cqi=cq_intra[ti];
//         for(int tj=ti;tj<ENVIRON::ntype;++tj)
//         {
//             cqj=cq_intra[tj];
//             xj=sq[bin++];
//             for(int iq=0;iq<nbin_q;++iq)
//             {
//                 xj[iq]=(xi[iq])*cqi[iq]*cqj[iq];
//             }
//         }
//     }

//     // final conv
//     for(int ti=0;ti<=ENVIRON::npair;++ti)
//     {
//         xi=sq[ti];
//         for(int i=0;i<nbin_q;++i)
//         {
//             ift=conv_ker[i];
//             temp=0;
//             for(int j=0;j<nbin_q;++j)
//             {
//                 temp+=ift[j]*xi[j];
//             }
//             gr[i]=temp;
//         }
//         memcpy(xi,gr,nbin_q*sizeof(numtype));
//     }
// }

// void SQM::dump(const char *fname)
// {
//     FILE *fp=fopen(fname,"w");
//     for(int i=0;i<nbin_q;++i)
//     {
//         fprintf(fp,"%.15e",qq[i]);
//         for(int p=0;p<=ENVIRON::npair;++p)
//             fprintf(fp," %.15e",sq[p][i]);
//         for(int p=0;p<ENVIRON::npair;++p)
//             fprintf(fp," %.15e",sq_intra[p][i]);
//         fprintf(fp,"\n");
//     }
//     fclose(fp);
// }
*/

//file format: q sq wt f[0] f[1] ... f[ntypes-1]
RMDF::RMDF(int nexp, char **fexp, SQ *c_sq, numtype strength, numtype _gamma, LJTableCoulDSF *pair, NNPOTENTIAL_TBL *pair_nnp):
potmx(nullptr)
{
    if(_gamma==1) std::cerr<<"DA/RMD\n";
    else if (_gamma>0 && _gamma<1)  std::cerr<<"FMIRL training\n";
    else if (_gamma==0)  std::cerr<<"FMIRL inference\n";
    else END_PROGRAM("invalid gamma");
    
    n_sq=nexp;
    gamma=_gamma;
    GR *c_gr=c_sq->gr;
    nbin_r=c_gr->nbin_r_inner;
    nbin_q=c_sq->nbin_q;
    partialsq=c_sq->sq;
    rr=c_gr->rr;
    qq=c_sq->qq;
    if(pair)
    {
#ifdef RMDF_IGNORE_SPECIAL
        std::cerr<<"Note: RMDF ignores special bonds\n";
        force_pr=pair->force_nsp;
        potential_pr=pair->potential_nsp;
#else
        force_bs=pair->force_base;
        force_pr=pair->force;
        potential_bs=pair->potential_base;
        potential_pr=pair->potential;
#endif
        if(pair->nbin_r!=nbin_r || std::fabs(pair->dr-c_gr->dr)>EPSILON) 
        {
            // std::cerr<<pair->nbin_r<<" "<<nbin_r<<" "<<pair->dr<<" "<<c_gr->dr<<"\n";
            END_PROGRAM("RMDF mismatch");
        }
    }
    else
    {
#ifdef RMDF_IGNORE_SPECIAL
        std::cerr<<"Note: RMDF ignores special bonds\n";
#endif
        force_bs=pair_nnp->tbl_f_pair_base[0];
        force_pr=pair_nnp->tbl_f_pair[0];
        potential_bs=pair_nnp->tbl_pair_base[0];
        potential_pr=pair_nnp->tbl_pair[0];
        if(pair_nnp->nbin_r!=nbin_r || std::fabs(pair_nnp->dr-c_gr->dr)>EPSILON) 
        {
            // std::cerr<<pair->nbin_r<<" "<<nbin_r<<" "<<pair->dr<<" "<<c_gr->dr<<"\n";
            END_PROGRAM("RMDF mismatch");
        }

    }

    create2DArray(ftmx,nbin_r,nbin_q);
    create2DArray(potmx,nbin_r,nbin_q);
    create2DArray(sqex,n_sq,nbin_q);
    create2DArray(sq,n_sq,nbin_q);
    create2DArray(deltasq,n_sq,nbin_q);
    create2DArray(deltasq_accu,n_sq,nbin_q);
    create2DArray(wt,n_sq,nbin_q);
    create3DArray(sqwt,n_sq,ENVIRON::npair,nbin_q);
    create3DArray(fwt,n_sq,ENVIRON::npair,nbin_q);
    create2DArray(force_q,ENVIRON::npair,nbin_q);


    numtype qi,cifi2,fi,*ff=new numtype[ENVIRON::ntype],natsq=sqr(ENVIRON::natom),ni;
    int ipr;
    numtype rri;
    FILE *fp;
    char *fexpi;
    numtype *sqexi,*wti,**sqwti,**fwti,*deltasqi,*deltasqai;
    for(int n=0;n<n_sq;++n)
    {
        fexpi=fexp[n];
        sqexi=sqex[n];
        wti=wt[n];
        sqwti=sqwt[n];
        fwti=fwt[n];
        deltasqi=deltasq[n];
        deltasqai=deltasq_accu[n];

        std::cerr<<"reading "<<fexpi<<"\n";
        fp=fopen(fexpi,"r");
        if(!fp)
        {
            END_PROGRAM("read error");
            // numtype *sqpr,*fwpr;
            // for(ipr=0;ipr<ENVIRON::npair;++ipr)
            // {
            //     sqpr=sqwti[ipr];
            //     fwpr=fwti[ipr];
            //     for(int i=0;i<nbin_q;++i)
            //         sqpr[i]=fwpr[i]=0;
            // }
            // for(int i=0;i<nbin_q;++i)
            //     sqexi[i]=wti[i]=deltasqi[i]=deltasqai[i]=0;
            // continue;
        }

        for(int i=0;i<nbin_q;++i)
        {
            deltasqi[i]=deltasqai[i]=0;
            fscanf(fp,FMT_NUMTYPE " ", &qi);
            if(std::fabs(qi-qq[i])>EPSILON)
                std::cerr<<"warning: large diff: "<<qi<<" "<<qq[i]<<"\n";
            fscanf(fp,FMT_NUMTYPE " ", sqexi+i);
            fscanf(fp,FMT_NUMTYPE " ", wti+i);
            wti[i]*=strength;

            cifi2=0;
            for(int j=0;j<ENVIRON::ntype;++j)
            {
                fscanf(fp,FMT_NUMTYPE " ", ff+j);
                cifi2+=ff[j]*ENVIRON::typecount[j];
            }
            cifi2=sqr(cifi2/ENVIRON::natom);

            ipr=0;
            for(int j=0;j<ENVIRON::ntype;++j)
            {
                fi=ff[j]; ni=ENVIRON::typecount[j]/natsq;
                for(int k=j;k<ENVIRON::ntype;++k)
                {
                    sqwti[ipr][i]=(fwti[ipr][i]=fi*ff[k]/cifi2)*ENVIRON::typecount[k]*ni * (k==j? 1 : 2);
                    fwti[ipr][i]*=wti[i]; //absorp wt into fwt
                    ++ipr;
                }
            }
        }
        fclose(fp);
        std::cerr<<"last entry: "<<qi<<" "<<sqexi[nbin_q-1]<<" "<<wti[nbin_q-1]<<"\n";
    }
    delete[] ff;

    numtype *pp,j0;
    for(int i=0;i<nbin_r;++i)
    {
        fi=1/sqr(rri=rr[i]);
        ff=ftmx[i];
        pp=potmx[i];
        for(int j=0;j<nbin_q;++j)
        {
            qi=qq[j]*rri;
            // no dq here! because it is included in wt
            pp[j]=(j0=std::sin(qi)/qi);
            ff[j]=fi*(j0-std::cos(qi));
        }
    }

    create1DArray(ibegin_r,ENVIRON::NUM_THREAD);
    create1DArray(ibegin_q,ENVIRON::NUM_THREAD);
    create1DArray(iend_r,ENVIRON::NUM_THREAD);
    create1DArray(iend_q,ENVIRON::NUM_THREAD);

    for(int i=1;i<ENVIRON::NUM_THREAD;++i)
    {
        iend_q[i-1]=ibegin_q[i]=i*nbin_q/ENVIRON::NUM_THREAD;
        iend_r[i-1]=ibegin_r[i]=i*nbin_r/ENVIRON::NUM_THREAD;
    }
    ibegin_q[0]=0; iend_q[ENVIRON::NUM_THREAD-1]=nbin_q;
    ibegin_r[0]=0; iend_r[ENVIRON::NUM_THREAD-1]=nbin_r;
}

RMDF::~RMDF()
{
    destroy2DArray(ftmx);
    destroy2DArray(potmx);
    destroy2DArray(sqex);
    destroy2DArray(sq);
    destroy2DArray(deltasq);
    destroy2DArray(deltasq_accu);
    destroy2DArray(wt);
    destroy3DArray(sqwt);
    destroy3DArray(fwt);
    destroy2DArray(force_q);
    destroy1DArray(ibegin_r);
    destroy1DArray(ibegin_q);
    destroy1DArray(iend_r);
    destroy1DArray(iend_q);
}

numtype RMDF::compute_Qspace(int thread_id)
{
    numtype erg_local=0;
    numtype *psqt,*psqwt,dsqi;
    int iqs=ibegin_q[thread_id],iqe=iend_q[thread_id];
    for(int ti=0;ti<ENVIRON::npair;++ti)
    {
        psqt=force_q[ti];
        for(int i=iqs;i<iqe;++i) psqt[i]=0;
    }
    numtype *sqi,*sqexi,*wti,*deltasqi,**fwti,**sqwti;
    for(int n=0;n<n_sq;++n)
    {
        sqi=sq[n];
        sqexi=sqex[n];
        wti=wt[n];
        deltasqi=deltasq[n];
        fwti=fwt[n];
        sqwti=sqwt[n];

        for(int i=iqs;i<iqe;++i) sqi[i]=0;
        for(int ti=0;ti<ENVIRON::npair;++ti)
        {
            psqt=partialsq[ti]; psqwt=sqwti[ti];
            for(int i=iqs;i<iqe;++i) sqi[i]+=psqt[i]*psqwt[i];
        }
        for(int i=iqs;i<iqe;++i) 
        {
            erg_local+=sqr(dsqi=sqi[i]-sqexi[i])*wti[i];
            deltasqi[i]+=gamma*(dsqi-deltasqi[i]);
        }

        for(int ti=0;ti<ENVIRON::npair;++ti)
        {
            psqt=force_q[ti]; psqwt=fwti[ti];
            for(int i=iqs;i<iqe;++i) psqt[i]+=deltasqi[i]*psqwt[i];
        }
    }
    return erg_local*ENVIRON::natom*.25F;
}

void RMDF::compute_Rspace(int thread_id)
{
    int nrs=ibegin_r[thread_id],nre=iend_r[thread_id];
    // std::cerr<<nrs<<" "<<nre<<std::endl;
#ifdef GRSQ_USE_BLAS

#ifdef RMDF_IGNORE_SPECIAL
    cblas_matmul(CblasRowMajor,CblasNoTrans,CblasTrans,ENVIRON::npair,nre-nrs,nbin_q,1.0,force_q[0],nbin_q,ftmx[nrs],nbin_q,0,force_pr[0]+nrs,nbin_r+1);
#else
    nre-=nrs;
    for(int i=0;i<ENVIRON::npair;++i) memcpy(force_pr[i]+nrs,force_bs[i]+nrs,nre*sizeof(numtype));
    cblas_matmul(CblasRowMajor,CblasNoTrans,CblasTrans,ENVIRON::npair,nre,nbin_q,1.0,force_q[0],nbin_q,ftmx[nrs],nbin_q,1,force_pr[0]+nrs,nbin_r+1);
#endif

#else

    numtype *ifr,*ifq,*ift,temp;
#ifndef RMDF_IGNORE_SPECIAL
    numtype *ifb;
#endif
    for(int i=0;i<ENVIRON::npair;++i)
    {
        ifq=force_q[i];
        ifr=force_pr[i];
#ifndef RMDF_IGNORE_SPECIAL
        ifb=force_bs[i];
#endif
        for(int ir=nrs;ir<nre;++ir)
        {
#ifdef RMDF_IGNORE_SPECIAL
            temp=0;
#else
            temp=ifb[ir];
#endif
            ift=ftmx[ir];
            for(int iq=0;iq<nbin_q;++iq)
            {
                temp+=ifq[iq]*ift[iq];
            }
            ifr[ir]=temp;
        }
    }

#endif
}

void RMDF::compute_Potential(int thread_id)
{
    int nrs=ibegin_r[thread_id],nre=iend_r[thread_id];
    // std::cerr<<nrs<<" "<<nre<<std::endl;
#ifdef GRSQ_USE_BLAS

#ifdef RMDF_IGNORE_SPECIAL
    cblas_matmul(CblasRowMajor,CblasNoTrans,CblasTrans,ENVIRON::npair,nre-nrs,nbin_q,1.0,force_q[0],nbin_q,potmx[nrs],nbin_q,0,potential_pr[0]+nrs,nbin_r+1);
#else
    nre-=nrs;
    for(int i=0;i<ENVIRON::npair;++i) memcpy(potential_pr[i]+nrs,potential_bs[i]+nrs,nre*sizeof(numtype));
    cblas_matmul(CblasRowMajor,CblasNoTrans,CblasTrans,ENVIRON::npair,nre,nbin_q,1.0,force_q[0],nbin_q,potmx[nrs],nbin_q,1,potential_pr[0]+nrs,nbin_r+1);
#endif

#else

    numtype *ifr,*ifq,*ift,temp;
#ifndef RMDF_IGNORE_SPECIAL
    numtype *ifb;
#endif
    for(int i=0;i<ENVIRON::npair;++i)
    {
        ifq=force_q[i];
        ifr=potential_pr[i];
#ifndef RMDF_IGNORE_SPECIAL
        ifb=potential_bs[i];
#endif
        for(int ir=nrs;ir<nre;++ir)
        {
#ifdef RMDF_IGNORE_SPECIAL
            temp=0;
#else
            temp=ifb[ir];
#endif
            ift=potmx[ir];
            for(int iq=0;iq<nbin_q;++iq)
            {
                temp+=ifq[iq]*ift[iq];
            }
            ifr[ir]=temp;
        }
    }

#endif
}


void RMDF::compute_Rspace_NNP(int thread_id)
{
    int nrs=ibegin_r[thread_id],nre=iend_r[thread_id];
#ifdef RMDF_IGNORE_SPECIAL
    numtype *ifr12,*ifb12,*ifr13,*ifb13;
#else
    constexpr numtype lj14=MOLECULE::special_lj[3];
#endif
    numtype *ifr14,*ifb14;
    numtype *ifr,*ifb,temp;
#ifdef GRSQ_USE_BLAS

    cblas_matmul(CblasRowMajor,CblasNoTrans,CblasTrans,ENVIRON::npair,nre-nrs,nbin_q,1.0,force_q[0],nbin_q,ftmx[nrs],nbin_q,0,force_pr[0]+nrs,4*(nbin_r+1));
    for(int i=0,ib=0;i<ENVIRON::npair;++i,++ib)
    {
        ifr=force_pr[ib];
        ifb=force_bs[ib];
#ifdef RMDF_IGNORE_SPECIAL
        ifr12=force_pr[++ib];
        ifb12=force_bs[ib];
        ifr13=force_pr[++ib];
        ifb13=force_bs[ib];
        ifr14=force_pr[++ib];
        ifb14=force_bs[ib];
#else
        ifr14=force_pr[ib+=3];
        ifb14=force_bs[ib];
#endif
        for(int ir=nrs;ir<nre;++ir)
        {
            temp=ifr[ir];
            ifr[ir]+=ifb[ir];
#ifdef RMDF_IGNORE_SPECIAL
            ifr12[ir]=temp+ifb12[ir];
            ifr13[ir]=temp+ifb13[ir];
            ifr14[ir]=temp+ifb14[ir];
#else
            ifr14[ir]=temp*lj14+ifb14[ir];
#endif
        }
    }

#else

    numtype *ifq,*ift;
    for(int i=0,ib=0;i<ENVIRON::npair;++i,++ib)
    {
        ifq=force_q[i];
        ifr=force_pr[ib];
        ifb=force_bs[ib];
#ifdef RMDF_IGNORE_SPECIAL
        ifr12=force_pr[++ib];
        ifb12=force_bs[ib];
        ifr13=force_pr[++ib];
        ifb13=force_bs[ib];
        ifr14=force_pr[++ib];
        ifb14=force_bs[ib];
#else
        ifr14=force_pr[ib+=3];
        ifb14=force_bs[ib];
#endif
        for(int ir=nrs;ir<nre;++ir)
        {
            temp=0;
            ift=ftmx[ir];
            for(int iq=0;iq<nbin_q;++iq)
            {
                temp+=ifq[iq]*ift[iq];
            }

            ifr[ir]=temp+ifb[ir];
#ifdef RMDF_IGNORE_SPECIAL
            ifr12[ir]=temp+ifb12[ir];
            ifr13[ir]=temp+ifb13[ir];
            ifr14[ir]=temp+ifb14[ir];
#else
            ifr14[ir]=temp*lj14+ifb14[ir];
#endif
        }
    }
#endif
}

void RMDF::compute_Potential_NNP(int thread_id)
{
    int nrs=ibegin_r[thread_id],nre=iend_r[thread_id];
#ifdef RMDF_IGNORE_SPECIAL
    numtype *ifr12,*ifb12,*ifr13,*ifb13;
#else
    constexpr numtype lj14=MOLECULE::special_lj[3];
#endif
    numtype *ifr14,*ifb14;
    numtype *ifr,*ifb,temp;
#ifdef GRSQ_USE_BLAS
    cblas_matmul(CblasRowMajor,CblasNoTrans,CblasTrans,ENVIRON::npair,nre-nrs,nbin_q,1.0,force_q[0],nbin_q,potmx[nrs],nbin_q,0,potential_pr[0]+nrs,4*(nbin_r+1));
    for(int i=0,ib=0;i<ENVIRON::npair;++i,++ib)
    {
        ifr=potential_pr[ib];
        ifb=potential_bs[ib];
#ifdef RMDF_IGNORE_SPECIAL
        ifr12=potential_pr[++ib];
        ifb12=potential_bs[ib];
        ifr13=potential_pr[++ib];
        ifb13=potential_bs[ib];
        ifr14=potential_pr[++ib];
        ifb14=potential_bs[ib];
#else
        ifr14=potential_pr[ib+=3];
        ifb14=potential_bs[ib];
#endif
        for(int ir=nrs;ir<nre;++ir)
        {
            temp=ifr[ir];
            ifr[ir]+=ifb[ir];
#ifdef RMDF_IGNORE_SPECIAL
            ifr12[ir]=temp+ifb12[ir];
            ifr13[ir]=temp+ifb13[ir];
            ifr14[ir]=temp+ifb14[ir];
#else
            ifr14[ir]=temp*lj14+ifb14[ir];
#endif
        }
    }

#else

    numtype *ifq,*ift;
    for(int i=0,ib=0;i<ENVIRON::npair;++i,++ib)
    {
        ifq=force_q[i];
        ifr=potential_pr[ib];
        ifb=potential_bs[ib];
#ifdef RMDF_IGNORE_SPECIAL
        ifr12=potential_pr[++ib];
        ifb12=potential_bs[ib];
        ifr13=potential_pr[++ib];
        ifb13=potential_bs[ib];
        ifr14=potential_pr[++ib];
        ifb14=potential_bs[ib];
#else
        ifr14=potential_pr[ib+=3];
        ifb14=potential_bs[ib];
#endif
        for(int ir=nrs;ir<nre;++ir)
        {
            temp=0;
            ift=potmx[ir];
            for(int iq=0;iq<nbin_q;++iq)
            {
                temp+=ifq[iq]*ift[iq];
            }

            ifr[ir]=temp+ifb[ir];
#ifdef RMDF_IGNORE_SPECIAL
            ifr12[ir]=temp+ifb12[ir];
            ifr13[ir]=temp+ifb13[ir];
            ifr14[ir]=temp+ifb14[ir];
#else
            ifr14[ir]=temp*lj14+ifb14[ir];
#endif
        }
    }

#endif
}

// format: q sq_i_acc deltasq_i deltasq_i_acc ... 
void RMDF::dumpSQ(const char *fname, bool append)
{
    int num=1;
    FILE *fp;
    numtype factor=1,cofactor=0,sqi;
    if(append && (fp=fopen(fname,"r")))
    {
        fscanf(fp,"# %d ",&num);
        num=std::min(num+1,MAXACC);
        cofactor=1-( factor=((numtype)1)/(num) );
        std::cerr<<"rmdf_acc = "<<num<<"\n";
        for(int i=0;i<nbin_q;++i)
        {
            fscanf(fp,FMT_NUMTYPE " ",&sqi);
            if(std::fabs(sqi-qq[i])>EPSILON) std::cerr<<"warning: large diff: "<<sqi<<" "<<qq[i]<<"\n";
            for(int n=0;n<n_sq;++n)
            {
                fscanf(fp,FMT_NUMTYPE " %*f " FMT_NUMTYPE " ",&sqi,deltasq_accu[n]+i);
                sq[n][i]=sq[n][i]*factor+cofactor*sqi;
            }
        }
        fclose(fp);
    }

    fp=fopen(fname,"w");
    fprintf(fp,"# %d\n",num);
    for(int i=0;i<nbin_q;++i)
    {
        fprintf(fp,"%.15e",qq[i]);
        for(int n=0;n<n_sq;++n)
        {
            fprintf(fp," %.15e %.15e %.15e",sq[n][i],deltasq[n][i],deltasq[n][i]*factor+cofactor*deltasq_accu[n][i]);
        }
        fprintf(fp,"\n");
    }
    fclose(fp);
}

// will load deltasq_i_acc if in fmirl inference mode
void RMDF::loadDeltaSQ(const char *fname)
{
    numtype qi;
    std::cerr<<"reading "<<fname<<"\n";
    FILE *fp=fopen(fname,"r");
    if(!fp) // does not throw error here because this data is sometimes not needed
    {
        std::cerr<<"cannot open file\n";
        return;
    }
    fscanf(fp,"# %*d ");
    const char *fmt= gamma==0? "%*f %*f " FMT_NUMTYPE " " : "%*f " FMT_NUMTYPE " %*f ";
    for(int i=0;i<nbin_q;++i)
    {
        fscanf(fp,FMT_NUMTYPE " ",&qi);
        if(std::fabs(qi-qq[i])>EPSILON) std::cerr<<"warning: large diff: "<<qi<<" "<<qq[i]<<"\n";
        for(int n=0;n<n_sq;++n)
        {
            fscanf(fp,fmt,deltasq[n]+i);
        }
    }

    fclose(fp);
    std::cerr<<"last entry: "<<qi<<" "<<deltasq[n_sq-1][nbin_q-1]<<"\n";
}
