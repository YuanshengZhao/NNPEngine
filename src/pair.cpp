#include "pair.h"
#include "memory.h"
#include "mathlib.h"
#include "util.h"
#include <iostream>
#include <cstring>

LJCutCoulDSF::LJCutCoulDSF(numtype cf,numtype alp) :
_48Epsilon(nullptr), sigma(nullptr), sigmasq(nullptr)
{
    cutsq=sqr(cutoff=cf);
    ntyp=ENVIRON::ntype;
    create2DArray(_48Epsilon,ntyp,ntyp);
    create2DArray(sigma,ntyp,ntyp);
    create2DArray(sigmasq,ntyp,ntyp);
    create2DArray(shift,ntyp,ntyp);
    for(int i=0;i<ntyp;++i)
        for(int j=0;j<ntyp;++j)
            shift[i][j]=shift[j][i]=_48Epsilon[i][j]=sigmasq[i][j]=0;

#ifdef DSF_ERFC
    alphasq=sqr(alpha=alp);
    _2alpha_sqrtpi=2*alpha/std::sqrt(ENVIRON::pi);
    vc=-std::erfc(alpha*cutoff)/cutoff;
    fc=vc/cutoff-_2alpha_sqrtpi*std::exp(-sqr(alpha*cutoff))/cutoff;
#else
    std::cerr<<"Note: erfc is disabled\n";
    vc=-1/cutoff;
    fc=vc/cutoff;
#endif
}

LJCutCoulDSF::~LJCutCoulDSF()
{
    destroy2DArray(_48Epsilon);
    destroy2DArray(sigma);
    destroy2DArray(sigmasq);
    destroy2DArray(shift);
}

void LJCutCoulDSF::setParam(int it,int jt,numtype ep,numtype sg)
{
    if(it<0 || it>=ntyp || jt<0 ||jt>=ntyp) END_PROGRAM("invalid type");
    _48Epsilon[it][jt]=_48Epsilon[jt][it]=ep*4*12; // priority for computing force;
    sigma[it][jt]=sigma[jt][it]=sg;
    sigmasq[it][jt]=sigmasq[jt][it]=sg*sg;
    numtype sr6=cub(sqr(sg/cutoff));
    shift[it][jt]=shift[jt][it]=sr6*(1-sr6);
}

void LJCutCoulDSF::mixParam(char ty)
{
    numtype sr6;
    for(int i=0;i<ntyp;++i)
        for(int j=i+1;j<ntyp;++j)
        {
            _48Epsilon[i][j]=_48Epsilon[j][i]=std::sqrt(_48Epsilon[i][i]*_48Epsilon[j][j]);
            if (ty=='G')
            {
                sigma[i][j]=sigma[j][i]=std::sqrt(sigma[i][i]*sigma[j][j]);
                sigmasq[i][j]=sigmasq[j][i]=std::sqrt(sigmasq[i][i]*sigmasq[j][j]);
                sr6=cub(sqr(sigma[i][j]/cutoff));
                shift[i][j]=shift[j][i]=sr6*(1-sr6);

            }
            else if (ty=='A')
            {
                sigma[i][j]=sigma[j][i]=0.5F*(sigma[i][i]+sigma[j][j]);
                sigmasq[i][j]=sigmasq[j][i]=sqr(sigma[i][j]);
                sr6=cub(sqr(sigma[i][j]/cutoff));
                shift[i][j]=shift[j][i]=sr6*(1-sr6);
            }
            else END_PROGRAM("invalid mix type");
        }
}

template void LJCutCoulDSF::compute<true>(NEIGHLIST *list, numtype *erg, numtype *viral);
template void LJCutCoulDSF::compute<false>(NEIGHLIST *list, numtype *erg, numtype *viral);

template <bool ev> void LJCutCoulDSF::compute(NEIGHLIST *list, numtype *erg, numtype *viral)
{
    numtype *xi,*xj;
    numtype *fi,*fj;
    int n_local=list->n_local, **nlist=list->nei_list,*num_nei=list->num_neigh;
    int *local_list=list->local_list;
    int ti,tj,inum,*ilist,ii,jj;
    int thread_id=list->thread_id;
    int *ispecial, **special=list->special,spij;
    numtype sdv6, rsq, dx, dy, dz, fpair, fx,fy,fz, _xi,_yi,_zi;
    numtype *sigmasqi,*fepsiloni,*shifti;
    numtype qi,r;

    for(int i=0;i<n_local;++i)
    {
        ii=local_list[i];
        xi=ENVIRON::x[ii];
        _xi=xi[0];_yi=xi[1];_zi=xi[2];
        qi=ENVIRON::q[ii];

        ilist=nlist[i];
        inum=num_nei[i];
        ispecial=special[i];

        ti=ENVIRON::typ[ii];
        sigmasqi=sigmasq[ti];
        fepsiloni=_48Epsilon[ti];

        fx=fy=fz=0;

        if(ev) shifti=shift[ti];

        for(int j=0; j<inum; ++j)
        {
            jj=ilist[j];
            xj=ENVIRON::x[jj];

            rsq=sqr(dx=_xi-xj[0])+sqr(dy=_yi-xj[1])+sqr(dz=_zi-xj[2]);
            if(rsq<cutsq)
            {
                tj=ENVIRON::typ[jj];
                spij=ispecial[j];

                sdv6=cub(sigmasqi[tj]/rsq);
                fpair=fepsiloni[tj]/rsq*(sdv6*(sdv6-.5F)) * MOLECULE::special_lj[spij];

                r=std::sqrt(rsq);
#ifdef DSF_ERFC
                fpair+=qi*ENVIRON::q[jj]*ENVIRON::electricK * (std::erfc(alpha*r)/(r*rsq) + _2alpha_sqrtpi*std::exp(-alphasq*rsq)/rsq + fc/r) *MOLECULE::special_coul[spij];
#else
                fpair+=qi*ENVIRON::q[jj]*ENVIRON::electricK * (1/(r*rsq) + fc/r) *MOLECULE::special_coul[spij];
#endif

                fx+=(dx*=fpair);fy+=(dy*=fpair);fz+=(dz*=fpair);
                if(ENVIRON::x_thr[jj]==thread_id)
                {
                    fj=ENVIRON::f[jj];
                    fj[0]-=dx;fj[1]-=dy;fj[2]-=dz;

                    if(ev) 
                    {
                        *erg+=fepsiloni[tj]/12*(sdv6*(sdv6-1)+shifti[tj]) * MOLECULE::special_lj[spij];
#ifdef DSF_ERFC
                        *erg+=qi*ENVIRON::q[jj]*ENVIRON::electricK * (std::erfc(alpha*r)/r +vc+ fc*(cutoff-r)) *MOLECULE::special_coul[spij];
#else
                        *erg+=qi*ENVIRON::q[jj]*ENVIRON::electricK * (1/r +vc+ fc*(cutoff-r)) *MOLECULE::special_coul[spij];
#endif
                        *viral+=fpair*rsq;
                    }
                }
                else if(ev)
                {
                    *erg+=0.5F*fepsiloni[tj]/12*(sdv6*(sdv6-1)+shifti[tj]) * MOLECULE::special_lj[spij];
#ifdef DSF_ERFC
                    *erg+=0.5F*qi*ENVIRON::q[jj]*ENVIRON::electricK * (std::erfc(alpha*r)/r +vc+ fc*(cutoff-r)) *MOLECULE::special_coul[spij];
#else
                    *erg+=0.5F*qi*ENVIRON::q[jj]*ENVIRON::electricK * (1/r +vc+ fc*(cutoff-r)) *MOLECULE::special_coul[spij];
#endif
                    *viral+=0.5F*fpair*rsq;
                }
            }

        }
        
        fi=ENVIRON::f[ii];
        fi[0]+=fx;fi[1]+=fy;fi[2]+=fz;
    }
}

// potfile format: r F_00(r)/r U_00(r) ...
// note that force should be F/r instead of F !!!
LJTableCoulDSF::LJTableCoulDSF(numtype cf,int nbin):
force_nsp(nullptr), potential_nsp(nullptr)
{
    ntyp=ENVIRON::ntype;
    cutsq=sqr(cutoff=cf);
    nbin_r=nbin;
    dr=cutoff/nbin_r; r_dr=nbin_r/cutoff;
    
    // create nbin_r+1 so that boundary check is not needed.
    force=create2DArray(force_base,ENVIRON::npair,nbin_r+1);
    potential=create2DArray(potential_base,ENVIRON::npair,nbin_r+1);
    create1DArray(rr,nbin_r);
    // external code is responsible to correctly assign these two pointers when modifying the potential

#ifdef DSF_ERFC
    alphasq=sqr(alpha=alp);
    _2alpha_sqrtpi=2*alpha/std::sqrt(ENVIRON::pi);
    vc=-std::erfc(alpha*cutoff)/cutoff;
    fc=vc/cutoff-_2alpha_sqrtpi*std::exp(-sqr(alpha*cutoff))/cutoff;
#else
    std::cerr<<"Note: erfc is disabled\n";
    vc=-1/cutoff;
    fc=vc/cutoff;
#endif
}

void LJTableCoulDSF::loadPotential(int kind, const char *potfile)
{
    numtype **fload=nullptr,**pload=nullptr;
    switch (kind)
    {
    case 0:
        fload=force_base;
        pload=potential_base;
        break;
    case 1:
        fload=create2DArray(force,ENVIRON::npair,nbin_r+1);
        pload=create2DArray(potential,ENVIRON::npair,nbin_r+1);
        break;
    case 2:
        fload=create2DArray(force_nsp,ENVIRON::npair,nbin_r+1);
        pload=create2DArray(potential_nsp,ENVIRON::npair,nbin_r+1);
        break;
    default:
        END_PROGRAM("invalid arg");
        break;
    }

    if(potfile)
    {
        std::cerr<<"reading "<<potfile<<"\n";
        numtype rri,rie;
        FILE *fp=fopen(potfile,"r");
        for(int i=0;i<nbin_r;++i)
        {
            rie=rr[i]=(i+.5F)*dr;
            fscanf(fp,FMT_NUMTYPE " ", &rri);
            if(std::fabs(rri-rie)>EPSILON)
                std::cerr<<"warning: large diff: "<<rri<<" "<<rie<<"\n";
            for(int pp=0;pp<ENVIRON::npair;++pp)
            {
                fscanf(fp,FMT_NUMTYPE " ",fload[pp]+i);
                fscanf(fp,FMT_NUMTYPE " ",pload[pp]+i);
            }
        }
        fclose(fp);
        std::cerr<<"last entry: "<<rri<<" "<<force_base[0][nbin_r-1]<<" "<<potential_base[0][nbin_r-1]<<"\n";
        for(int pp=0;pp<ENVIRON::npair;++pp)
        {
            fload[pp][nbin_r]=0;
            pload[pp][nbin_r]=0;
        }
    }
    else if (kind==1)
    {
        auto sz=(nbin_r+1)*ENVIRON::npair*sizeof(numtype);
        memcpy(force[0],force_base[0],sz);
        memcpy(potential[0],potential_base[0],sz);
    }
    else if (kind==2)
    {
        numtype *f1,*p1;
        for(int pp=0;pp<ENVIRON::npair;++pp)
        {
            f1=force_nsp[pp];
            p1=potential_nsp[pp];
            for(int i=0;i<=nbin_r;++i)
            {
                f1[i]=p1[i]=0;
            }
        }
    }
}

LJTableCoulDSF::~LJTableCoulDSF()
{
    destroy1DArray(rr);
    //must check force first
    if(force != force_base)
    {
        destroy2DArray(force);
        destroy2DArray(potential);
    }
    destroy2DArray(force_base);
    destroy2DArray(potential_base);
    if(force_nsp)
    {
        destroy2DArray(force_nsp);
        destroy2DArray(potential_nsp);
    }
}

#define TBL_INTERP_LINEAR

template void LJTableCoulDSF::compute<true ,false>(NEIGHLIST *list, numtype *erg, numtype *viral);
template void LJTableCoulDSF::compute<false,false>(NEIGHLIST *list, numtype *erg, numtype *viral);
template void LJTableCoulDSF::compute<true ,true >(NEIGHLIST *list, numtype *erg, numtype *viral);
template void LJTableCoulDSF::compute<false,true >(NEIGHLIST *list, numtype *erg, numtype *viral);
template <bool ev, bool nsp> 
void LJTableCoulDSF::compute(NEIGHLIST *list, numtype *erg, numtype *viral)
{
    numtype *xi,*xj;
    numtype *fi,*fj;
    int n_local=list->n_local, **nlist=list->nei_list,*num_nei=list->num_neigh;
    int *local_list=list->local_list;
    int *ipair,inum,*ilist,ii,jj;
    int thread_id=list->thread_id;
    int *ispecial, **special=list->special,spij;
    numtype rsq, dx, dy, dz, fpair, fx,fy,fz, _xi,_yi,_zi;
    numtype qi,r;

    int tbr1,ijp;
#ifdef TBL_INTERP_LINEAR
    int tbr2;
    numtype ipw1,ipw2;
    numtype *ijf,*ijfn,*ijpo,*ijpon;
#endif

    for(int i=0;i<n_local;++i)
    {
        ii=local_list[i];
        xi=ENVIRON::x[ii];
        _xi=xi[0];_yi=xi[1];_zi=xi[2];
        qi=ENVIRON::q[ii];

        ilist=nlist[i];
        inum=num_nei[i];
        ispecial=special[i];

        ipair=ENVIRON::pairtype[ENVIRON::typ[ii]];

        fx=fy=fz=0;

        for(int j=0; j<inum; ++j)
        {
            jj=ilist[j];
            spij=ispecial[j];

            xj=ENVIRON::x[jj];

            rsq=sqr(dx=_xi-xj[0])+sqr(dy=_yi-xj[1])+sqr(dz=_zi-xj[2]);
            if(rsq<cutsq)
            {
                r=std::sqrt(rsq);
                ijp=ipair[ENVIRON::typ[jj]];
#ifdef TBL_INTERP_LINEAR
                ipw1=r*r_dr-0.5F;
                tbr2=1+(tbr1=(int)ipw1);
                ipw1=1-(ipw2=ipw1-tbr1);
                ijf=force[ijp];
                if(nsp) 
                {
                    ijfn=force_nsp[ijp];
                    fpair=(ijf[tbr1]*ipw1+ijf[tbr2]*ipw2) * MOLECULE::special_lj[spij]+(ijfn[tbr1]*ipw1+ijfn[tbr2]*ipw2);
                }
                else
                    fpair=(ijf[tbr1]*ipw1+ijf[tbr2]*ipw2) * MOLECULE::special_lj[spij];
#else //lookup
                tbr1=r*r_dr;
                if(nsp) fpair=force[ijp][tbr1] * MOLECULE::special_lj[spij]+force_nsp[ijp][tbr1];
                else  fpair=force[ijp][tbr1] * MOLECULE::special_lj[spij];
#endif
#ifdef DSF_ERFC
                fpair+=qi*ENVIRON::q[jj]*ENVIRON::electricK * (std::erfc(alpha*r)/(r*rsq) + _2alpha_sqrtpi*std::exp(-alphasq*rsq)/rsq + fc/r) *MOLECULE::special_coul[spij];
#else
                fpair+=qi*ENVIRON::q[jj]*ENVIRON::electricK * (1/(r*rsq) + fc/r) *MOLECULE::special_coul[spij];
#endif

                fx+=(dx*=fpair);fy+=(dy*=fpair);fz+=(dz*=fpair);
                if(ENVIRON::x_thr[jj]==thread_id)
                {
                    fj=ENVIRON::f[jj];
                    fj[0]-=dx;fj[1]-=dy;fj[2]-=dz;

                    if(ev) 
                    {
#ifdef TBL_INTERP_LINEAR
                        ijpo=potential[ijp];
                        if(nsp)
                        {
                            ijpon=potential_nsp[ijp];
                            *erg+=((ijpo[tbr1]*ipw1+ijpo[tbr2]*ipw2)) * MOLECULE::special_lj[spij] + (ijpon[tbr1]*ipw1+ijpon[tbr2]*ipw2);
                        }
                        else 
                            *erg+=((ijpo[tbr1]*ipw1+ijpo[tbr2]*ipw2)) * MOLECULE::special_lj[spij];
#else
                        if(nsp) *erg+=potential[ijp][tbr1] * MOLECULE::special_lj[spij] + potential_nsp[ijp][tbr1];
                        else *erg+=potential[ijp][tbr1] * MOLECULE::special_lj[spij];
#endif

#ifdef DSF_ERFC
                        *erg+=qi*ENVIRON::q[jj]*ENVIRON::electricK * (std::erfc(alpha*r)/r +vc+ fc*(cutoff-r)) *MOLECULE::special_coul[spij];
#else
                        *erg+=qi*ENVIRON::q[jj]*ENVIRON::electricK * (1/r +vc+ fc*(cutoff-r)) *MOLECULE::special_coul[spij];
#endif
                        *viral+=fpair*rsq;
                    }
                }
                else if(ev)
                {
#ifdef TBL_INTERP_LINEAR
                    ijpo=potential[ijp];
                    if(nsp)
                    {
                        ijpon=potential_nsp[ijp];
                        *erg+=0.5F*(((ijpo[tbr1]*ipw1+ijpo[tbr2]*ipw2)) * MOLECULE::special_lj[spij] + (ijpon[tbr1]*ipw1+ijpon[tbr2]*ipw2));
                    }
                    else 
                        *erg+=0.5F*((ijpo[tbr1]*ipw1+ijpo[tbr2]*ipw2)) * MOLECULE::special_lj[spij];
#else
                    if(nsp) *erg+=0.5F*(potential[ijp][tbr1] * MOLECULE::special_lj[spij] + potential_nsp[ijp][tbr1]);
                    else *erg+=0.5F*potential[ijp][tbr1] * MOLECULE::special_lj[spij];
#endif

#ifdef DSF_ERFC
                    *erg+=0.5F*qi*ENVIRON::q[jj]*ENVIRON::electricK * (std::erfc(alpha*r)/r +vc+ fc*(cutoff-r)) *MOLECULE::special_coul[spij];
#else
                    *erg+=0.5F*qi*ENVIRON::q[jj]*ENVIRON::electricK * (1/r +vc+ fc*(cutoff-r)) *MOLECULE::special_coul[spij];
#endif
                    *viral+=0.5F*fpair*rsq;
                }
            }

        }
        
        fi=ENVIRON::f[ii];
        fi[0]+=fx;fi[1]+=fy;fi[2]+=fz;
    }
}

void LJTableCoulDSF::dumpForce(const char *fname, bool nsp, bool append)
{
    FILE *fp=fopen(fname,append?"a+":"w");
    numtype **ff,**pp;
    if(nsp){ff=force_nsp;pp=potential_nsp;}
    else{ff=force;pp=potential;}

    for(int i=0;i<nbin_r;++i)
    {
        fprintf(fp,"%.15e",rr[i]);
        for(int p=0;p<ENVIRON::npair;++p)
            // fprintf(fp," %.15e",gr_local[0][p][i]*factor);
            fprintf(fp," %.15e %.15e",ff[p][i],pp[p][i]);
        fprintf(fp,"\n");
    }
    fclose(fp);
}