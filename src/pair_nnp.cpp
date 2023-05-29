#include "pair_nnp.h"
#include "util.h"
#include "memory.h"
#include "mathlib.h"
#include <algorithm>
#include <inttypes.h>

static_assert(sizeof(float)==4);
static_assert(sizeof(int)==4);
static_assert(NNPOTENTIAL::n_descriptor==NNPOTENTIAL_TBL::n_descriptor);

void TFCALL::init(const char* saved_model_dir,int n_in, char **in_names, int64_t **dim_in, TF_DataType *dtype,int *index_in,int n_out, char **out_names,int *index_out)
{
#ifndef FLOAT_PRECESION
    END_PROGRAM("NNP must use FLOAT_PRECESION");
#endif
    // Read model
    Graph = TF_NewGraph();
    Status = TF_NewStatus();

    SessionOpts = TF_NewSessionOptions();
    TF_Buffer* RunOpts = NULL;
    uint8_t config[] = {0x10, 0x1, 0x28, 0x1};
    TF_SetConfig(SessionOpts, (void*)config, 4, Status);

    if (TF_GetCode(Status)!= TF_OK) END_PROGRAM(TF_Message(Status));

    const char* tags = "serve"; // default model serving tag; can change in future
    int ntags = 1;

    Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_model_dir, &tags, ntags, Graph, NULL, Status);
    if(TF_GetCode(Status) != TF_OK) END_PROGRAM(TF_Message(Status));
    
    // Get input tensor, Allocate data
    //TODO : need to use saved_model_cli to read saved_model arch
    Input = new TF_Output[NumInputs=n_in];
    InputValues =  new TF_Tensor*[NumInputs ];
    inputs = new void*[NumInputs];

    int ndims;
    int ndata;
    int64_t *idim;

    for(int i=0;i<NumInputs;++i)
    {
        Input[i] = {TF_GraphOperationByName(Graph, in_names[i]), index_in[i]};
        if(Input[i].oper == NULL) END_PROGRAM("Failed TF_GraphOperationByName");
        std::cerr<<"input "<<i<<" dt"<<dtype[i]<<" "<<in_names[i]<<":"<<index_in[i]<<" ( ";

        ndata=1;
        idim=dim_in[i];
        for(ndims=0;idim[ndims];++ndims)
        {
            ndata*=idim[ndims];
            std::cerr<<idim[ndims]<<" ";
        }
        std::cerr<<")\n";
        if(ndata<0) 
        {
            inputs[i]=nullptr;
            continue;
        }
        switch (dtype[i])
        {
        case TF_FLOAT:
            ndata*=sizeof(numtype); // This is tricky, it number of bytes not number of element
            break;
        case TF_INT32:
            ndata*=sizeof(int);
            break;
        default:
            END_PROGRAM("unknown dtype");
            break;
        }
        InputValues[i]=TF_AllocateTensor(dtype[i],idim,ndims,ndata);
        inputs[i]=TF_TensorData(InputValues[i]);
    }
    
    // Get Output tensor, Allocate data
    Output = new TF_Output[NumOutputs=n_out];
    OutputValues = new TF_Tensor*[NumOutputs];
    outputs = new numtype*[NumOutputs];
    for(int i=0;i<NumOutputs;++i)
    {
        Output[i] = {TF_GraphOperationByName(Graph, out_names[i]), index_out[i]};
        if(Output[i].oper == NULL) END_PROGRAM("Failed TF_GraphOperationByName");
        std::cerr<<"output "<<i<<" "<<out_names[i]<<":"<<index_out[i]<<"\n";
    }

}

void TFCALL::evaluate()
{
    TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0,NULL , Status);
    if(TF_GetCode(Status) != TF_OK) END_PROGRAM(TF_Message(Status));
    for(int i=0;i<NumOutputs;++i) outputs[i]=(numtype*)TF_TensorData(OutputValues[i]);
}

void TFCALL::clearOutput()
{
    for(int i=0;i<NumOutputs;++i) TF_DeleteTensor(OutputValues[i]);
}

TFCALL::~TFCALL()
{
    TF_DeleteGraph(Graph);
    TF_DeleteStatus(Status);
    TF_DeleteSessionOptions(SessionOpts);
    for(int i=0;i<NumInputs;++i)
        if(inputs[i])
            TF_DeleteTensor(InputValues[i]);

    delete[] Input;
    delete[] Output;
    delete[] InputValues;
    delete[] OutputValues;
    delete[] inputs;
    delete[] outputs;
}

NNPOTENTIAL::NNPOTENTIAL(numtype cf,const char* paramfile,const char* wtfile, BOND *bnd, ANGLE* agl, DIHEDRAL *dih)
{
#ifndef FLOAT_PRECESION
    END_PROGRAM("NNP must use FLOAT_PRECESION");
#endif
    if(ENVIRON::nmoltype != 1) END_PROGRAM("NNP does not support multiple molecules");

    int ntyp=ENVIRON::ntype;
    cutsq=sqr(cutoff=cf);
    int szd=n_descriptor*4;
    create3DArray(coef_w,ntyp,ntyp,szd);
    create3DArray(coef_d,ntyp,ntyp,szd);
    create3DArray(coef_r,ntyp,ntyp,szd);
    create3DArray(coef_p,ntyp,ntyp,6*4);
    create3DArray(vfc,ntyp,ntyp,2*4);

    tfcall=new TFCALL[ENVIRON::nmoltype];
    char path_nn[SZ_FBF],dt; 
    char **names_in,**names_out;
    constexpr int n_input=1, n_output=2;
    TF_DataType dtype[n_input];
    int index_in[n_input], index_out[n_output];
    create2DArray(names_in,n_input,SZ_FBF);
    create2DArray(names_out,n_output,SZ_FBF);

    int64_t *dimi,**dim;
    create2DArray(dim,n_input,4);

    FILE *fp=fopen(paramfile,"r");
    if(fscanf(fp,"%s ",path_nn)!=1) END_PROGRAM("read error");
    for(int i=0;i<n_input;++i)
    {   
        dimi=dim[i];
        if(fscanf(fp,"%s %c %d %" SCNd64 " %" SCNd64 " %" SCNd64 " %" SCNd64 " ",names_in[i],&dt,index_in+i,dimi,dimi+1,dimi+2,dimi+3)!=7) END_PROGRAM("read error");
        switch (dt)
        {
        case 'f':
        case 'F':
            dtype[i]=TF_FLOAT;
            break;
        case 'i':
        case 'I':
            dtype[i]=TF_INT32;
            break;
        default:
            dtype[i]=TF_FLOAT;
            END_PROGRAM("unknown dtype");
            break;
        }
        if(*dimi==-2) *dimi=ENVIRON::nmol;
        // std::cerr<<i<<" "<<names_in[i]<<" "<<dt<<"\n";
    }

    for(int i=0;i<n_output;++i) if(fscanf(fp,"%s %d ",names_out[i],index_out+i)!=2) END_PROGRAM("read error");
    tfcall->init(path_nn,n_input,names_in,dim,dtype,index_in,n_output,names_out,index_out);
    fclose(fp);

    destroy2DArray(names_in);
    destroy2DArray(names_out);
    destroy2DArray(dim);
    
    descriptor=new numtype*[ENVIRON::natom];
    dedg=new numtype*[ENVIRON::natom];
    descriptor[0]=(numtype*)tfcall->inputs[0];
    for(int i=1;i<ENVIRON::natom;++i)
    {
        descriptor[i]=descriptor[i-1]+n_descriptor;
    }

    fp=fopen(wtfile,"rb");

    fread(*(bnd->coef),sizeof(numtype),bnd->nbtyp*4,fp);
    fread(*(agl->coef),sizeof(numtype),agl->natyp*4,fp);
    fread(*(dih->coef),sizeof(numtype),dih->ndtyp*4,fp);
    bnd->setParam();
    agl->setParam();
    dih->setParam();

    for(int i=0;i<ntyp;++i) fread(coef_p[i][i],sizeof(numtype),(ntyp-i)*6*4,fp);
    for(int i=0;i<ntyp;++i) fread(coef_w[i][i],sizeof(numtype),(ntyp-i)*szd,fp);
    for(int i=0;i<ntyp;++i) fread(coef_d[i][i],sizeof(numtype),(ntyp-i)*szd,fp);
    for(int i=0;i<ntyp;++i) fread(coef_r[i][i],sizeof(numtype),(ntyp-i)*szd,fp);
    numtype *tcpi,*tvfi,src2,src6;
    for(int i=0;i<ntyp;++i)
        for(int j=i;j<ntyp;++j)
        {
            for(int k=0;k<4;++k)
            {
                tcpi=coef_p[i][j]+k*6;
                tvfi=vfc[i][j]   +k*2;

                src2=sqr(tcpi[1]/cutoff);
                src6=cub(src2);
                tvfi[1]=-2*tcpi[5]/cutoff-tcpi[0]*src6*(-7+src2*(9*tcpi[2]+src2*(11*tcpi[3]+src2*13*tcpi[4]))); 
                tvfi[0]=(tcpi[5]/cutoff+tcpi[0]*src6*(-6+src2*((tcpi[2]*=8)+src2*((tcpi[3]*=10)+src2*(tcpi[4]*=12)))))/cutoff;
                tcpi[1]*=tcpi[1];
            }
        }
    for(int i=0;i<ntyp;++i)
        for(int j=i+1;j<ntyp;++j)
        {
            for(int k=0;k<2*4;++k) 
                vfc[j][i][k]=vfc[i][j][k];
            for(int k=0;k<6*4;++k) 
                coef_p[j][i][k]=coef_p[i][j][k];
            for(int k=0;k<szd;++k)
            {
                coef_w[j][i][k]=coef_w[i][j][k];
                coef_d[j][i][k]=coef_d[i][j][k];
                coef_r[j][i][k]=coef_r[i][j][k];
            }
        }
    fclose(fp);

}

NNPOTENTIAL::~NNPOTENTIAL()
{
    destroy3DArray(coef_w);
    destroy3DArray(coef_d);
    destroy3DArray(coef_r);
    destroy3DArray(coef_p);
    destroy3DArray(vfc);
    delete[] tfcall;
    delete[] descriptor;
    delete[] dedg;
}

void NNPOTENTIAL::computeDescriptor(NEIGHLIST *list)
{
    numtype *xi,*xj;
    int n_local=list->n_local, **nlist=list->nei_list,*num_nei=list->num_neigh;
    int *local_list=list->local_list;
    int ti,tj,inum,*ilist,ii,jj;
    int thread_id=list->thread_id;
    int *ispecial, **special=list->special,spij;
    numtype rsq, dx, dy, dz, _xi,_yi,_zi;
    numtype **icfw,**icfd,**icfr,*ijcfw,*ijcfd,*ijcfr;
    numtype r,*idesc,*jdesc,desck,cff;

    for(int i=0;i<n_local;++i)
    {
        ii=local_list[i];
        xi=ENVIRON::x[ii];
        _xi=xi[0];_yi=xi[1];_zi=xi[2];

        ilist=nlist[i];
        inum=num_nei[i];
        ispecial=special[i];

        ti=ENVIRON::typ[ii];
        icfw=coef_w[ti]; icfd=coef_d[ti]; icfr=coef_r[ti]; 

        idesc=descriptor[ENVIRON::atom_id[ii]];

        for(int j=0; j<inum; ++j)
        {
            jj=ilist[j];
            xj=ENVIRON::x[jj];
            rsq=sqr(dx=_xi-xj[0])+sqr(dy=_yi-xj[1])+sqr(dz=_zi-xj[2]);
            if(rsq<cutsq)
            {
                r=std::sqrt(rsq);
                tj=ENVIRON::typ[jj];
                ijcfw=icfw[tj]+(spij=n_descriptor*ispecial[j]); 
                ijcfd=icfd[tj]+spij; 
                ijcfr=icfr[tj]+spij;

                cff=1-1/(1+sqr(r-cutoff));

                if(ENVIRON::x_thr[jj]==thread_id)
                {
                    jdesc=descriptor[ENVIRON::atom_id[jj]];
                    for(int k=0;k<n_descriptor;++k)
                    {
                        
                        desck=ijcfw[k]*std::exp(ijcfd[k]*sqr(r-ijcfr[k])) * cff;
                        idesc[k]+=desck;
                        jdesc[k]+=desck;
                    }
                }
                else
                {
                    for(int k=0;k<n_descriptor;++k)
                        idesc[k]+=ijcfw[k]*std::exp(ijcfd[k]*sqr(r-ijcfr[k])) * cff;
                }

            }
        }        
    }
}

void NNPOTENTIAL::nnEval()
{
     // modify for multi mols
    tfcall->evaluate();

}

void NNPOTENTIAL::evalFinalize(LOCAL *local)
{
    int iend=local->i_local_end;
    int i=local->i_local_start;
    dedg[i]=tfcall->outputs[1]+n_descriptor*i;
    for(++i;i<iend;++i)
    {
        dedg[i]=dedg[i-1]+n_descriptor; // modify for multi mols
    }
}

template void NNPOTENTIAL::compute<false>(NEIGHLIST *list, numtype *erg, numtype *viral);
template void NNPOTENTIAL::compute<true>(NEIGHLIST *list, numtype *erg, numtype *viral);
template <bool ev> 
void NNPOTENTIAL::compute(NEIGHLIST *list, numtype *erg, numtype *viral)
{
    numtype *xi,*xj;
    numtype *fi,*fj;
    int n_local=list->n_local, **nlist=list->nei_list,*num_nei=list->num_neigh;
    int *local_list=list->local_list;
    int ti,tj,inum,*ilist,ii,jj;
    int thread_id=list->thread_id;
    int *ispecial, **special=list->special,spij;
    numtype rsq, dx, dy, dz, _xi,_yi,_zi, fpair, fx, fy, fz;
    numtype **icfw,**icfd,**icfr,*ijcfw,*ijcfd,*ijcfr;
    numtype **icfp,*ijcfp,**ifvc,*ijfvc;
    numtype r,*idesc,*jdesc,rr0,drr0,cf2,cfr,rrcsq_1;
    numtype sr2,sr6;

    for(int i=0;i<n_local;++i)
    {
        ii=local_list[i];
        xi=ENVIRON::x[ii];
        _xi=xi[0];_yi=xi[1];_zi=xi[2];

        ilist=nlist[i];
        inum=num_nei[i];
        ispecial=special[i];

        ti=ENVIRON::typ[ii];
        icfw=coef_w[ti]; icfd=coef_d[ti]; icfr=coef_r[ti]; 
        icfp=coef_p[ti]; ifvc=vfc[ti];

        idesc=dedg[ENVIRON::atom_id[ii]];

        fx=fy=fz=0;

        for(int j=0; j<inum; ++j)
        {
            jj=ilist[j];
            xj=ENVIRON::x[jj];
            rsq=sqr(dx=_xi-xj[0])+sqr(dy=_yi-xj[1])+sqr(dz=_zi-xj[2]);
            if(rsq<cutsq)
            {
                tj=ENVIRON::typ[jj];
                ijcfw=icfw[tj]+(spij=n_descriptor*ispecial[j]); 
                ijcfd=icfd[tj]+spij; 
                ijcfr=icfr[tj]+spij;
                ijcfp=icfp[tj]+6*ispecial[j];
                ijfvc=ifvc[tj]+2*ispecial[j];

                r=std::sqrt(rsq);
                jdesc=dedg[ENVIRON::atom_id[jj]];

                rrcsq_1=1+sqr(cf2=r-cutoff);
                cfr=(2-2/rrcsq_1)/r;
                cf2/=(.5f*r*sqr(rrcsq_1));

                sr6=cub(sr2=ijcfp[1]/rsq);
                fpair=(
                    ijcfp[5]/r + 
                    (*ijcfp)*sr6*(-6+sr2*(ijcfp[2]+sr2*(ijcfp[3]+sr2*ijcfp[4])))
                    )/rsq - (*ijfvc)/r;
                for(int k=0;k<n_descriptor;++k)
                {
                    rr0=r-ijcfr[k];
                    drr0=ijcfd[k]*rr0;
                    fpair-=(idesc[k]+jdesc[k])*std::exp(drr0*rr0)*(cf2+drr0*cfr)*ijcfw[k];
                }

                fx+=(dx*=fpair);fy+=(dy*=fpair);fz+=(dz*=fpair);

                if(ENVIRON::x_thr[jj]==thread_id)
                {
                    fj=ENVIRON::f[jj];
                    fj[0]-=dx;fj[1]-=dy;fj[2]-=dz;

                    if(ev) 
                    {
                        *erg+= ijcfp[5]/r +
                            (*ijcfp)*sr6*(-1+sr2*(ijcfp[2]*.125F+sr2*(ijcfp[3]*.1F+sr2*ijcfp[4]*.08333333333333333333F)))
                            + (*ijfvc) * r + ijfvc[1];
                        *viral+=fpair*rsq;
                    }
                }
                else
                {
                    if(ev) 
                    {
                        *erg+=.5F*( ijcfp[5]/r +
                            (*ijcfp)*sr6*(-1+sr2*(ijcfp[2]*.125F+sr2*(ijcfp[3]*.1F+sr2*ijcfp[4]*.08333333333333333333F)))
                            + (*ijfvc) * r + ijfvc[1] );
                        *viral+=.5F * fpair*rsq;
                    }
                }
            }
        }      

        fi=ENVIRON::f[ii];
        fi[0]+=fx;fi[1]+=fy;fi[2]+=fz; 
    }
}

NNPOTENTIAL_TBL::NNPOTENTIAL_TBL(numtype cf,int nbr, const char* paramfile,const char* wtfile, BOND *bnd, ANGLE* agl, DIHEDRAL *dih)
{
#ifndef FLOAT_PRECESION
    END_PROGRAM("NNP must use FLOAT_PRECESION");
#endif
    if(ENVIRON::nmoltype != 1) END_PROGRAM("NNP does not support multiple molecules");

    nbin_r=nbr;
    int npair=ENVIRON::npair;
    cutsq=sqr(cutoff=cf);
    int szd=n_descriptor*4;

    tfcall=new TFCALL[ENVIRON::nmoltype];
    char path_nn[SZ_FBF],dt; 
    char **names_in,**names_out;
    constexpr int n_input=1, n_output=2;
    TF_DataType dtype[n_input];
    int index_in[n_input], index_out[n_output];
    create2DArray(names_in,n_input,SZ_FBF);
    create2DArray(names_out,n_output,SZ_FBF);

    int64_t *dimi,**dim;
    create2DArray(dim,n_input,4);

    FILE *fp=fopen(paramfile,"r");
    if(fscanf(fp,"%s ",path_nn)!=1) END_PROGRAM("read error");
    for(int i=0;i<n_input;++i)
    {   
        dimi=dim[i];
        if(fscanf(fp,"%s %c %d %" SCNd64 " %" SCNd64 " %" SCNd64 " %" SCNd64 " ",names_in[i],&dt,index_in+i,dimi,dimi+1,dimi+2,dimi+3)!=7) END_PROGRAM("read error");
        switch (dt)
        {
        case 'f':
        case 'F':
            dtype[i]=TF_FLOAT;
            break;
        case 'i':
        case 'I':
            dtype[i]=TF_INT32;
            break;
        default:
            dtype[i]=TF_FLOAT;
            END_PROGRAM("unknown dtype");
            break;
        }
        if(*dimi==-2) *dimi=ENVIRON::nmol;
        // std::cerr<<i<<" "<<names_in[i]<<" "<<dt<<"\n";
    }

    for(int i=0;i<n_output;++i) if(fscanf(fp,"%s %d ",names_out[i],index_out+i)!=2) END_PROGRAM("read error");
    tfcall->init(path_nn,n_input,names_in,dim,dtype,index_in,n_output,names_out,index_out);
    fclose(fp);

    destroy2DArray(names_in);
    destroy2DArray(names_out);
    destroy2DArray(dim);
    
    descriptor=new numtype*[ENVIRON::natom];
    dedg=new numtype*[ENVIRON::natom];
    descriptor[0]=(numtype*)tfcall->inputs[0];
    for(int i=1;i<ENVIRON::natom;++i)
    {
        descriptor[i]=descriptor[i-1]+n_descriptor;
    }

    numtype **coef_w,**coef_d,**coef_r,**coef_p,**vfc;
    create2DArray(coef_w,npair,szd);
    create2DArray(coef_d,npair,szd);
    create2DArray(coef_r,npair,szd);
    create2DArray(coef_p,npair,6*4);
    create2DArray(vfc,   npair,2*4);

    fp=fopen(wtfile,"rb");

    fread(*(bnd->coef),sizeof(numtype),bnd->nbtyp*4,fp);
    fread(*(agl->coef),sizeof(numtype),agl->natyp*4,fp);
    fread(*(dih->coef),sizeof(numtype),dih->ndtyp*4,fp);
    bnd->setParam();
    agl->setParam();
    dih->setParam();

    fread(*coef_p,sizeof(numtype),npair*6*4,fp);
    fread(*coef_w,sizeof(numtype),npair*szd,fp);
    fread(*coef_d,sizeof(numtype),npair*szd,fp);
    fread(*coef_r,sizeof(numtype),npair*szd,fp);

    fclose(fp);

    // generate table from params

    nbr+=1;
    create3DArray(tbl_desc,npair,szd,nbr);
    create3DArray(tbl_f_desc,npair,szd,nbr);
    tbl_pair=create3DArray(tbl_pair_base,npair,4,nbr);
    tbl_f_pair=create3DArray(tbl_f_pair_base,npair,4,nbr);

    numtype src2,src6,elim,flim;
    numtype *p_cfp,*p_cfw,*p_cfd,*p_cfr;
    numtype **p_des,**p_fes,**p_pir,**p_fir;
    numtype *pk_cfp;
    numtype *pk_pir,*pk_fir;
    dr=cutoff/nbin_r;
    r_dr=nbin_r/cutoff;
    numtype rr,rsq;
    numtype cff,cfr,rrcsq_1,cf2,rr0,drr0,wexp;
    for(int p=0;p<npair;++p)
    {
        p_cfp=coef_p[p];
        p_pir=tbl_pair_base[p]; p_fir=tbl_f_pair_base[p];
        for(int k=0;k<4;++k)
        {
            pk_cfp=p_cfp+k*6;
            src2=sqr(pk_cfp[1]/cutoff);
            src6=cub(src2);
            elim=-2*pk_cfp[5]/cutoff-pk_cfp[0]*src6*(-7+src2*(9*pk_cfp[2]+src2*(11*pk_cfp[3]+src2*13*pk_cfp[4]))); 
            flim=(pk_cfp[5]/cutoff+pk_cfp[0]*src6*(-6+src2*((pk_cfp[2]*=8)+src2*((pk_cfp[3]*=10)+src2*(pk_cfp[4]*=12)))))/cutoff;
            pk_cfp[1]*=pk_cfp[1];

            pk_pir=p_pir[k];
            pk_fir=p_fir[k];
            for(int i=0;i<nbin_r;++i)
            {
                rsq=sqr(rr=dr*(i+0.5));
                src6=cub(src2=pk_cfp[1]/rsq);
                pk_fir[i]=(
                    pk_cfp[5]/rr + 
                    (*pk_cfp)*src6*(-6+src2*(pk_cfp[2]+src2*(pk_cfp[3]+src2*pk_cfp[4])))
                    )/rsq - (flim)/rr;
                pk_pir[i]=pk_cfp[5]/rr +
                            (*pk_cfp)*src6*(-1+src2*(pk_cfp[2]*.125F+src2*(pk_cfp[3]*.1F+src2*pk_cfp[4]*.08333333333333333333F)))
                            + (flim) * rr + elim;
            }
            pk_pir[nbin_r]=pk_fir[nbin_r]=0;
        }
        
        p_cfw=coef_w[p]; p_cfd=coef_d[p]; p_cfr=coef_r[p];
        p_des=tbl_desc[p]; p_fes=tbl_f_desc[p];

        for(int i=0;i<nbin_r;++i)
        {
            rr=dr*(i+0.5);
            rrcsq_1=1+sqr(cf2=rr-cutoff);
            cfr=(-2*(cff=1-1/rrcsq_1))/rr;
            cf2/=(-.5f*rr*sqr(rrcsq_1));
            for(int k=0;k<szd;++k)
            {

                rr0=rr-p_cfr[k];
                drr0=p_cfd[k]*rr0;
                p_fes[k][i]=(wexp=std::exp(drr0*rr0)*p_cfw[k])*(cf2+drr0*cfr);
                p_des[k][i]=wexp * cff;
            }
        }
        for(int k=0;k<szd;++k)
        {
            p_des[k][nbin_r]=p_fes[k][nbin_r]=0;
        }
    }

    destroy2DArray(coef_w);
    destroy2DArray(coef_d);
    destroy2DArray(coef_r);
    destroy2DArray(coef_p);
    destroy2DArray(vfc   );

}

void NNPOTENTIAL_TBL::allocateTable()
{
    create3DArray(tbl_pair,ENVIRON::npair,4,nbin_r+1);
    create3DArray(tbl_f_pair,ENVIRON::npair,4,nbin_r+1);
    auto sz=ENVIRON::npair*4*(nbin_r+1)*sizeof(numtype);
    memcpy(tbl_pair[0][0],tbl_pair_base[0][0],sz);
    memcpy(tbl_f_pair[0][0],tbl_f_pair_base[0][0],sz);
}

NNPOTENTIAL_TBL::~NNPOTENTIAL_TBL()
{
    destroy3DArray(tbl_desc);
    destroy3DArray(tbl_f_desc);
    if(tbl_pair_base!=tbl_pair)
    {
        destroy3DArray(tbl_pair);
        destroy3DArray(tbl_f_pair);
    }
    destroy3DArray(tbl_pair_base);
    destroy3DArray(tbl_f_pair_base);
    delete[] tfcall;
    delete[] descriptor;
    delete[] dedg;
}

void NNPOTENTIAL_TBL::computeDescriptor(NEIGHLIST *list)
{
    numtype *xi,*xj;
    int n_local=list->n_local, **nlist=list->nei_list,*num_nei=list->num_neigh;
    int *local_list=list->local_list;
    int inum,*ilist,ii,jj;
    int thread_id=list->thread_id;
    int *ispecial, **special=list->special,spij;
    numtype rsq, dx, dy, dz, _xi,_yi,_zi;
    numtype r,*idesc,*jdesc,desck;
    int *ipair,tbr1,tbr2;
    numtype ipw1,ipw2,**descij,*descijk;

    for(int i=0;i<n_local;++i)
    {
        ii=local_list[i];
        xi=ENVIRON::x[ii];
        _xi=xi[0];_yi=xi[1];_zi=xi[2];

        ilist=nlist[i];
        inum=num_nei[i];
        ispecial=special[i];

        ipair=ENVIRON::pairtype[ENVIRON::typ[ii]];
        idesc=descriptor[ENVIRON::atom_id[ii]];

        for(int j=0; j<inum; ++j)
        {
            jj=ilist[j];
            xj=ENVIRON::x[jj];
            rsq=sqr(dx=_xi-xj[0])+sqr(dy=_yi-xj[1])+sqr(dz=_zi-xj[2]);
            if(rsq<cutsq)
            {
                r=std::sqrt(rsq);
                ipw1=r*r_dr-0.5F;
                tbr2=1+(tbr1=(int)ipw1);
                ipw1=1-(ipw2=ipw1-tbr1);
                descij=tbl_desc[ipair[ENVIRON::typ[jj]]];
                spij=n_descriptor*ispecial[j];

                if(ENVIRON::x_thr[jj]==thread_id)
                {
                    jdesc=descriptor[ENVIRON::atom_id[jj]];
                    for(int k=0;k<n_descriptor;++k)
                    {
                        descijk=descij[spij+k];
                        desck=descijk[tbr1]*ipw1 + descijk[tbr2]*ipw2;
                        idesc[k]+=desck;
                        jdesc[k]+=desck;
                    }
                }
                else
                {
                    for(int k=0;k<n_descriptor;++k)
                    {
                        descijk=descij[spij+k];
                        idesc[k]+=descijk[tbr1]*ipw1 + descijk[tbr2]*ipw2;
                    }
                }

            }
        }        
    }
}

void NNPOTENTIAL_TBL::nnEval()
{
     // modify for multi mols
    tfcall->evaluate();

}

void NNPOTENTIAL_TBL::evalFinalize(LOCAL *local)
{
    int iend=local->i_local_end;
    int i=local->i_local_start;
    dedg[i]=tfcall->outputs[1]+n_descriptor*i;
    for(++i;i<iend;++i)
    {
        dedg[i]=dedg[i-1]+n_descriptor; // modify for multi mols
    }
}

template void NNPOTENTIAL_TBL::compute<false>(NEIGHLIST *list, numtype *erg, numtype *viral);
template void NNPOTENTIAL_TBL::compute<true>(NEIGHLIST *list, numtype *erg, numtype *viral);
template <bool ev> 
void NNPOTENTIAL_TBL::compute(NEIGHLIST *list, numtype *erg, numtype *viral)
{
    numtype *xi,*xj;
    numtype *fi,*fj;
    int n_local=list->n_local, **nlist=list->nei_list,*num_nei=list->num_neigh;
    int *local_list=list->local_list;
    int inum,*ilist,ii,jj;
    int thread_id=list->thread_id;
    int *ispecial, **special=list->special,spij;
    numtype rsq, dx, dy, dz, _xi,_yi,_zi, fpair, fx, fy, fz;
    numtype r,*idesc,*jdesc;
    int *ipair,tbr1,tbr2,ipr;
    numtype ipw1,ipw2,**descij,*descijk;

    for(int i=0;i<n_local;++i)
    {
        ii=local_list[i];
        xi=ENVIRON::x[ii];
        _xi=xi[0];_yi=xi[1];_zi=xi[2];

        ilist=nlist[i];
        inum=num_nei[i];
        ispecial=special[i];

        ipair=ENVIRON::pairtype[ENVIRON::typ[ii]];
        idesc=dedg[ENVIRON::atom_id[ii]];

        fx=fy=fz=0;

        for(int j=0; j<inum; ++j)
        {
            jj=ilist[j];
            xj=ENVIRON::x[jj];
            rsq=sqr(dx=_xi-xj[0])+sqr(dy=_yi-xj[1])+sqr(dz=_zi-xj[2]);
            if(rsq<cutsq)
            {
                r=std::sqrt(rsq);
                ipw1=r*r_dr-0.5F;
                tbr2=1+(tbr1=(int)ipw1);
                ipw1=1-(ipw2=ipw1-tbr1);
                descij=tbl_f_desc[ipr=ipair[ENVIRON::typ[jj]]];

                descijk=tbl_f_pair[ipr][ispecial[j]];
                fpair=descijk[tbr1]*ipw1 + descijk[tbr2]*ipw2;
                
                spij=n_descriptor*ispecial[j];
                jdesc=dedg[ENVIRON::atom_id[jj]];
                for(int k=0;k<n_descriptor;++k)
                {
                    descijk=descij[spij+k];
                    fpair+=(idesc[k]+jdesc[k])*(descijk[tbr1]*ipw1 + descijk[tbr2]*ipw2);
                }

                fx+=(dx*=fpair);fy+=(dy*=fpair);fz+=(dz*=fpair);

                if(ENVIRON::x_thr[jj]==thread_id)
                {
                    fj=ENVIRON::f[jj];
                    fj[0]-=dx;fj[1]-=dy;fj[2]-=dz;

                    if(ev) 
                    {
                        descijk=tbl_pair[ipr][ispecial[j]];
                        *erg+= descijk[tbr1]*ipw1 + descijk[tbr2]*ipw2;
                        *viral+=fpair*rsq;
                    }
                }
                else
                {
                    if(ev) 
                    {
                        descijk=tbl_pair[ipr][ispecial[j]];
                        *erg+= 0.5F*(descijk[tbr1]*ipw1 + descijk[tbr2]*ipw2);
                        *viral+=.5F * fpair*rsq;
                    }
                }
            }
        }      

        fi=ENVIRON::f[ii];
        fi[0]+=fx;fi[1]+=fy;fi[2]+=fz; 
    }
}

/*

NNP_TRAIN::NNP_TRAIN(numtype cf,const char* paramfile)
{
    if(ENVIRON::nmoltype != 1) END_PROGRAM("NNP does not support multiple molecules");

    ntyp=ENVIRON::ntype;
    cutsq=sqr(cutoff=cf);
    int _4nd=NNPOTENTIAL::n_descriptor*4;
    create3DArray(coef_w,ntyp,ntyp,_4nd);
    create3DArray(coef_d,ntyp,ntyp,_4nd);
    create3DArray(coef_r,ntyp,ntyp,_4nd);
    create3DArray(grd_w,ntyp,ntyp,_4nd);
    create3DArray(grd_d,ntyp,ntyp,_4nd);
    create3DArray(grd_r,ntyp,ntyp,_4nd);
    tfcall=new TFCALL[ENVIRON::nmoltype];

    char path_nn[SZ_FBF]; 
    char **names_in,**names_out;
    create2DArray(names_in,11,SZ_FBF);
    create2DArray(names_out,10,SZ_FBF);
    int64_t **dim;
    create2DArray(dim,11,4);

    FILE *fp=fopen(paramfile,"r");
    if(fscanf(fp,"%s ",path_nn)!=1) END_PROGRAM("read error");
    for(int i=0;i<11;++i) if(fscanf(fp,"%s %" SCNd64 " %" SCNd64 " %" SCNd64 " %" SCNd64 " ",names_in[i],dim[i],dim[i]+1,dim[i]+2,dim[i]+3)!=5) END_PROGRAM("read error");
    for(int i=0;i<10;++i) if(fscanf(fp,"%s ",names_out[i])!=1) END_PROGRAM("read error");
    tfcall->init(path_nn,11,names_in,dim,10,names_out);
    fclose(fp);

    destroy2DArray(names_in);
    destroy2DArray(names_out);
    destroy2DArray(dim);
    
    create3DArray<numtype*>(D_ijk,ENVIRON::natom,3,ENVIRON::natom);
    numtype **ptr=D_ijk[0][0];
    int ptt=ENVIRON::natom*3*ENVIRON::natom;
    ptr[0]=tfcall->inputs[8];
    for(int i=1;i<ptt;++i) ptr[i]=ptr[i-1]+NNPOTENTIAL::n_descriptor;

    // create2DArray(descriptor,ENVIRON::natom,NNPOTENTIAL::n_descriptor);

    descriptor=new numtype*[ENVIRON::natom];
    dedg=new numtype*[ENVIRON::natom];
    descriptor[0]=tfcall->inputs[0];
    for(int i=1;i<ENVIRON::natom;++i)
    {
        descriptor[i]=descriptor[i-1]+NNPOTENTIAL::n_descriptor;; // modify for multi mols
    }
    
    fp=fopen("test/nnp_desc.bin","rb");
    for(int i=0;i<ntyp;++i) fread(coef_w[i][i],sizeof(numtype),(ntyp-i)*_4nd,fp);
    for(int i=0;i<ntyp;++i) fread(coef_d[i][i],sizeof(numtype),(ntyp-i)*_4nd,fp);
    for(int i=0;i<ntyp;++i) fread(coef_r[i][i],sizeof(numtype),(ntyp-i)*_4nd,fp);
    fclose(fp);

    for(int i=0;i<ntyp;++i)
        for(int j=i+1;j<ntyp;++j)
            for(int k=0;k<_4nd;++k)
            {
                coef_w[j][i][k]=coef_w[i][j][k];
                coef_d[j][i][k]=coef_d[i][j][k];
                coef_r[j][i][k]=coef_r[i][j][k];
            }
    
    // load weight
    fp=fopen("ann/weights.bin","rb");
    fread(tfcall->inputs[1],sizeof(numtype),448,fp);
    fread(tfcall->inputs[2],sizeof(numtype),224,fp);
    fread(tfcall->inputs[3],sizeof(numtype),2352,fp);
    fread(tfcall->inputs[4],sizeof(numtype),56,fp);
    fread(tfcall->inputs[5],sizeof(numtype),896,fp);
    fread(tfcall->inputs[6],sizeof(numtype),16,fp);
    fread(tfcall->inputs[7],sizeof(numtype),16,fp);
    fclose(fp);
}

void NNP_TRAIN::computeDescriptor(NEIGHLIST *list)
{
    numtype *xi,*xj;
    int n_local=list->n_local, **nlist=list->nei_list,*num_nei=list->num_neigh;
    int *local_list=list->local_list;
    int ti,tj,inum,*ilist,ii,jj;
    int thread_id=list->thread_id;
    int *ispecial, **special=list->special,spij,idi,idj;
    numtype rsq, dx, dy, dz, _xi,_yi,_zi;
    numtype **icfw,**icfd,**icfr,*ijcfw,*ijcfd,*ijcfr;
    numtype r,*idesc,*jdesc,desck,cff,cfr,rr0,rrcsq_1,cf2,wexp,drr0,fx,fy,fz;
    numtype **Dix,*Dixj,**Diy,*Diyj,**Diz,*Dizj;
    numtype *Dixi,*Diyi,*Dizi,*Djxi,*Djyi,*Djzi,*Djxj,*Djyj,*Djzj,**Djx,**Djy,**Djz;

    for(int i=0;i<n_local;++i)
    {
        ii=local_list[i];
        xi=ENVIRON::x[ii];
        _xi=xi[0];_yi=xi[1];_zi=xi[2];

        ilist=nlist[i];
        inum=num_nei[i];
        ispecial=special[i];

        ti=ENVIRON::typ[ii];
        icfw=coef_w[ti]; icfd=coef_d[ti]; icfr=coef_r[ti]; 

        idesc=descriptor[idi=ENVIRON::atom_id[ii]];
        Dix=D_ijk[idi][0]; Diy=D_ijk[idi][1]; Diz=D_ijk[idi][2];
        Dixi=Dix[idi]; Diyi=Diy[idi]; Dizi=Diz[idi]; 

        for(int j=0; j<inum; ++j)
        {
            jj=ilist[j];
            xj=ENVIRON::x[jj];
            rsq=sqr(dx=xj[0]-_xi)+sqr(dy=xj[1]-_yi)+sqr(dz=xj[2]-_zi);
            if(rsq<cutsq)
            {
                r=std::sqrt(rsq);
                tj=ENVIRON::typ[jj];
                ijcfw=icfw[tj]+(spij=n_descriptor*ispecial[j]); 
                ijcfd=icfd[tj]+spij; 
                ijcfr=icfr[tj]+spij;

                rrcsq_1=1+sqr(cf2=r-cutoff);
                cfr=2*(cff=1-1/rrcsq_1)/r;
                cf2/=(.5f*r*sqr(rrcsq_1));

                Dixj=Dix[idj=ENVIRON::atom_id[jj]]; Diyj=Diy[idj]; Dizj=Diz[idj]; 

                // if(ENVIRON::x_thr[jj]==thread_id)
                // {
                //     jdesc=descriptor[idj];
                //     Djx=D_ijk[idj][0]; Djy=D_ijk[idj][1]; Djz=D_ijk[idj][2]; 
                //     Djxi=Djx[idi]; Djyi=Djy[idi]; Djzi=Djz[idi]; 
                //     Djxj=Djx[idj]; Djyj=Djy[idj]; Djzj=Djz[idj]; 

                //     for(int k=0;k<NNPOTENTIAL::n_descriptor;++k)
                //     {
                //         rr0=r-ijcfr[k];
                //         drr0=ijcfd[k]*rr0;
                //         wexp=ijcfw[k]*std::exp(drr0*rr0);
                //         desck=wexp * cff;
                //         idesc[k]+=desck;
                //         jdesc[k]+=desck;
                //         desck=wexp * (cf2+drr0*cfr);
                //         Djxi[k]= -(Dixj[k]=fx=desck*dx);
                //         Djyi[k]= -(Diyj[k]=fy=desck*dy);
                //         Djzi[k]= -(Dizj[k]=fz=desck*dz);
                //         Dixi[k]+=fx; Diyi[k]+=fy; Dizi[k]+=fz;
                //         Djxj[k]-=fx; Djyj[k]-=fy; Djzj[k]-=fz;
                //         // if(ijcfw[k]) std::cerr<<ii<<" "<<jj<<" "<<k<<" "<<(std::exp(drr0*rr0)*(cf2+drr0*rrcsq_1)*ijcfw[k])<<" "<<desck<<"\n";
                //     }
                // }
                // else
                {
                    for(int k=0;k<NNPOTENTIAL::n_descriptor;++k)
                    {
                        rr0=r-ijcfr[k];
                        drr0=ijcfd[k]*rr0;
                        wexp=ijcfw[k]*std::exp(drr0*rr0);
                        idesc[k]+=wexp * cff;
                        desck=wexp * (cf2+drr0*cfr);
                        Dixi[k] += (Dixj[k]=desck*dx);
                        Diyi[k] += (Diyj[k]=desck*dy);
                        Dizi[k] += (Dizj[k]=desck*dz);
                        if(ijcfw[k]) std::cerr<<ENVIRON::atom_id[ii]<<ENVIRON::atom_id[jj]<<" "<<k<<" "<<(cf2+drr0*cfr)*2<<"\n";
                    }
                }
            }
        }        
    }
}

NNP_TRAIN::~NNP_TRAIN()
{
    // delete[] tfcall;
    // destroy3DArray(coef_w);
    // destroy3DArray(coef_d);
    // destroy3DArray(coef_r);
    // destroy3DArray(D_i_jk_3);
    // delete[] descriptor;
}

void NNP_TRAIN::nnEval()
{
    tfcall->evaluate(); 
    dedg[0]=tfcall->outputs[2];
    for(int i=1;i<ENVIRON::natom;++i)
    {
        dedg[i]=dedg[i-1]+NNPOTENTIAL::n_descriptor;; // modify for multi mols
    }
    int sz=ntyp*ntyp*NNPOTENTIAL::n_descriptor*4;
    numtype *gw=**grd_w, *gd=**grd_d, *gr=**grd_r;
    for(int i=0;i<sz;++i)
    {
        gw[i]=gd[i]=gr[i]=0;
    }
    numtype *fnnp=tfcall->outputs[1];
    for(int i=0;i<ENVIRON::natom;++i)
    {
        ENVIRON::f_mol[i][0]=*(fnnp++);
        ENVIRON::f_mol[i][1]=*(fnnp++);
        ENVIRON::f_mol[i][2]=*(fnnp++);
    }
}

void NNP_TRAIN::compute(NEIGHLIST *list,numtype d_e)
{
    numtype *xi,*xj;
    numtype *fi,*fj;
    int n_local=list->n_local, **nlist=list->nei_list,*num_nei=list->num_neigh;
    int *local_list=list->local_list;
    int ti,tj,inum,*ilist,ii,jj;
    int thread_id=list->thread_id;
    int *ispecial, **special=list->special,spij;
    numtype rsq, dx, dy, dz, _xi,_yi,_zi;
    numtype **icfw,**icfd,**icfr,*ijcfw,*ijcfd,*ijcfr;
    numtype **d_icfw,**d_icfd,**d_icfr,*d_ijcfw,*d_ijcfd,*d_ijcfr;
    numtype r,*idesc,*jdesc,cff,rr0,drr0,drr0sq,_exp,wexp,rrcsq_1,cf2,cfr,dot,dedgij;

    for(int i=0;i<n_local;++i)
    {
        ii=local_list[i];
        xi=ENVIRON::x[ii];
        fi=ENVIRON::f[ii];
        _xi=xi[0];_yi=xi[1];_zi=xi[2];

        ilist=nlist[i];
        inum=num_nei[i];
        ispecial=special[i];

        ti=ENVIRON::typ[ii];
        icfw=coef_w[ti]; icfd=coef_d[ti]; icfr=coef_r[ti]; 
        d_icfw=grd_w[ti]; d_icfd=grd_d[ti]; d_icfr=grd_r[ti]; 

        idesc=dedg[ENVIRON::atom_id[ii]];

        for(int j=0; j<inum; ++j)
        {
            jj=ilist[j];
            xj=ENVIRON::x[jj];
            rsq=sqr(dx=_xi-xj[0])+sqr(dy=_yi-xj[1])+sqr(dz=_zi-xj[2]);
            if(rsq<cutsq)
            {
                r=std::sqrt(rsq);
                tj=ENVIRON::typ[jj];
                ijcfw=icfw[tj]+(spij=n_descriptor*ispecial[j]); 
                ijcfd=icfd[tj]+spij; 
                ijcfr=icfr[tj]+spij;
                d_ijcfw=d_icfw[tj]+spij; 
                d_ijcfd=d_icfd[tj]+spij; 
                d_ijcfr=d_icfr[tj]+spij;

                rrcsq_1=1+sqr(cf2=r-cutoff);
                cfr=2*(cff=2-2/rrcsq_1)/r;
                cf2/=(.25f*r*sqr(rrcsq_1));
                jdesc=dedg[ENVIRON::atom_id[jj]];

                // if(ENVIRON::x_thr[jj]==thread_id)
                // {
                //     fj=ENVIRON::f[jj];
                //     dot=(fi[0]-fj[0])*dx+(fi[1]-fj[1])*dy+(fi[2]-fj[2])*dz;
                // }
                // else
                    dot=fi[0]*dx+fi[1]*dy+fi[2]*dz;
                for(int k=0;k<NNPOTENTIAL::n_descriptor;++k)
                {
                    rr0=r-ijcfr[k];
                    drr0sq=(drr0=ijcfd[k]*rr0)*rr0;
                    wexp=ijcfw[k]*(_exp=std::exp(drr0sq));
                    dedgij=idesc[k]+jdesc[k];
                    d_ijcfw[k]+=_exp*((cf2+drr0*cfr)*dot - cff*d_e)*dedgij;
                    d_ijcfd[k]+=wexp*(rr0*(rr0*cf2+(1+drr0sq)*cfr)*dot - cff*sqr(rr0)*d_e)*dedgij;
                    d_ijcfr[k]+=2*wexp*(cff*drr0*d_e - ijcfd[k]*(rr0*cf2+(.5F+drr0sq)*cfr)*dot)*dedgij;
                    if(ijcfw[k])
                    {
                        std::cerr<<ENVIRON::atom_id[ii]<<ENVIRON::atom_id[jj]<<" "<<dot<<" "<<(cf2+drr0*cfr)<<std::endl;
                    }
                }
            }
        }        
    }
}

void NNP_TRAIN::reduce_grad()
{
    constexpr int _4nd=4*NNPOTENTIAL::n_descriptor;
    for(int i=0;i<ntyp;++i)
        for(int j=i+1;j<ntyp;++j)
            for(int k=0;k<_4nd;++k)
            {
                grd_w[j][i][k]=grd_w[i][j][k]=(grd_w[j][i][k]+grd_w[i][j][k]);
                grd_d[j][i][k]=grd_d[i][j][k]=(grd_d[j][i][k]+grd_d[i][j][k]);
                grd_r[j][i][k]=grd_r[i][j][k]=(grd_r[j][i][k]+grd_r[i][j][k]);
            }
}

*/

NNP_TRAIN_TF::NNP_TRAIN_TF(const char* paramfile, const char* weightfile,const char* optzfile,numtype cf, BOND *bnd, ANGLE* agl, DIHEDRAL *dih)
{
#ifndef FLOAT_PRECESION
    END_PROGRAM("NNP must use FLOAT_PRECESION");
#endif
    if(ENVIRON::nmoltype != 1) END_PROGRAM("NNP does not support multiple molecules");
#ifndef NNP_USE_X_MOL
    create1DArray(inv_atom_id,ENVIRON::natom);
#endif

#ifdef NNP_TRAIN_TF_NORMALIZE_CHARGE
    create1DArray(num_pair,npair);
    numtype natsq=sqr(ENVIRON::natom);
    int ipr=0;
    for(int i=0;i<ENVIRON::ntype;++i)
        for(int j=i;j<ENVIRON::ntype;++j)
        {
            num_pair[ipr++]=ENVIRON::typecount[i]*ENVIRON::typecount[j]*(i==j? 1 : 2)/natsq;
        }
#endif

    tfcall=new TFCALL[ENVIRON::nmoltype];
    char path_nn[SZ_FBF],dt; 
    char **names_in,**names_out;
    TF_DataType dtype[n_input];
    int index_in[n_input], index_out[n_output];
    create2DArray(names_in,n_input,SZ_FBF);
    create2DArray(names_out,n_output,SZ_FBF);
    create1DArray(sz_in,n_input);

    int64_t *dimi;
    int ndims;
    create2DArray(dim,n_input,4);
    create1DArray(ndim,n_input);

    FILE *fp=fopen(paramfile,"r");
    if(fscanf(fp,"%s ",path_nn)!=1) END_PROGRAM("read error");
    for(int i=0;i<n_input;++i)
    {   
        dimi=dim[i];
        if(fscanf(fp,"%s %c %d %" SCNd64 " %" SCNd64 " %" SCNd64 " %" SCNd64 " ",names_in[i],&dt,index_in+i,dimi,dimi+1,dimi+2,dimi+3)!=7) END_PROGRAM("read error");
        switch (dt)
        {
        case 'f':
        case 'F':
            dtype[i]=TF_FLOAT;
            break;
        case 'i':
        case 'I':
            dtype[i]=TF_INT32;
            break;
        default:
            dtype[i]=TF_FLOAT;
            END_PROGRAM("unknown dtype");
            break;
        }
        sz_in[i]=1;
        for(ndims=0;dimi[ndims];++ndims)
        {
            if(dimi[ndims]==-2) dimi[ndims]=ENVIRON::natom;
            sz_in[i]*=(int)dimi[ndims];
        }
        ndim[i]=ndims;
        // std::cerr<<i<<" "<<names_in[i]<<" "<<dt<<" "<<sz_in[i]<<"\n";
    }
    for(int i=0;i<n_input;++i) sz_in[i]=std::abs(sz_in[i]);

    for(int i=0;i<n_output;++i) if(fscanf(fp,"%s %d ",names_out[i],index_out+i)!=2) END_PROGRAM("read error");
    tfcall->init(path_nn,n_input,names_in,dim,dtype,index_in,n_output,names_out,index_out);
    fclose(fp);

    destroy2DArray(names_in);
    destroy2DArray(names_out);
    // destroy2DArray(dim);
    
    fp=fopen(weightfile,"rb");
    int totalsz=0,nwtaccu=0,curw=0;
    create1DArray(iwt_begin,ENVIRON::NUM_THREAD);
    create1DArray(iwt_end,ENVIRON::NUM_THREAD);
    for(int i=0;i<n_wts;++i)
    {
        totalsz+=sz_in[i];
        fread(tfcall->inputs[i],sizeof(numtype),sz_in[i],fp);
    }
    iwt_begin[0]=0;iwt_end[ENVIRON::NUM_THREAD-1]=n_wts;
    // std::cerr<<"sz_tot "<<totalsz<<"\n";
    for(int i=1;i<ENVIRON::NUM_THREAD;++i)
    {
        ndims=i*totalsz/ENVIRON::NUM_THREAD; // target sz begin
        while(nwtaccu<=ndims) nwtaccu+=sz_in[curw++];
        iwt_end[i-1]=iwt_begin[i]=curw;
        // std::cerr<<"thr "<<i<<" "<<iwt_begin[i]<<" "<<nwtaccu<<"\n";
    }
    // for(int i=0;i<ENVIRON::NUM_THREAD;++i)
    // {
    //     std::cerr<<"thr "<<i<<" "<<iwt_begin[i]<<" "<<iwt_end[i]<<"\n";
    // }

    fclose(fp);

    ndims=n_wts+5;
    sz_in[ndims] *= ( dim[ndims][0]=bnd->nbonds );
    tfcall->inputs[ndims] = TF_TensorData( tfcall->InputValues[ndims]=TF_AllocateTensor(TF_INT32,dim[ndims],ndim[ndims],sz_in[ndims]*sizeof(int)) );
    memcpy(tfcall->inputs[ndims],*(bnd->bond),sz_in[ndims]*sizeof(int));

    ++ndims;
    sz_in[ndims] *= ( dim[ndims][0]=agl->nangles );
    tfcall->inputs[ndims] = TF_TensorData( tfcall->InputValues[ndims]=TF_AllocateTensor(TF_INT32,dim[ndims],ndim[ndims],sz_in[ndims]*sizeof(int)) );
    memcpy(tfcall->inputs[ndims],*(agl->angle),sz_in[ndims]*sizeof(int));

    ++ndims;
    sz_in[ndims] *= ( dim[ndims][0]=dih->ndihedrals );
    tfcall->inputs[ndims] = TF_TensorData( tfcall->InputValues[ndims]=TF_AllocateTensor(TF_INT32,dim[ndims],ndim[ndims],sz_in[ndims]*sizeof(int)) );
    memcpy(tfcall->inputs[ndims],*(dih->dihedral),sz_in[ndims]*sizeof(int));

    // std::cerr<<(tfcall->inputs[n_wts+5])<<" "<<bnd->bond[0]<<"\n";
    // std::cerr<<(tfcall->inputs[n_wts+6])<<" "<<agl->angle[0]<<"\n";
    // std::cerr<<(tfcall->inputs[n_wts+7])<<" "<<dih->dihedral[0]<<"\n";


    *(int*)tfcall->inputs[n_wts+8]=ENVIRON::nmol;
    *(numtype*)tfcall->inputs[n_wts+9]=cutoff=cf;
    *(numtype*)tfcall->inputs[n_wts+10]=ENVIRON::bl;
    *(numtype*)tfcall->inputs[n_wts+13]=0;

    // capacity=cap*ENVIRON::natom;
    // create1DArray(i_all,capacity);
    // create1DArray(j_all,capacity);
    // create1DArray(im_all,capacity);
    // create1DArray(tp_all,2*capacity);

    wts=new numtype*[n_wts];
    acc1=new numtype*[n_wts];
    acc2=new numtype*[n_wts];
    for(int i=0;i<n_wts;++i)
    {
        wts[i]=(numtype*)tfcall->inputs[i];
        acc1[i]=new numtype [sz_in[i]];
        acc2[i]=new numtype [sz_in[i]];
    }

    if( (fp=fopen(optzfile,"rb")) )
    {
        fread(&n_update,sizeof(int),1,fp);
        for(int i=0;i<n_wts;++i)
        {
            fread(acc1[i],sizeof(numtype),sz_in[i],fp);
            fread(acc2[i],sizeof(numtype),sz_in[i],fp);
        }
        fclose(fp);
    }
    else 
    {
        numtype *aci1,*aci2;
        n_update=0;
        for(int i=0;i<n_wts;++i)
        {
            aci1=acc1[i];
            aci2=acc2[i];
            ndims=sz_in[i];
            for(int j=0;j<ndims;++j) aci1[j]=aci2[j]=0;
        }
        std::cerr<<"warning: cannot open optzfile\n";
    }
}

NNP_TRAIN_TF::~NNP_TRAIN_TF()
{
#ifndef NNP_USE_X_MOL
    destroy1DArray(inv_atom_id);
#endif
#ifdef NNP_TRAIN_TF_NORMALIZE_CHARGE
    destroy1DArray(num_pair);
#endif
    destroy1DArray(sz_in);
    destroy2DArray(dim);
    destroy1DArray(ndim);
    destroy1DArray(iwt_begin);
    destroy1DArray(iwt_end);
    // destroy1DArray(i_all);
    // destroy1DArray(j_all);
    // destroy1DArray(im_all);
    // destroy1DArray(tp_all);
    delete[] tfcall;
    for(int i=0;i<n_wts;++i)
    {
        delete[] acc1[i];
        delete[] acc2[i];
    }
    delete[]wts ;
    delete[]acc1;
    delete[]acc2;

    destroy2DArray(fnames);
    delete[] fnames_shuffled;
    destroy3DArray(sym_group);
    destroy2DArray<int*>(sym_group_shuffed);
    destroy1DArray(symsz);
    destroy1DArray(grpsz);
    destroy2DArray(x_temp);
    destroy2DArray(f_temp);
}

void NNP_TRAIN_TF::gather_neilist(NEIGHLIST* list)
{
    int n_local=list->n_local, **nlist=list->nei_list,*num_nei=list->num_neigh;
    int *local_list=list->local_list;
    int inum,*ilist,ii,jj,ii_mol,*ipair_typ;
    int thread_id=list->thread_id;
    int *ispecial, **special=list->special;

    int ofst=num_neigh_local[thread_id],
        *i_all_local =(int*)(tfcall->inputs[n_wts+1])+ofst,
        *j_all_local =(int*)(tfcall->inputs[n_wts+2])+ofst,
        *im_all_local=(int*)(tfcall->inputs[n_wts+3])+ofst,
        *tp_all_local=(int*)(tfcall->inputs[n_wts+4])+ofst*2;

    numtype *df=(numtype*)tfcall->inputs[n_wts+11];
    numtype *xs=(numtype*)tfcall->inputs[n_wts];

    for(int i=0;i<n_local;++i)
    {
        ii=local_list[i];
        ii_mol=ENVIRON::atom_id[ii];
#ifdef NNP_USE_X_MOL
        memcpy(df+3*ii_mol,ENVIRON::f_prev[ii],3*sizeof(numtype)); // because f is not sorted, need to use fprev
        memcpy(xs+3*ii_mol,ENVIRON::x[ii],3*sizeof(numtype));
#else
        memcpy(df+3*ii,ENVIRON::f_prev[ii],3*sizeof(numtype));
        memcpy(xs+3*ii,ENVIRON::x[ii],3*sizeof(numtype));
        inv_atom_id[ii_mol]=ii;
#endif
        ilist=nlist[i];
        inum=num_nei[i];
        ispecial=special[i];
        ipair_typ=ENVIRON::pairtype[ENVIRON::typ[ii]];

        for(int j=0; j<inum; ++j)
        {
            jj=ilist[j];
#ifdef NNP_USE_X_MOL
            *(i_all_local++)=ii_mol;
            *(j_all_local++)=ENVIRON::atom_id[jj];
#else
            *(i_all_local++)=ii;
            *(j_all_local++)=jj<ENVIRON::natom? jj : ENVIRON::i_ghost[jj-ENVIRON::natom];
#endif
            *(im_all_local++)=ii_mol;
            *(tp_all_local++)=ipair_typ[ENVIRON::typ[jj]];
            *(tp_all_local++)=ispecial[j];
        }        
    }

}

numtype NNP_TRAIN_TF::nnEval(numtype e0)
{
    *(numtype*)tfcall->inputs[n_wts+12]=e0;

    tfcall->evaluate();
    ++n_update;
    return **(tfcall->outputs+(n_output-3));
}
void NNP_TRAIN_TF::allocate_neilist()
{
    num_neigh=*num_neigh_local;
    *num_neigh_local=0;
    int temp;
    for(int i=1;i<ENVIRON::NUM_THREAD;++i)
    {
        temp=num_neigh_local[i];
        num_neigh_local[i]=num_neigh;
        num_neigh+=temp;
    }

    TF_DeleteTensor(tfcall->InputValues[n_wts+1]);
    TF_DeleteTensor(tfcall->InputValues[n_wts+2]);
    TF_DeleteTensor(tfcall->InputValues[n_wts+3]);
    TF_DeleteTensor(tfcall->InputValues[n_wts+4]);

    dim[n_wts+1][0]=num_neigh;
    dim[n_wts+2][0]=num_neigh;
    dim[n_wts+3][0]=num_neigh;
    dim[n_wts+4][0]=num_neigh;
    tfcall->inputs[n_wts+1] = TF_TensorData( tfcall->InputValues[n_wts+1]=TF_AllocateTensor(TF_INT32,dim[n_wts+1],ndim[n_wts+1],sz_in[n_wts+1]*num_neigh*sizeof(int)) ); 
    tfcall->inputs[n_wts+2] = TF_TensorData( tfcall->InputValues[n_wts+2]=TF_AllocateTensor(TF_INT32,dim[n_wts+2],ndim[n_wts+2],sz_in[n_wts+2]*num_neigh*sizeof(int)) ); 
    tfcall->inputs[n_wts+3] = TF_TensorData( tfcall->InputValues[n_wts+3]=TF_AllocateTensor(TF_INT32,dim[n_wts+3],ndim[n_wts+3],sz_in[n_wts+3]*num_neigh*sizeof(int)) ); 
    tfcall->inputs[n_wts+4] = TF_TensorData( tfcall->InputValues[n_wts+4]=TF_AllocateTensor(TF_INT32,dim[n_wts+4],ndim[n_wts+4],sz_in[n_wts+4]*num_neigh*sizeof(int)) ); 
}

// #define FF_NO_L2

void NNP_TRAIN_TF::apply_gradient(numtype lr,int thread_id)
{
    lr*=std::sqrt(1-std::pow(beta_2,n_update))/(1-std::pow(beta_1,n_update));
    int szi;
    numtype *wti,*a1i,*a2i,*gdi,grad;
    int iend=iwt_end[thread_id];
    for(int i=iwt_begin[thread_id];i<iend;++i)
    {
        szi=sz_in[i];
        wti=wts[i];
        a1i=acc1[i];
        a2i=acc2[i];
        gdi=tfcall->outputs[i];
        switch (i)
        {
#ifdef FF_NO_L2
        case 0: //bond, b0>=0.5, nol2; a>=0; b, no l2; c>=0;
            for(int j=0;j<szi;j+=4) //b0
            {
                grad=gdi[j];
                wti[j] -= (1+std::fabs(wti[j]))* lr * (a1i[j]=beta_1*a1i[j] + cbta_1*grad) / (std::sqrt(a2i[j]=beta_2*a2i[j] + cbta_2*sqr(grad)) + epsl);
                wti[j]=std::max(wti[j],0.5f);
            }
            for(int j=1;j<szi;j+=2) //a c
            {
                grad=gdi[j];
                wti[j] -= (1+std::fabs(wti[j]))* lr * (a1i[j]=beta_1*a1i[j] + cbta_1*grad) / (std::sqrt(a2i[j]=beta_2*a2i[j] + cbta_2*sqr(grad)) + epsl);
                wti[j]=std::max(wti[j],0.F);
                // std::cerr<<"bond "<<j<<" "<<grad<<" "<<wti[j]<<"\n";
            }
            for(int j=2;j<szi;j+=4) //b
            {
                grad=gdi[j];
                wti[j] -= (1+std::fabs(wti[j]))* lr * (a1i[j]=beta_1*a1i[j] + cbta_1*grad) / (std::sqrt(a2i[j]=beta_2*a2i[j] + cbta_2*sqr(grad)) + epsl);
            }
            break;
        case 1: //angle, -1<cb0<=.5 (60 deg), nol2; a>=0; b, no l2; c>=0;
            for(int j=0;j<szi;j+=4) //cb0
            {
                grad=gdi[j];
                wti[j] -= (1+std::fabs(wti[j]))* lr * (a1i[j]=beta_1*a1i[j] + cbta_1*grad) / (std::sqrt(a2i[j]=beta_2*a2i[j] + cbta_2*sqr(grad)) + epsl);
                wti[j]=std::clamp(wti[j],-1.f,0.5f);
            }
            for(int j=1;j<szi;j+=2) //a c
            {
                grad=gdi[j];
                wti[j] -= (1+std::fabs(wti[j]))* lr * (a1i[j]=beta_1*a1i[j] + cbta_1*grad) / (std::sqrt(a2i[j]=beta_2*a2i[j] + cbta_2*sqr(grad)) + epsl);
                wti[j]=std::max(wti[j],0.F);
            }
            for(int j=2;j<szi;j+=4) //b
            {
                grad=gdi[j];
                wti[j] -= (1+std::fabs(wti[j]))* lr * (a1i[j]=beta_1*a1i[j] + cbta_1*grad) / (std::sqrt(a2i[j]=beta_2*a2i[j] + cbta_2*sqr(grad)) + epsl);
            }
            break;
        case 2: //dihedral
            for(int j=0;j<szi;++j)
            {
                grad=gdi[j];
                wti[j] -= (1+std::fabs(wti[j]))* lr * (a1i[j]=beta_1*a1i[j] + cbta_1*grad) / (std::sqrt(a2i[j]=beta_2*a2i[j] + cbta_2*sqr(grad)) + epsl);
            }
            break;
        case 3: // coef_p , e>=0; s>=.5,no l2; (c8 c10 c12) norm 1 no l2; q
            {
                for(int j=0;j<szi;j+=6) //e
                {
                    grad=gdi[j];
                    wti[j] -= (1+std::fabs(wti[j]))* lr * (a1i[j]=beta_1*a1i[j] + cbta_1*grad) / (std::sqrt(a2i[j]=beta_2*a2i[j] + cbta_2*sqr(grad)) + epsl);
                    wti[j]=std::max(wti[j],0.f);
                }
                for(int j=1;j<szi;j+=6) //s
                {
                    grad=gdi[j];
                    wti[j] -= (1+std::fabs(wti[j]))* lr * (a1i[j]=beta_1*a1i[j] + cbta_1*grad) / (std::sqrt(a2i[j]=beta_2*a2i[j] + cbta_2*sqr(grad)) + epsl);
                    wti[j]=std::max(wti[j],0.5f);
                }
                numtype norm=0,a2mean=0;
#ifdef NNP_TRAIN_TF_NORMALIZE_CHARGE
                for(int j=5;j<szi;j+=6*4) //q
                {
                    for(int k=j+6*3;k>j;k-=6)
                    {
                        grad=gdi[k];
                        wti[k] -= (1+std::fabs(wti[k]))* lr * (a1i[k]=beta_1*a1i[k] + cbta_1*grad) / (std::sqrt(a2i[k]=beta_2*a2i[k] + cbta_2*sqr(grad)) + epsl);
                    }
                    grad=gdi[j];
                    a1i[j]=beta_1*a1i[j] + cbta_1*grad;
                    a2mean+=(a2i[j]=beta_2*a2i[j] + cbta_2*sqr(grad));
                    norm+=std::fabs(wti[j]);
                }
                a2mean=(1+norm/npair)*lr/(std::sqrt(a2mean/npair) + epsl);
                norm=0;
                for(int j=5,k=0;j<szi;j+=6*4,++k)
                {
                    norm+=(wti[j]-=a1i[j]*a2mean)*num_pair[k];
                }
                norm/=npair;
                for(int j=5,k=0;j<szi;j+=6*4,++k)
                {
                    wti[j]-=norm/num_pair[k];
                }
#else
                for(int j=5;j<szi;j+=6) //q
                {
                    grad=gdi[j];
                    wti[j] -= (1+std::fabs(wti[j]))* lr * (a1i[j]=beta_1*a1i[j] + cbta_1*grad) / (std::sqrt(a2i[j]=beta_2*a2i[j] + cbta_2*sqr(grad)) + epsl);
                }
#endif
                for(int j=2;j<szi;j+=6) //cs the sqr momentum should be shared!
                {
                    norm=a2mean=0;
                    for(int k=j+2;k>=j;--k)
                    {
                        grad=gdi[k];
                        a1i[k]=beta_1*a1i[k] + cbta_1*grad;
                        a2mean+=(a2i[k]=beta_2*a2i[k] + cbta_2*sqr(grad));
                    }
                    a2mean=lr/(std::sqrt(a2mean*.33333F) + epsl);
                    for(int k=j+2;k>=j;--k)
                    {
                        wti[k] -= a1i[k]*a2mean;
                        norm+=sqr(wti[k]=std::max(wti[k],0.f));
                    }
                    norm=1/std::sqrt(norm);
                    for(int k=j+2;k>=j;--k)
                    {
                        wti[k]*=norm;
                    }
                }
            }
            break;
#else
        case 0: //bond, b0>=0.5, nol2; a>=0; b, no l2; c>=0;
            for(int j=0;j<szi;j+=4) //b0
            {
                grad=gdi[j];
                wti[j] -= (1+std::fabs(wti[j]))* lr * (a1i[j]=beta_1*a1i[j] + cbta_1*grad) / (std::sqrt(a2i[j]=beta_2*a2i[j] + cbta_2*sqr(grad)) + epsl);
                wti[j]=std::max<numtype>(wti[j],0.5f);
            }
            for(int j=1;j<szi;j+=2) //a c
            {
                grad=gdi[j];
                wti[j] -= (1+std::fabs(wti[j]))* lr * (a1i[j]=beta_1*a1i[j] + cbta_1*grad) / (std::sqrt(a2i[j]=beta_2*a2i[j] + cbta_2*sqr(grad)) + epsl);
                wti[j]=std::max<numtype>(wti[j],0.F);
                // std::cerr<<"bond "<<j<<" "<<grad<<" "<<wti[j]<<"\n";
            }
            for(int j=2;j<szi;j+=4) //b
            {
                grad=gdi[j];
                wti[j] -= (1+std::fabs(wti[j]))* lr * (a1i[j]=beta_1*a1i[j] + cbta_1*grad) / (std::sqrt(a2i[j]=beta_2*a2i[j] + cbta_2*sqr(grad)) + epsl);
            }
            break;
        case 1: //angle, -1<cb0<=.5 (60 deg), nol2; a>=0; b, no l2; c>=0;
            for(int j=0;j<szi;j+=4) //cb0
            {
                grad=gdi[j];
                wti[j] -= (1+std::fabs(wti[j]))* lr * (a1i[j]=beta_1*a1i[j] + cbta_1*grad) / (std::sqrt(a2i[j]=beta_2*a2i[j] + cbta_2*sqr(grad)) + epsl);
                wti[j]=std::clamp<numtype>(wti[j],-1.f,0.5f);
            }
            for(int j=1;j<szi;j+=2) //a c
            {
                grad=gdi[j];
                wti[j] -= (1+std::fabs(wti[j]))* lr * (a1i[j]=beta_1*a1i[j] + cbta_1*grad) / (std::sqrt(a2i[j]=beta_2*a2i[j] + cbta_2*sqr(grad)) + epsl);
                wti[j]=std::max<numtype>(wti[j],0.F);
            }
            for(int j=2;j<szi;j+=4) //b
            {
                grad=gdi[j];
                wti[j] -= (1+std::fabs(wti[j]))* lr * (a1i[j]=beta_1*a1i[j] + cbta_1*grad) / (std::sqrt(a2i[j]=beta_2*a2i[j] + cbta_2*sqr(grad)) + epsl);
            }
            break;
        case 2: //dihedral
            for(int j=0;j<szi;++j)
            {
                grad=gdi[j];
                wti[j] -= (1+std::fabs(wti[j]))* lr * (a1i[j]=beta_1*a1i[j] + cbta_1*grad) / (std::sqrt(a2i[j]=beta_2*a2i[j] + cbta_2*sqr(grad)) + epsl);
            }
            break;
        case 3: // coef_p , e>=0; s>=.5,no l2; (c8 c10 c12) norm 1 no l2; q
            {
                for(int j=0;j<szi;j+=6) //e
                {
                    grad=gdi[j]+wti[j]*l2;
                    wti[j] -= (1+std::fabs(wti[j]))* lr * (a1i[j]=beta_1*a1i[j] + cbta_1*grad) / (std::sqrt(a2i[j]=beta_2*a2i[j] + cbta_2*sqr(grad)) + epsl);
                    wti[j]=std::max<numtype>(wti[j],0.f);
                }
                for(int j=1;j<szi;j+=6) //s
                {
                    grad=gdi[j];
                    wti[j] -= (1+std::fabs(wti[j]))* lr * (a1i[j]=beta_1*a1i[j] + cbta_1*grad) / (std::sqrt(a2i[j]=beta_2*a2i[j] + cbta_2*sqr(grad)) + epsl);
                    wti[j]=std::max<numtype>(wti[j],0.5f);
                }
                numtype norm=0,a2mean=0;
#ifdef NNP_TRAIN_TF_NORMALIZE_CHARGE
                for(int j=5;j<szi;j+=6*4) //q
                {
                    for(int k=j+6*3;k>j;k-=6)
                    {
                        grad=gdi[k]+wti[j]*l2;
                        wti[k] -= (1+std::fabs(wti[k]))* lr * (a1i[k]=beta_1*a1i[k] + cbta_1*grad) / (std::sqrt(a2i[k]=beta_2*a2i[k] + cbta_2*sqr(grad)) + epsl);
                    }
                    grad=gdi[j]+wti[j]*l2;
                    a1i[j]=beta_1*a1i[j] + cbta_1*grad;
                    a2mean+=(a2i[j]=beta_2*a2i[j] + cbta_2*sqr(grad));
                    norm+=std::fabs(wti[j]);
                }
                a2mean=(1+norm/npair)*lr/(std::sqrt(a2mean/npair) + epsl);
                norm=0;
                for(int j=5,k=0;j<szi;j+=6*4,++k)
                {
                    norm+=(wti[j]-=a1i[j]*a2mean)*num_pair[k];
                }
                norm/=npair;
                for(int j=5,k=0;j<szi;j+=6*4,++k)
                {
                    wti[j]-=norm/num_pair[k];
                }
#else
                for(int j=5;j<szi;j+=6) //q
                {
                    grad=gdi[j]+wti[j]*l2;
                    wti[j] -= (1+std::fabs(wti[j]))* lr * (a1i[j]=beta_1*a1i[j] + cbta_1*grad) / (std::sqrt(a2i[j]=beta_2*a2i[j] + cbta_2*sqr(grad)) + epsl);
                }
#endif
                for(int j=2;j<szi;j+=6) //cs the sqr momentum should be shared!
                {
                    norm=a2mean=0;
                    for(int k=j+2;k>=j;--k)
                    {
                        grad=gdi[k];
                        a1i[k]=beta_1*a1i[k] + cbta_1*grad;
                        a2mean+=(a2i[k]=beta_2*a2i[k] + cbta_2*sqr(grad));
                    }
                    a2mean=lr/(std::sqrt(a2mean*.33333F) + epsl);
                    for(int k=j+2;k>=j;--k)
                    {
                        wti[k] -= a1i[k]*a2mean;
                        norm+=sqr(wti[k]=std::max<numtype>(wti[k],0.f));
                    }
                    norm=1/std::sqrt(norm);
                    for(int k=j+2;k>=j;--k)
                    {
                        wti[k]*=norm;
                    }
                }
            }
            break;
#endif
        case 5: // coef_d , must <=-.01 (0 will freeze all parameters!)
            for(int j=0;j<szi;++j)
            {
                grad=gdi[j]+wti[j]*l2;
                wti[j] -= (1+std::fabs(wti[j]))* lr * (a1i[j]=beta_1*a1i[j] + cbta_1*grad) / (std::sqrt(a2i[j]=beta_2*a2i[j] + cbta_2*sqr(grad)) + epsl);
                wti[j]=std::min<numtype>(wti[j],-0.01f);
            }
            break;
        case 6: // coef_r , no l2,  must 0<=x<=cf
            for(int j=0;j<szi;++j)
            {
                grad=gdi[j];
                wti[j] -= (1+std::fabs(wti[j]))* lr * (a1i[j]=beta_1*a1i[j] + cbta_1*grad) / (std::sqrt(a2i[j]=beta_2*a2i[j] + cbta_2*sqr(grad)) + epsl);
                wti[j]=std::clamp<numtype>(wti[j],0.F,cutoff);
            }
            break;
        default:
            for(int j=0;j<szi;++j)
            {
                grad=gdi[j]+wti[j]*l2;
                wti[j] -= (1+std::fabs(wti[j]))* lr * (a1i[j]=beta_1*a1i[j] + cbta_1*grad) / (std::sqrt(a2i[j]=beta_2*a2i[j] + cbta_2*sqr(grad)) + epsl);
            }
            break;
        }
    }
}

void NNP_TRAIN_TF::save_wts(const char* weightfile,const char* optzfile)
{
    FILE* fp=fopen(weightfile,"wb");
    for(int i=0;i<n_wts;++i)
        fwrite(tfcall->inputs[i],sizeof(numtype),sz_in[i],fp);
    fclose(fp);

    fp=fopen(optzfile,"wb");
    fwrite(&n_update,sizeof(int),1,fp);
    for(int i=0;i<n_wts;++i)
    {
        fwrite(acc1[i],sizeof(numtype),sz_in[i],fp);
        fwrite(acc2[i],sizeof(numtype),sz_in[i],fp);
    }
    fclose(fp);
}

int NNP_TRAIN_TF::load_train_info(const char* fname_data, const char* fname_sym)
{
    FILE *fp=fopen(fname_data,"r");
    fscanf(fp,"%d ",&n_train);
    create2DArray(fnames,n_train,SZ_FBF);
    fnames_shuffled=new char*[n_train];
    memcpy(fnames_shuffled,fnames,n_train*sizeof(char*));
    for(int i=0;i<n_train;++i)
    {
        fscanf(fp,"%s ",fnames[i]);
    }
    fclose(fp);
    std::cerr<<"loaded "<<n_train<<" configurations\n";
    fp=fopen(fname_sym,"r");
    int maxs,maxg;
    int **sgpi,*sgpij;
    fscanf(fp,"%d %d %d ",&n_sym,&maxs,&maxg);
    create3DArray(sym_group,n_sym,maxs,maxg);
    create2DArray<int*>(sym_group_shuffed,n_sym,maxs);
    create2DArray(x_temp,maxs,maxg*3);
    create2DArray(f_temp,maxs,maxg*3);
    create1DArray(symsz,n_sym);
    create1DArray(grpsz,n_sym);
    memcpy(*sym_group_shuffed,*sym_group,n_sym*maxs*sizeof(int*));

    for(int i=0;i<n_sym;++i)
    {
        std::cerr<<"symmetry group "<<i<<"\n";
        fscanf(fp,"%d %d ",symsz+i,grpsz+i);
        maxs=symsz[i]; maxg=grpsz[i];
        sgpi=sym_group[i];
        for(int j=0;j<maxs;++j)
        {
            std::cerr<<" ";
            sgpij=sgpi[j];
            for(int k=0;k<maxg;++k)
            {
                fscanf(fp,"%d ",sgpij+k);
                std::cerr<<" "<<sgpij[k];
            }
            std::cerr<<"\n";
        }
    }
    fclose(fp);

    return n_train;
}

double NNP_TRAIN_TF::load_data(const char* fname)
{
    FILE *fp=fopen(fname,"rb");
    double erg;
    fread(*ENVIRON::x,sizeof(numtype),3*ENVIRON::natom,fp);
    fread(&erg,sizeof(double),1,fp);
    fread(*ENVIRON::f_prev,sizeof(numtype),3*ENVIRON::natom,fp);
    fclose(fp);
    
    //data augmentation
    int szg,szs,rpid;
    int **sgi,*sgij,**ssi,*ssij;
    int id1;
    numtype *x0,*f0, *tempx,*tempf;
    for(int mid=0;mid<ENVIRON::nmol;++mid)
    {
        x0=ENVIRON::x[rpid=ENVIRON::mol_repre_id[mid]];
        f0=ENVIRON::f_prev[rpid];
        for(int i=0;i<n_sym;++i)
        {
            szg=grpsz[i];
            szs=symsz[i];
            sgi=sym_group[i];
            ssi=sym_group_shuffed[i];
            shuffle(ssi,szs);

            //copy x to x_temp
            for(int j=0;j<szs;++j)
            {
                tempx=x_temp[j];tempf=f_temp[j];
                sgij=sgi[j];
                for(int k=0;k<szg;++k)
                {
                    id1=3*sgij[k];
                    memcpy(tempx+3*k,x0+id1,3*sizeof(numtype));
                    memcpy(tempf+3*k,f0+id1,3*sizeof(numtype));
                }
            }
            //copy x_temp to x
            for(int j=0;j<szs;++j)
            {
                tempx=x_temp[j];tempf=f_temp[j];
                ssij=ssi[j];
                for(int k=0;k<szg;++k)
                {
                    id1=3*ssij[k];
                    memcpy(x0+id1,tempx+3*k,3*sizeof(numtype));
                    memcpy(f0+id1,tempf+3*k,3*sizeof(numtype));
                }
            }
        }
    }
    return erg;
}

#ifndef NNP_USE_X_MOL
void NNP_TRAIN_TF::update_intra(int thread_id, BOND *bnd, ANGLE* agl, DIHEDRAL *dih)
{
    int iend=bnd->i_end[thread_id],ibegin=bnd->i_start[thread_id];
    int *dst=(int*)tfcall->inputs[n_wts+5] + ibegin*3,*src=*(bnd->bond) + ibegin*3;
    for(int i=ibegin;i<iend;++i)
    {
        *(++dst)=inv_atom_id[*(++src)];
        *(++dst)=inv_atom_id[*(++src)];
        ++dst; ++src;
        
    }
    
    iend=agl->i_end[thread_id];
    ibegin=agl->i_start[thread_id];
    dst=(int*)tfcall->inputs[n_wts+6] + ibegin*4;
    src=*(agl->angle) + ibegin*4;
    for(int i=ibegin;i<iend;++i)
    {
        *(++dst)=inv_atom_id[*(++src)];
        *(++dst)=inv_atom_id[*(++src)];
        *(++dst)=inv_atom_id[*(++src)];
        ++dst; ++src;
    }

    iend=dih->i_end[thread_id];
    ibegin=dih->i_start[thread_id];
    dst=(int*)tfcall->inputs[n_wts+7] + ibegin*5;
    src=*(dih->dihedral) + ibegin*5;
    for(int i=ibegin;i<iend;++i)
    {
        *(++dst)=inv_atom_id[*(++src)];
        *(++dst)=inv_atom_id[*(++src)];
        *(++dst)=inv_atom_id[*(++src)];
        *(++dst)=inv_atom_id[*(++src)];
        ++dst; ++src;
    }

}
#endif

NNP_DEPLOY_TF::NNP_DEPLOY_TF(const char* paramfile,numtype cf, BOND *bnd, ANGLE* agl, DIHEDRAL *dih)
{
#ifndef FLOAT_PRECESION
    END_PROGRAM("NNP must use FLOAT_PRECESION");
#endif
    if(ENVIRON::nmoltype != 1) END_PROGRAM("NNP does not support multiple molecules");
#ifndef NNP_USE_X_MOL
    create1DArray(inv_atom_id,ENVIRON::natom);
#endif

    tfcall=new TFCALL[ENVIRON::nmoltype];
    char path_nn[SZ_FBF],dt; 
    char **names_in,**names_out;
    TF_DataType dtype[n_input];
    int index_in[n_input], index_out[2];
    create2DArray(names_in,n_input,SZ_FBF);
    create2DArray(names_out,2,SZ_FBF);
    create1DArray(sz_in,n_input);

    int64_t *dimi;
    int ndims;
    create2DArray(dim,n_input,4);
    create1DArray(ndim,n_input);

    FILE *fp=fopen(paramfile,"r");
    if(fscanf(fp,"%s ",path_nn)!=1) END_PROGRAM("read error");
    for(int i=0;i<n_input;++i)
    {   
        dimi=dim[i];
        if(fscanf(fp,"%s %c %d %" SCNd64 " %" SCNd64 " %" SCNd64 " %" SCNd64 " ",names_in[i],&dt,index_in+i,dimi,dimi+1,dimi+2,dimi+3)!=7) END_PROGRAM("read error");
        switch (dt)
        {
        case 'f':
        case 'F':
            dtype[i]=TF_FLOAT;
            break;
        case 'i':
        case 'I':
            dtype[i]=TF_INT32;
            break;
        default:
            dtype[i]=TF_FLOAT;
            END_PROGRAM("unknown dtype");
            break;
        }
        sz_in[i]=1;
        for(ndims=0;dimi[ndims];++ndims)
        {
            if(dimi[ndims]==-2) dimi[ndims]=ENVIRON::natom;
            sz_in[i]*=(int)dimi[ndims];
        }
        ndim[i]=ndims;
        // std::cerr<<i<<" "<<names_in[i]<<" "<<dt<<" "<<sz_in[i]<<"\n";
    }
    for(int i=0;i<n_input;++i) sz_in[i]=std::abs(sz_in[i]);

    for(int i=0;i<2;++i) if(fscanf(fp,"%s %d ",names_out[i],index_out+i)!=2) END_PROGRAM("read error");
    tfcall->init(path_nn,n_input,names_in,dim,dtype,index_in,2,names_out,index_out);
    fclose(fp);

    destroy2DArray(names_in);
    destroy2DArray(names_out);
    // destroy2DArray(dim);
    
    sz_in[5] *= ( dim[5][0]=bnd->nbonds );
    tfcall->inputs[5] = TF_TensorData( tfcall->InputValues[5]=TF_AllocateTensor(TF_INT32,dim[5],ndim[5],sz_in[5]*sizeof(int)) );
    memcpy(tfcall->inputs[5],*(bnd->bond),sz_in[5]*sizeof(int));

    sz_in[6] *= ( dim[6][0]=agl->nangles );
    tfcall->inputs[6] = TF_TensorData( tfcall->InputValues[6]=TF_AllocateTensor(TF_INT32,dim[6],ndim[6],sz_in[6]*sizeof(int)) );
    memcpy(tfcall->inputs[6],*(agl->angle),sz_in[6]*sizeof(int));

    sz_in[7] *= ( dim[7][0]=dih->ndihedrals );
    tfcall->inputs[7] = TF_TensorData( tfcall->InputValues[7]=TF_AllocateTensor(TF_INT32,dim[7],ndim[7],sz_in[7]*sizeof(int)) );
    memcpy(tfcall->inputs[7],*(dih->dihedral),sz_in[7]*sizeof(int));


    *(int*)tfcall->inputs[8]=ENVIRON::nmol;
    *(numtype*)tfcall->inputs[9]=cutoff=cf;
    *(numtype*)tfcall->inputs[10]=ENVIRON::bl;

    // capacity=cap*ENVIRON::natom;
    // create1DArray(i_all,capacity);
    // create1DArray(j_all,capacity);
    // create1DArray(im_all,capacity);
    // create1DArray(tp_all,2*capacity);

}

NNP_DEPLOY_TF::~NNP_DEPLOY_TF()
{
#ifndef NNP_USE_X_MOL
    destroy1DArray(inv_atom_id);
#endif
    destroy1DArray(sz_in);
    destroy2DArray(dim);
    destroy1DArray(ndim);
    // destroy1DArray(i_all);
    // destroy1DArray(j_all);
    // destroy1DArray(im_all);
    // destroy1DArray(tp_all);
    delete[] tfcall;
}

void NNP_DEPLOY_TF::compute_pair(NEIGHLIST* list, numtype *erg)
{
    int n_local=list->n_local;
    int *local_list=list->local_list;
    int ii;

    numtype *xs=(numtype*)tfcall->inputs[0];
    for(int i=0;i<n_local;++i)
    {
        ii=local_list[i];
#ifdef NNP_USE_X_MOL
        memcpy(xs+3*ENVIRON::atom_id[ii],ENVIRON::x[ii],3*sizeof(numtype));
#else
        memcpy(xs+3*ii,ENVIRON::x[ii],3*sizeof(numtype));
#endif
    }
}
void NNP_DEPLOY_TF::gather_neilist(NEIGHLIST* list)
{
    int n_local=list->n_local, **nlist=list->nei_list,*num_nei=list->num_neigh;
    int *local_list=list->local_list;
    int inum,*ilist,ii,jj,ii_mol,*ipair_typ;
    int thread_id=list->thread_id;
    int *ispecial, **special=list->special;

    int ofst=num_neigh_local[thread_id],
        *i_all_local =(int*)(tfcall->inputs[1])+ofst,
        *j_all_local =(int*)(tfcall->inputs[2])+ofst,
        *im_all_local=(int*)(tfcall->inputs[3])+ofst,
        *tp_all_local=(int*)(tfcall->inputs[4])+ofst*2;

    for(int i=0;i<n_local;++i)
    {
        ii=local_list[i];
        ii_mol=ENVIRON::atom_id[ii];
        ilist=nlist[i];
        inum=num_nei[i];
        ispecial=special[i];
        ipair_typ=ENVIRON::pairtype[ENVIRON::typ[ii]];

#ifndef NNP_USE_X_MOL
        inv_atom_id[ii_mol]=ii;
#endif

        for(int j=0; j<inum; ++j)
        {
            jj=ilist[j];
#ifdef NNP_USE_X_MOL
            *(i_all_local++)=ii_mol;
            *(j_all_local++)=ENVIRON::atom_id[jj];
#else
            *(i_all_local++)=ii;
            *(j_all_local++)=jj<ENVIRON::natom? jj : ENVIRON::i_ghost[jj-ENVIRON::natom];
#endif
            *(im_all_local++)=ii_mol;
            *(tp_all_local++)=ipair_typ[ENVIRON::typ[jj]];
            *(tp_all_local++)=ispecial[j];
        }        
    }

}

void NNP_DEPLOY_TF::nnEval()
{
    tfcall->evaluate();
}

void NNP_DEPLOY_TF::evalFinalize(LOCAL *local)
{
#ifdef NNP_USE_X_MOL
    int iend=local->i_local_end;
    numtype* f0=tfcall->outputs[1];
    for(int i=local->i_local_start;i<iend;++i)
    {
        memcpy(ENVIRON::f_mol[i],f0+3*i,3*sizeof(numtype));
    }
#else
    int ibegin=local->i_local_start,inum=local->i_local_end-ibegin;
    numtype* f0=tfcall->outputs[1];
    memcpy(*ENVIRON::f+3*ibegin,f0+3*ibegin,3*inum*sizeof(numtype));
#endif
}

void NNP_DEPLOY_TF::allocate_neilist()
{
    num_neigh=*num_neigh_local;
    *num_neigh_local=0;
    int temp;
    for(int i=1;i<ENVIRON::NUM_THREAD;++i)
    {
        temp=num_neigh_local[i];
        num_neigh_local[i]=num_neigh;
        num_neigh+=temp;
    }

    TF_DeleteTensor(tfcall->InputValues[1]);
    TF_DeleteTensor(tfcall->InputValues[2]);
    TF_DeleteTensor(tfcall->InputValues[3]);
    TF_DeleteTensor(tfcall->InputValues[4]);

    dim[1][0]=num_neigh;
    dim[2][0]=num_neigh;
    dim[3][0]=num_neigh;
    dim[4][0]=num_neigh;
    tfcall->inputs[1] = TF_TensorData( tfcall->InputValues[1]=TF_AllocateTensor(TF_INT32,dim[1],ndim[1],sz_in[1]*num_neigh*sizeof(int)) ); 
    tfcall->inputs[2] = TF_TensorData( tfcall->InputValues[2]=TF_AllocateTensor(TF_INT32,dim[2],ndim[2],sz_in[2]*num_neigh*sizeof(int)) ); 
    tfcall->inputs[3] = TF_TensorData( tfcall->InputValues[3]=TF_AllocateTensor(TF_INT32,dim[3],ndim[3],sz_in[3]*num_neigh*sizeof(int)) ); 
    tfcall->inputs[4] = TF_TensorData( tfcall->InputValues[4]=TF_AllocateTensor(TF_INT32,dim[4],ndim[4],sz_in[4]*num_neigh*sizeof(int)) ); 
}

#ifndef NNP_USE_X_MOL
void NNP_DEPLOY_TF::update_intra(int thread_id, BOND *bnd, ANGLE* agl, DIHEDRAL *dih)
{
    int iend=bnd->i_end[thread_id],ibegin=bnd->i_start[thread_id];
    int *dst=(int*)tfcall->inputs[5] + ibegin*3,*src=*(bnd->bond) + ibegin*3;
    for(int i=ibegin;i<iend;++i)
    {
        *(++dst)=inv_atom_id[*(++src)];
        *(++dst)=inv_atom_id[*(++src)];
        ++dst; ++src;
    }
    
    iend=agl->i_end[thread_id];
    ibegin=agl->i_start[thread_id];
    dst=(int*)tfcall->inputs[6] + ibegin*4;
    src=*(agl->angle) + ibegin*4;
    for(int i=ibegin;i<iend;++i)
    {
        *(++dst)=inv_atom_id[*(++src)];
        *(++dst)=inv_atom_id[*(++src)];
        *(++dst)=inv_atom_id[*(++src)];
        ++dst; ++src;
    }

    iend=dih->i_end[thread_id];
    ibegin=dih->i_start[thread_id];
    dst=(int*)tfcall->inputs[7] + ibegin*5;
    src=*(dih->dihedral) + ibegin*5;
    for(int i=ibegin;i<iend;++i)
    {
        *(++dst)=inv_atom_id[*(++src)];
        *(++dst)=inv_atom_id[*(++src)];
        *(++dst)=inv_atom_id[*(++src)];
        *(++dst)=inv_atom_id[*(++src)];
        ++dst; ++src;
    }

}
#endif
