import numpy as np
import os
import UnitConverter as UC
import time
import sys
import struct
import glob

NMOL=100
NATM=6
elms=["C","H","H","H","H","O"]

cod=[]


def qe_read(fname):
    f=[]
    prs=None
    with open(fname,"r") as fp:
        while True:
            line=fp.readline()
            if line=="": break
            if "Forces acting on atoms (cartesian axes, Ry/au):" in line:
                fp.readline()
                for i in range(NMOL*NATM):
                    ls=fp.readline().split()
                    f.append([float(ls[ig]) for ig in [6,7,8]])  
            if "!" in line:
                e=(float(line.split()[-2]))
            if "total   stress" in line:
                prs=(float(line.split()[-1]))
    return np.array(f)*((UC.Hartree/2/UC.BohrRadius)/(UC.KilocaloriesPerMole/UC.Angstrom)).scl,np.array(e),prs

def qe_run(stress=False):
    fl="""&control
 title = 'meoh_train',
 calculation = 'scf',
 restart_mode = 'from_scratch',
 tstress = .%s.,
 tprnfor = .TRUE.,
 disk_io = 'none',
 prefix = 'meoh_train',
 pseudo_dir='../pseudo/',
 outdir='./output/',
/
&system
 ibrav = 1,
 A = %f,
 B = %f,
 C = %f,
 ntyp = 3,
 nat = 600,
 nbnd = 700,
 ecutwfc = 60.0,
 ecutrho = 480.0,
 vdw_corr = 'dft-d3',
/
&electrons
 scf_must_converge = .TRUE.,
 startingwfc = 'atomic+random'
/
&ions
 ion_temperature = 'svr',
 tempw = 300.0,
/
ATOMIC_SPECIES
C 12.0d0 C.pbe-n-kjpaw_psl.1.0.0.UPF
H 2.00d0 H.pbe-rrkjus_psl.1.0.0.UPF
O 16.0d0 O.pbe-n-kjpaw_psl.0.1.UPF
ATOMIC_POSITIONS (angstrom)
"""%("TRUE" if stress else "FALSE",BL,BL,BL)
    with open("/mnt/ntfs2/methanol_nnp/aimd/train/meoh_train.qe","w") as fp:
       fp.write(fl)
       for i in range(NMOL*NATM):
           fp.write("%s %.15e %.15e %.15e\n"%(elms[i%NATM],cod[i,0],cod[i,1],cod[i,2]))
    if os.system("export OMP_NUM_THREADS=1 && cd /mnt/ntfs2/methanol_nnp/aimd/train && mpiexec -n 32 /home/yuansheng/quantum_espresso/qe-7.0_cpu/build/bin/pw.x -i meoh_train.qe > res.log"): raise ValueError("qe failed!")
def save_qe():
    ff,e,prs=qe_read("/mnt/ntfs2/methanol_nnp/aimd/train/res.log")
    if prs is not None: 
        np.savez("/mnt/ntfs2/methanol_nnp/aimd/train_data/qe_%f_%d.npz"%(BL,time.time()),x=cod,f=ff,e=e,p=prs)
        print("erg =",float(e),"Ry; prs =",float(prs),"Kbar")
    else:               
        np.savez("/mnt/ntfs2/methanol_nnp/aimd/train_data/qe_%f_%d.npz"%(BL,time.time()),x=cod,f=ff,e=e)
        print("erg =",float(e),"Ry")


def load_npz(fname):
    with np.load(fname) as fp:
        return fp["x"],fp["e"],fp["f"]
def npz2bin(fname, offset=0, skip_exist=False):
    if skip_exist and (offset==0) and os.path.isfile(fname+".bin"): return
    with np.load(fname) as fp:
        x,e,f = fp["x"].reshape([-1]),fp["e"]+offset,fp["f"].reshape([-1])
    with open(fname+".bin", "wb") as fp:
        fp.write(struct.pack('f'*len(x), *x))
        fp.write(struct.pack('d', e)) # energy is stored as double to avoid precesion loss
        fp.write(struct.pack('f'*len(f), *f))
    assert len(x)==len(f) and len(x)==3*NMOL*NATM
    # print(fname,len(x),len(f))

    

if __name__=="__main__":
    BL=np.round(float(sys.argv[1].split("_")[-2]),4)

    if len(sys.argv)>=3:
        if sys.argv[2]!="--no-qe": raise RuntimeError("bad argv")
    else:
        with open(sys.argv[1],"r") as fp:
            if int(fp.readline())!=NMOL*NATM: raise RuntimeError("wrong num atoms")
            lns=fp.readline().split()
            if BL!=np.round(float(lns[-1])-float(lns[-2]),4): raise RuntimeError("BL not match")
            for _ in range(NMOL*NATM):
                lns=fp.readline().split()
                cod.append([float(xxx) for xxx in lns[-3:]])
        cod=np.array(cod)
        
        qe_run()
        save_qe()

    els=[]
    flst=glob.glob("/mnt/ntfs2/methanol_nnp/aimd/train_data/qe_*_*.npz")
    for f in flst:
        with np.load(f) as npz:
            els.append(npz["e"])
    emean=-np.mean(els)
    print(emean)

    for fn in flst:
        npz2bin(fn,offset=emean)

    flst=glob.glob("/mnt/ntfs2/methanol_nnp/aimd/train_data/qe_%f*.npz"%(BL))
    tms=[float(xx.split("_")[-1][:-4]) for xx in flst]
    flst=np.array(flst)[np.argsort(tms)]
    
    ll=len(flst)
    ll-=(ll%10)
    fl1=flst[:ll].reshape([-1,10])
    fl2=flst[ll:]
    f_train=np.concatenate((fl1[:,:-1].reshape(-1),fl2))
    f_eval=fl1[:,-1].reshape(-1)
    if len(f_eval)==0: f_eval=f_train[-1:]

    with open("/mnt/ntfs2/methanol_nnp/nnp_data/list_300_%.4f.txt"%(BL),"w") as fp:
        fp.write(str(len(f_train))+"\n")
        for fn in f_train:
            fp.write(fn+".bin\n")
    with open("/mnt/ntfs2/methanol_nnp/nnp_data/listeval_300_%.4f.txt"%(BL),"w") as fp:
        fp.write(str(len(f_eval))+"\n")
        for fn in f_eval:
            fp.write(fn+".bin\n")
