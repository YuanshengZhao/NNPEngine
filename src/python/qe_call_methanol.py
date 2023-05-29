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
def npz2bin(fname, skip_exist=False):
    if skip_exist and os.path.isfile(fname+".bin"): return
    with np.load(fname) as fp:
        x,e,f = fp["x"].reshape([-1]),fp["e"],fp["f"].reshape([-1])
    with open(fname+".bin", "wb") as fp:
        fp.write(struct.pack('f'*len(x), *x))
        fp.write(struct.pack('d', e)) # energy is stored as double to avoid precesion loss
        fp.write(struct.pack('f'*len(f), *f))

    

if __name__=="__main__":
    with open(sys.argv[1],"r") as fp:
        if int(fp.readline())!=NMOL*NATM: raise RuntimeError("wrong num atoms")
        lns=fp.readline().split()
        BL=np.round(float(lns[-1])-float(lns[-2]),4)
        for _ in range(NMOL*NATM):
            lns=fp.readline().split()
            cod.append([float(xxx) for xxx in lns[-3:]])
    cod=np.array(cod)

    qe_run()
    save_qe()

    flst=glob.glob("/mnt/ntfs2/methanol_nnp/aimd/train_data/qe_%f*.npz"%(BL))
    tms=[float(xx.split("_")[-1][:-4]) for xx in flst]
    flst=np.array(flst)[np.argsort(tms)]

    for fn in flst:
        npz2bin(fn,True)
    
    ll=len(flst)
    lt=max(int(ll*.9),1)
    le=max(ll-lt,1)
    with open("/mnt/ntfs2/methanol_nnp/nnp_data/list_300_%.4f.txt"%(BL),"w") as fp:
        fp.write(str(lt)+"\n")
        for fn in flst[:lt]:
            fp.write(fn+".bin\n")
    with open("/mnt/ntfs2/methanol_nnp/nnp_data/listeval_300_%.4f.txt"%(BL),"w") as fp:
        fp.write(str(le)+"\n")
        for fn in flst[-le:]:
            fp.write(fn+".bin\n")