import tensorflow as tf
import keras as K
import numpy as np
import struct
import os
import sys

NUM_DESC=20 # note: this must be the same as in pair_nnp.h!

NUM_ATOM=10
NUM_PAIR=10

def sft_sign(x):
    return x/tf.sqrt(1+tf.square(x))

_x=tf.keras.layers.Input([NUM_ATOM,NUM_DESC])
_y=tf.keras.layers.LocallyConnected1D(filters=16,kernel_size=1,data_format="channels_last")(_x)
_y=tf.keras.layers.Activation(sft_sign)(_y)
_y=tf.keras.layers.LocallyConnected1D(filters=8,kernel_size=3,strides=2,data_format="channels_first")(_y)
_y=tf.keras.layers.Activation(sft_sign)(_y)
_y=tf.keras.layers.Flatten()(_y)
_y=tf.keras.layers.Dense(16)(_y)
_y=tf.keras.layers.Activation(sft_sign)(_y)
_y=tf.keras.layers.Dense(1)(_y)

class MODEL_DEPLOY(tf.keras.Model):
    def __init__(self) -> None:
        super().__init__()
        self.mdl=tf.keras.Model(inputs=_x,outputs=_y)
        self.mdl.summary()
    def call(self,inputs):
        with tf.GradientTape() as g:
            g.watch(inputs)
            erg=tf.reduce_sum(self.mdl(inputs))
        return erg, g.gradient(erg,inputs)
mdl_dep=MODEL_DEPLOY()
mdl_dep(tf.convert_to_tensor(np.random.uniform(size=[1,NUM_ATOM,NUM_DESC])))

wts=mdl_dep.get_weights()
wts[3]=wts[3].reshape(np.flip(wts[3].shape))
cfw=np.random.uniform(-0,0,size=(NUM_PAIR,4,NUM_DESC))
cfd=np.random.uniform(-2,0,size=(NUM_PAIR,4,NUM_DESC))
cfr=np.random.uniform( 0,5,size=(NUM_PAIR,4,NUM_DESC))

ljfct,lj8,lj10,lj12=1/3**.5,1/3**.5,1/3**.5,1/3**.5
# ljfct,lj8,lj10,lj12=1,0,0,1
ff_scale,ff_scale_i=1,1
h_eps=2e-3 # opls -> 0.030 | 2e-3
pair_coef=np.array([
[[(0.066*0.066)**.5*4*ljfct*ff_scale, (3.50*3.50)**.5, lj8,lj10,lj12, (0.145*0.145)*332.06371*ff_scale]],
[[(0.066*h_eps)**.5*4*ljfct*ff_scale, (3.50*2.50)**.5, lj8,lj10,lj12, (0.145*0.060)*332.06371*ff_scale]],
[[(0.066*0.000)**.5*4*ljfct*ff_scale, (3.50*0.00)**.5, lj8,lj10,lj12, (0.145*0.418)*332.06371*ff_scale]],
[[(0.066*0.170)**.5*4*ljfct*ff_scale, (3.50*3.12)**.5, lj8,lj10,lj12, (0.145*-.683)*332.06371*ff_scale]],
[[(h_eps*h_eps)**.5*4*ljfct*ff_scale, (2.50*2.50)**.5, lj8,lj10,lj12, (0.060*0.060)*332.06371*ff_scale]],
[[(h_eps*0.000)**.5*4*ljfct*ff_scale, (2.50*0.00)**.5, lj8,lj10,lj12, (0.060*0.418)*332.06371*ff_scale]],
[[(h_eps*0.170)**.5*4*ljfct*ff_scale, (2.50*3.12)**.5, lj8,lj10,lj12, (0.060*-.683)*332.06371*ff_scale]],
[[(0.000*0.000)**.5*4*ljfct*ff_scale, (0.00*0.00)**.5, lj8,lj10,lj12, (0.418*0.418)*332.06371*ff_scale]],
[[(0.000*0.170)**.5*4*ljfct*ff_scale, (0.00*3.12)**.5, lj8,lj10,lj12, (0.418*-.683)*332.06371*ff_scale]],
[[(0.170*0.170)**.5*4*ljfct*ff_scale, (3.12*3.12)**.5, lj8,lj10,lj12, (-.683*-.683)*332.06371*ff_scale]],
])*np.array([[[1,1,1,1,1,1],[0,.4,1,1,1,0],[0,.4,1,1,1,0],[.5,1,1,1,1,5/6.]]])
wts=[pair_coef,cfw,cfd,cfr]+wts
bond_coef=np.array([
[1.529,0,0,268.0*ff_scale_i],
[1.410,0,0,320.0*ff_scale_i],
[1.090,0,0,340.0*ff_scale_i],
[0.960,0,0,553.0*ff_scale_i],
])
angle_coef=np.array([
[np.cos(110.7*np.pi/180), 25.065263841028393*ff_scale_i, 0.34531565734984593, 39.86556812703493 *ff_scale_i],
[np.cos(109.5*np.pi/180), 31.035138693856492*ff_scale_i, 0.34056135730550974, 52.67047953244677 *ff_scale_i],
[np.cos(108.5*np.pi/180), 32.1847926045434  *ff_scale_i, 0.33522167278748977, 57.54075148912716 *ff_scale_i],
[np.cos(107.8*np.pi/180), 18.55869561697983 *ff_scale_i, 0.3307054673345205 , 34.372035317347134*ff_scale_i],
[np.cos(109.5*np.pi/180), 31.035138693856492*ff_scale_i, 0.34056135730550974, 52.67047953244677 *ff_scale_i],
])
dihedral_coef=np.array([
[(0.1556 - 0)*ff_scale_i, (0 - 3*0.1556)*ff_scale_i, (0*2)*ff_scale_i, 0.1556*4*ff_scale_i],
[(0.1556 - 0)*ff_scale_i, (0 - 3*0.1556)*ff_scale_i, (0*2)*ff_scale_i, 0.1556*4*ff_scale_i],
[(1.1440 - 1)*ff_scale_i, (0 - 3*0.1440)*ff_scale_i, (1*2)*ff_scale_i, 0.1440*4*ff_scale_i],
[(0.1667 - 0)*ff_scale_i, (0 - 3*0.1667)*ff_scale_i, (0*2)*ff_scale_i, 0.1667*4*ff_scale_i],
[(0.1667 - 0)*ff_scale_i, (0 - 3*0.1667)*ff_scale_i, (0*2)*ff_scale_i, 0.1667*4*ff_scale_i],
])
wts=[bond_coef,angle_coef,dihedral_coef]+wts

def cut_func(rr,RCUT):
    return (1-1/(1+tf.square(rr-RCUT)))
# plt.plot(np.linspace(0,12,1000),cut_func(np.linspace(0,12,1000)))
def pair_func(rr,cfe,cfs,c8,c10,c12,qij,RCUT):
    sr2=tf.square(cfs/rr)
    sr6=sr2*sr2*sr2
    src2=tf.square(cfs/RCUT)
    src6=src2*src2*src2
    rrc=rr/RCUT
    return (qij*(1/rr+(rrc-2)/RCUT)+cfe*(sr6*(((c12*sr2+c10)*sr2+c8)*sr2-1)+src6*(((c12*(12*rrc-13)*src2+c10*(10*rrc-11))*src2+c8*(8*rrc-9))*src2-(6*rrc-7))))
def to_dist_pbc(rr,BL):
    return tf.sqrt(tf.reduce_sum(tf.square(rr-tf.math.round(rr/BL)*BL),axis=-1))
def pbc(rr,BL):
    return rr-tf.math.round(rr/BL)*BL
def desc(rr,_cw,_cd,_cr,RCUT):
    return cut_func(rr,RCUT)*_cw*tf.math.exp(_cd*tf.square(rr-_cr))

class TRAINER(tf.Module):
    @tf.function(input_signature=[
                                  tf.TensorSpec(list(wts[0].shape), tf.float32),
                                  tf.TensorSpec(list(wts[1].shape), tf.float32),
                                  tf.TensorSpec(list(wts[2].shape), tf.float32),
                                  tf.TensorSpec(list(wts[3].shape), tf.float32),
                                  tf.TensorSpec(list(wts[4].shape), tf.float32),
                                  tf.TensorSpec(list(wts[5].shape), tf.float32),
                                  tf.TensorSpec(list(wts[6].shape), tf.float32),
                                  tf.TensorSpec(list(wts[7].shape), tf.float32),
                                  tf.TensorSpec(list(wts[8].shape), tf.float32),
                                  tf.TensorSpec(list(wts[9].shape), tf.float32),
                                  tf.TensorSpec(list(wts[10].shape), tf.float32),
                                  tf.TensorSpec(list(wts[11].shape), tf.float32),
                                  tf.TensorSpec(list(wts[12].shape), tf.float32),
                                  tf.TensorSpec(list(wts[13].shape), tf.float32),
                                  tf.TensorSpec(list(wts[14].shape), tf.float32),

                                  tf.TensorSpec([None,3], tf.float32),
                                  tf.TensorSpec([None], tf.int32),
                                  tf.TensorSpec([None], tf.int32),
                                  tf.TensorSpec([None,1], tf.int32),
                                  tf.TensorSpec([None,2], tf.int32),
                                  tf.TensorSpec([None,3], tf.int32),
                                  tf.TensorSpec([None,4], tf.int32),
                                  tf.TensorSpec([None,5], tf.int32),
                                  tf.TensorSpec([], tf.int32),
                                  tf.TensorSpec([], tf.float32),
                                  tf.TensorSpec([], tf.float32),

                                  tf.TensorSpec([None,3], tf.float32),
                                  tf.TensorSpec([], tf.float32),
                                  tf.TensorSpec([], tf.float32),
                                  ])
    def __call__(this,bcf,acf,dcf,coefp,coefw,coefd,coefr,lck1,lcb1,lck2,lcb2,dk1,db1,dk2,db2,xs,ilist,jlist,mlist,tlist,blist,alist,dlist,nmol,rcut,bl,f0,e0,alpha):
        with tf.GradientTape() as gv:
            gv.watch(bcf)
            gv.watch(acf)
            gv.watch(dcf)
            gv.watch(coefp)
            gv.watch(coefw)
            gv.watch(coefd)
            gv.watch(coefr)
            gv.watch(lck1)
            gv.watch(lcb1)
            gv.watch(lck2)
            gv.watch(lcb2)
            gv.watch(dk1)
            gv.watch(db1)
            gv.watch(dk2)
            gv.watch(db2)
            with tf.GradientTape() as g:
                g.watch(xs)
                # compute descriptors
                rs=to_dist_pbc(tf.gather(xs,ilist)-tf.gather(xs,jlist),bl)
                globalcut=tf.maximum(0.,tf.math.sign(rcut-rs))
                
                src2=tf.square(coefp[:,:,1]/rcut)
                src6=src2*src2*src2
                pvc=-2*coefp[:,:,5]/rcut-coefp[:,:,0]*src6*(-7+src2*(9*coefp[:,:,2]+src2*(11*coefp[:,:,3]+src2*13*coefp[:,:,4])))
                pvf=(coefp[:,:,5]/rcut+coefp[:,:,0]*src6*(-6+src2*(coefp[:,:,2]*8+src2*(coefp[:,:,3]*10+src2*coefp[:,:,4]*12))))/rcut
                pf=tf.gather_nd(coefp,tlist)
                sr2=tf.square(pf[:,1]/rs)
                sr6=sr2*sr2*sr2
                erg=.5*tf.reduce_sum( (pf[:,5]/rs +
                            pf[:,0]*sr6*(-1+sr2*(pf[:,2]+sr2*(pf[:,3]+sr2*pf[:,4])))
                            + tf.gather_nd(pvf,tlist) * rs + tf.gather_nd(pvc,tlist)) * globalcut )

                # pf=tf.gather_nd(coefp,tlist)
                # erg=0.5 * tf.reduce_sum(pair_func(rs,pf[:,0],pf[:,1],pf[:,2],pf[:,3],pf[:,4],pf[:,5],rcut))

                descij=desc(tf.expand_dims(rs,-1),tf.gather_nd(coefw,tlist),tf.gather_nd(coefd,tlist),tf.gather_nd(coefr,tlist),rcut)*tf.expand_dims(globalcut,-1)
                descriptor=tf.tensor_scatter_nd_add(tf.zeros([nmol*NUM_ATOM,NUM_DESC],dtype=tf.float32),mlist,descij)

                descriptor=tf.reshape(descriptor,[nmol,NUM_ATOM,NUM_DESC])
                y=sft_sign(K.backend.local_conv1d(descriptor,lck1,[1],[1],data_format="channels_last")+lcb1)
                y=sft_sign(K.backend.local_conv1d(y     ,lck2,[3],[2],data_format="channels_first")+lcb2)
                y=tf.reshape(y,[nmol,-1])
                y=sft_sign(tf.matmul(y,dk1)+db1)
                erg+=tf.reduce_sum(tf.matmul(y,dk2)+db2)         #*0

                bd=to_dist_pbc(tf.gather(xs,blist[:,1])-tf.gather(xs,blist[:,2]),bl)
                bf=tf.gather(bcf,blist[:,0])
                bd=bd-bf[:,0]
                erg+=tf.reduce_sum(tf.square(bd)*(bf[:,1]*tf.square(bd-bf[:,2])+bf[:,3]))      #*0
            
                agc=tf.gather(xs,alist[:,2])
                ag1=pbc(tf.gather(xs,alist[:,1])-agc,bl)
                ag2=pbc(tf.gather(xs,alist[:,3])-agc,bl)
                cq=tf.reduce_sum(ag1*ag2,axis=-1)/tf.sqrt(tf.reduce_sum(tf.square(ag1),axis=-1)*tf.reduce_sum(tf.square(ag2),axis=-1))
                af=tf.gather(acf,alist[:,0])
                cq=cq-af[:,0]
                erg+=tf.reduce_sum(tf.square(cq)*(af[:,1]*tf.square(cq-af[:,2])+af[:,3]))      #*0
            
                dhc1=tf.gather(xs,dlist[:,2])
                dhc2=tf.gather(xs,dlist[:,3])
                dh1=pbc(tf.gather(xs,dlist[:,1])-dhc1,bl)
                dh2=pbc(dhc2-dhc1,bl)
                dh3=pbc(tf.gather(xs,dlist[:,4])-dhc2,bl)
                lgdh2=tf.reduce_sum(tf.square(dh2),axis=-1,keepdims=True)
                dh1=dh1-dh2*tf.reduce_sum(dh1*dh2,axis=-1,keepdims=True)/lgdh2
                dh3=dh3-dh2*tf.reduce_sum(dh3*dh2,axis=-1,keepdims=True)/lgdh2
                cd=tf.reduce_sum(dh1*dh3,axis=-1)/tf.sqrt(tf.reduce_sum(tf.square(dh1),axis=-1)*tf.reduce_sum(tf.square(dh3),axis=-1))
                df=tf.gather(dcf,dlist[:,0])
                erg+=tf.reduce_sum(df[:,0]+cd*(df[:,1]+cd*(df[:,2]+cd*df[:,3])))      #*0

            force=-tf.convert_to_tensor(g.gradient(erg,xs))
            loss=tf.reduce_sum(tf.square(force-f0))+alpha*tf.square(erg-e0)
        return *[tf.convert_to_tensor(grd) for grd in gv.gradient(loss,(bcf,acf,dcf,coefp,coefw,coefd,coefr,lck1,lcb1,lck2,lcb2,dk1,db1,dk2,db2))],loss,erg,force

def load_wts(fname):
    wtt=[]
    with open(fname,"rb") as fp:
        for w in wts:
            wtt.append(np.reshape(struct.unpack('f'*w.size,fp.read(4*w.size)),w.shape))
    return wtt

class MODEL_DEPLOY_ALL(tf.Module):
    def __init__(self,wtfile) -> None:
        super().__init__()
        weights=load_wts(wtfile)
        self.bcf    = tf.Variable(weights[0 ],dtype=tf.float32,trainable=False)
        self.acf    = tf.Variable(weights[1 ],dtype=tf.float32,trainable=False)
        self.dcf    = tf.Variable(weights[2 ],dtype=tf.float32,trainable=False)
        self.coefp  = tf.Variable(weights[3 ],dtype=tf.float32,trainable=False)
        self.coefw  = tf.Variable(weights[4 ],dtype=tf.float32,trainable=False)
        self.coefd  = tf.Variable(weights[5 ],dtype=tf.float32,trainable=False)
        self.coefr  = tf.Variable(weights[6 ],dtype=tf.float32,trainable=False)
        self.lck1   = tf.Variable(weights[7 ],dtype=tf.float32,trainable=False)
        self.lcb1   = tf.Variable(weights[8 ],dtype=tf.float32,trainable=False)
        self.lck2   = tf.Variable(weights[9 ],dtype=tf.float32,trainable=False)
        self.lcb2   = tf.Variable(weights[10],dtype=tf.float32,trainable=False)
        self.dk1    = tf.Variable(weights[11],dtype=tf.float32,trainable=False)
        self.db1    = tf.Variable(weights[12],dtype=tf.float32,trainable=False)
        self.dk2    = tf.Variable(weights[13],dtype=tf.float32,trainable=False)
        self.db2    = tf.Variable(weights[14],dtype=tf.float32,trainable=False)
    @tf.function(input_signature=[
                                  tf.TensorSpec([None,3], tf.float32),
                                  tf.TensorSpec([None], tf.int32),
                                  tf.TensorSpec([None], tf.int32),
                                  tf.TensorSpec([None,1], tf.int32),
                                  tf.TensorSpec([None,2], tf.int32),
                                  tf.TensorSpec([None,3], tf.int32),
                                  tf.TensorSpec([None,4], tf.int32),
                                  tf.TensorSpec([None,5], tf.int32),
                                  tf.TensorSpec([], tf.int32),
                                  tf.TensorSpec([], tf.float32),
                                  tf.TensorSpec([], tf.float32),
                                  ])
    def __call__(this,xs,ilist,jlist,mlist,tlist,blist,alist,dlist,nmol,rcut,bl):
        with tf.GradientTape() as g:
            g.watch(xs)
            # compute descriptors
            rs=to_dist_pbc(tf.gather(xs,ilist)-tf.gather(xs,jlist),bl)
            globalcut=tf.maximum(0.,tf.math.sign(rcut-rs))
            
            src2=tf.square(this.coefp[:,:,1]/rcut)
            src6=src2*src2*src2
            pvc=-2*this.coefp[:,:,5]/rcut-this.coefp[:,:,0]*src6*(-7+src2*(9*this.coefp[:,:,2]+src2*(11*this.coefp[:,:,3]+src2*13*this.coefp[:,:,4])))
            pvf=(this.coefp[:,:,5]/rcut+this.coefp[:,:,0]*src6*(-6+src2*(this.coefp[:,:,2]*8+src2*(this.coefp[:,:,3]*10+src2*this.coefp[:,:,4]*12))))/rcut
            pf=tf.gather_nd(this.coefp,tlist)
            sr2=tf.square(pf[:,1]/rs)
            sr6=sr2*sr2*sr2
            erg=.5*tf.reduce_sum( (pf[:,5]/rs +
                        pf[:,0]*sr6*(-1+sr2*(pf[:,2]+sr2*(pf[:,3]+sr2*pf[:,4])))
                        + tf.gather_nd(pvf,tlist) * rs + tf.gather_nd(pvc,tlist)) * globalcut )
            # pf=tf.gather_nd(coefp,tlist)
            # erg=0.5 * tf.reduce_sum(pair_func(rs,pf[:,0],pf[:,1],pf[:,2],pf[:,3],pf[:,4],pf[:,5],rcut))
            descij=desc(tf.expand_dims(rs,-1),tf.gather_nd(this.coefw,tlist),tf.gather_nd(this.coefd,tlist),tf.gather_nd(this.coefr,tlist),rcut)*tf.expand_dims(globalcut,-1)
            descriptor=tf.tensor_scatter_nd_add(tf.zeros([nmol*NUM_ATOM,NUM_DESC],dtype=tf.float32),mlist,descij)
            descriptor=tf.reshape(descriptor,[nmol,NUM_ATOM,NUM_DESC])
            y=sft_sign(K.backend.local_conv1d(descriptor,this.lck1,[1],[1],data_format="channels_last")+this.lcb1)
            y=sft_sign(K.backend.local_conv1d(y     ,this.lck2,[3],[2],data_format="channels_first")+this.lcb2)
            y=tf.reshape(y,[nmol,-1])
            y=sft_sign(tf.matmul(y,this.dk1)+this.db1)
            erg+=tf.reduce_sum(tf.matmul(y,this.dk2)+this.db2)         #*0
            bd=to_dist_pbc(tf.gather(xs,blist[:,1])-tf.gather(xs,blist[:,2]),bl)
            bf=tf.gather(this.bcf,blist[:,0])
            bd=bd-bf[:,0]
            erg+=tf.reduce_sum(tf.square(bd)*(bf[:,1]*tf.square(bd-bf[:,2])+bf[:,3]))      #*0
        
            agc=tf.gather(xs,alist[:,2])
            ag1=pbc(tf.gather(xs,alist[:,1])-agc,bl)
            ag2=pbc(tf.gather(xs,alist[:,3])-agc,bl)
            cq=tf.reduce_sum(ag1*ag2,axis=-1)/tf.sqrt(tf.reduce_sum(tf.square(ag1),axis=-1)*tf.reduce_sum(tf.square(ag2),axis=-1))
            af=tf.gather(this.acf,alist[:,0])
            cq=cq-af[:,0]
            erg+=tf.reduce_sum(tf.square(cq)*(af[:,1]*tf.square(cq-af[:,2])+af[:,3]))      #*0
        
            dhc1=tf.gather(xs,dlist[:,2])
            dhc2=tf.gather(xs,dlist[:,3])
            dh1=pbc(tf.gather(xs,dlist[:,1])-dhc1,bl)
            dh2=pbc(dhc2-dhc1,bl)
            dh3=pbc(tf.gather(xs,dlist[:,4])-dhc2,bl)
            lgdh2=tf.reduce_sum(tf.square(dh2),axis=-1,keepdims=True)
            dh1=dh1-dh2*tf.reduce_sum(dh1*dh2,axis=-1,keepdims=True)/lgdh2
            dh3=dh3-dh2*tf.reduce_sum(dh3*dh2,axis=-1,keepdims=True)/lgdh2
            cd=tf.reduce_sum(dh1*dh3,axis=-1)/tf.sqrt(tf.reduce_sum(tf.square(dh1),axis=-1)*tf.reduce_sum(tf.square(dh3),axis=-1))
            df=tf.gather(this.dcf,dlist[:,0])
            erg+=tf.reduce_sum(df[:,0]+cd*(df[:,1]+cd*(df[:,2]+cd*df[:,3])))      #*0

        force=-tf.convert_to_tensor(g.gradient(erg,xs))
        return erg,force

if __name__=="__main__":
    # argv: what_to_do model_fname weight_fname
    if sys.argv[1]=="reset":
        mdl_dep.save(sys.argv[2]+"_deploy")

        mdl_trn=TRAINER()
        tf.saved_model.save(mdl_trn,sys.argv[2]+"_train")

        fp = open(sys.argv[3],'wb')
        for w in wts:
            fl=np.reshape(w,[-1])
            fp.write(struct.pack('f'*len(fl), *fl))
            print(w.shape)
        fp.close()

        mdl_dep_all=MODEL_DEPLOY_ALL(sys.argv[3])
        tf.saved_model.save(mdl_dep_all,sys.argv[2]+"_dpl_tf")
    elif sys.argv[1]=="update":
        wtt=load_wts(sys.argv[3])
        wtt[10]=wtt[10].reshape(np.flip(wtt[10].shape))
        mdl_dep.set_weights(wtt[7:])
        mdl_dep.save(sys.argv[2]+"_deploy")

        mdl_dep_all=MODEL_DEPLOY_ALL(sys.argv[3])
        tf.saved_model.save(mdl_dep_all,sys.argv[2]+"_dpl_tf")
    elif sys.argv[1]=="zero":
        with open(sys.argv[3],"rb") as fp:
            data=fp.read()
            data=struct.unpack('f'*(len(data)//4),data)
        data=np.array(data)
        data[-1]-=float(sys.argv[-1])
        with open(sys.argv[3],"wb") as fp:
            fp.write(struct.pack('f'*len(data), *data))
    else:
        raise RuntimeError("unknown command")
