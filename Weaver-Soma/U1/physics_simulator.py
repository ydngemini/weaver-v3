"""Vectorized, dimensioned *synthetic* thermal/contact robot simulator.
Not validated hardware physics. Physical idea: action actuator lag, pressure and load balance,
first-order thermal flux lag, Newton-style cooling. All parameters are chosen scenarios.
"""
import numpy as np

LIMITS=np.array([62.,43.,.82],dtype=np.float32)
SCALES=np.array([20.,8.,.24],dtype=np.float32)
STEP=0.12
HORIZON=6
HISTORY=8
ACTIONS=('hold','ease','withdraw')

def params_for(rng,n,mode):
    p={}
    def u(lo,hi):return rng.uniform(lo,hi,n).astype(np.float32)
    if mode=='normal':
        p['source']=u(3,17);p['heater']=u(43,82);p['k_heat']=u(.065,.23)
        p['tau_heat']=u(.12,.48);p['tau_motor']=u(.12,.38);p['relief']=u(18,33)
        p['joint_drive']=u(.02,.13);p['cool']=u(.03,.12)
    elif mode=='changed':
        p['source']=u(2,22);p['heater']=u(43,94);p['k_heat']=u(.05,.34)
        p['tau_heat']=u(.24,.8);p['tau_motor']=u(.27,.7);p['relief']=u(10,26)
        p['joint_drive']=u(.0,.16);p['cool']=u(.025,.10)
    elif mode=='extreme':
        p['source']=u(6,27);p['heater']=u(70,112);p['k_heat']=u(.15,.42)
        p['tau_heat']=u(.42,.95);p['tau_motor']=u(.48,.95);p['relief']=u(6,18)
        p['joint_drive']=u(.04,.22);p['cool']=u(.015,.075)
    else:raise ValueError(mode)
    return p

def step(state, par, act):
    """state: [P_kPa, T_C, joint_fraction, heat_flow_C_per_s, withdrawal_state]"""
    z=state.copy();press,temp,joint,flux,withdraw=np.moveaxis(z,1,0)
    cmd=np.asarray(act,dtype=np.float32) # 0=hold; 1=ease; 2=withdraw
    target=np.where(cmd<.5,0.,np.where(cmd<1.5,.42,1.))
    withdraw=np.clip(withdraw+STEP*(target-withdraw)/par['tau_motor'],0,1)
    contact=np.maximum(0,1-withdraw)
    dp=par['source']*(.6+.4*contact)-.07*press-par['relief']*withdraw
    dj=par['joint_drive']-.03*(joint-.3)-.32*withdraw
    flux_target=par['k_heat']*contact*(par['heater']-temp)
    flux=flux+STEP*(flux_target-flux)/par['tau_heat']
    temp=temp+STEP*(flux-par['cool']*(temp-24.))
    press=press+STEP*dp
    joint=joint+STEP*dj
    z[:,0]=np.clip(press,0,100)
    z[:,1]=np.clip(temp,16,70)
    z[:,2]=np.clip(joint,.05,1.2)
    z[:,3]=np.clip(flux,-18,40)
    z[:,4]=withdraw
    return z

def make(n=1000,seed=19,mode='normal'):
    rng=np.random.default_rng(seed)
    par=params_for(rng,n,mode)
    state=np.stack([rng.uniform(19,58,n),rng.uniform(27,39,n),rng.uniform(.25,.71,n),
                    rng.uniform(-1,3,n),rng.uniform(0,.15,n)],axis=1).astype(np.float32)
    seq=[]
    for k in range(HISTORY+1):
        seq.append(state.copy())
        history_action=rng.choice(3,size=n,p=[.76,.23,.01])
        state=step(state,par,history_action)
    seq=np.stack(seq,axis=1)
    x=seq[:,1:,0:3]
    slopes=(seq[:,1:,0:3]-seq[:,:-1,0:3])/STEP
    contact=(seq[:,1:,0]>8).astype(np.float32)[...,None]
    observations=np.concatenate([x,slopes,contact],axis=-1).astype(np.float32)
    current=seq[:,-1].copy()
    futures=[]
    for a in range(3):
        s=current.copy();xs=[]
        for h in range(HORIZON):
            s=step(s,par,np.full(n,a))
            xs.append(s[:,:3].copy())
        futures.append(np.stack(xs,axis=1))
    future=np.stack(futures,axis=1).astype(np.float32) # (n,3,6,3)
    violate=(future>LIMITS).any(axis=(2,3)).astype(np.float32)
    return observations, future, violate

if __name__=='__main__':
    for m in ('normal','changed','extreme'):
        x,y,z=make(10000,seed=100,mode=m)
        print(m,x.shape,'unsafe',np.round(z.mean(axis=0),3),'oracle floor',z.min(axis=1).mean(),'avoidable',((z.sum(axis=1)>0)&(z.min(axis=1)==0)).mean())
