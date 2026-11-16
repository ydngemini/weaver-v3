"""Connected, long (32-frame) synthetic sensor episodes for WEAVER-SOMA.

No physical observations. Future labels are computed by counterfactual rollouts
of the same evolving physical state, not by seeing held-out futures at inference.
"""
from __future__ import annotations
import numpy as np, torch
from physics_simulator import params_for,step,STEP,LIMITS
from unified_model import STATE_CENTER,STATE_SCALE

CENTER=np.array(STATE_CENTER,dtype=np.float32)
SCALE=np.array(STATE_SCALE,dtype=np.float32)
WINDOW=8
SEGMENTS=4
LONG_HORIZON=24

def _param_mix(rng,n,regime):
    p=params_for(rng,n,regime if regime in ('normal','changed','extreme') else 'changed')
    if regime=='thermal_shift':
        p['tau_heat']*=rng.uniform(1.5,2.3,n).astype('float32')
        p['k_heat']*=rng.uniform(.8,1.2,n).astype('float32')
        p['cool']*=rng.uniform(.5,.9,n).astype('float32')
    elif regime=='motor_shift':
        p['tau_motor']*=rng.uniform(1.5,2.1,n).astype('float32')
        p['relief']*=rng.uniform(.65,1.0,n).astype('float32')
    return p

def make_long(n:int,seed:int,regime:str='normal',fault:str='clean'):
    rng=np.random.default_rng(seed)
    pars=_param_mix(rng,n,regime)
    state=np.stack([rng.uniform(18,54,n),rng.uniform(26,39,n),rng.uniform(.28,.72,n),
                    rng.uniform(-1,2,n),rng.uniform(0,.15,n)],axis=1).astype('float32')
    seq=[state.copy()]; acts=[]
    # Actions persist over multiple frames, producing temporally correlated physics.
    cmd=rng.integers(0,3,size=n)
    for t in range(32):
        if t%4==0:cmd=rng.choice(3,size=n,p=(.56,.34,.10))
        acts.append(np.asarray(cmd).copy());state=step(state,pars,cmd);seq.append(state.copy())
    seq=np.stack(seq,axis=1)
    acts=np.stack(acts,axis=1)
    allobs=[];allquality=[];allfuture=[];allrisk=[];alllong=[]
    allcontext=[];allaction=[];allsummary=[];allbody=[]
    allbest=[];allevent=[];allattn=[];allroute=[];allreg=[];allreliability=[]
    group_context=np.zeros((n,8),dtype='float32')
    goal=rng.integers(0,3,n)
    group_context[np.arange(n),goal]=1
    group_context[:,3:]=rng.uniform(0,1,(n,5))
    external=(rng.random(n)<.32).astype('float32')
    for seg in range(SEGMENTS):
        end=(seg+1)*WINDOW
        before=seq[:,end-WINDOW:end,:3]
        after=seq[:,end-WINDOW+1:end+1,:3]
        slope=(after-before)/STEP
        contact=(after[:,:,0]>8).astype('float32')[...,None]
        obs=np.concatenate((after,slope,contact),axis=-1).astype('float32')
        cur=seq[:,end].copy()
        fut=[];longrisk=[]
        for a in range(3):
            z=cur.copy();xs=[];cross=np.zeros(n,dtype=bool)
            for h in range(LONG_HORIZON):
                z=step(z,pars,np.full(n,a,dtype='int64'))
                if h<6:xs.append(z[:,:3].copy())
                cross|=(z[:,:3]>LIMITS).any(axis=1)
            fut.append(np.stack(xs,axis=1))
            longrisk.append(cross.astype('float32'))
        future=np.stack(fut,axis=1).astype('float32')
        risk=(future>LIMITS).any(axis=(2,3)).astype('float32')
        longrisk=np.stack(longrisk,axis=1)
        q=np.ones((n,WINDOW,3),dtype='float32')
        # Hardware-verified dropout can mask any modality. Unflagged silent faults
        # remain intentionally difficult and are not counted as detected.
        if fault in ('flagged','mixed'):
            flagged=rng.random((n,3))<(.33 if fault=='flagged' else .15)
            for channel in range(3):q[flagged[:,channel],:,channel]=0
        if fault in ('noise','mixed'):
            sd=[1.6,.22,.011,2.0,.6,.02]
            obs[:,:,:6]+=rng.normal(0,sd,(n,WINDOW,6)).astype('float32')
        if fault=='silent':
            bad=rng.random(n)<.35
            badch=rng.integers(0,3,n)
            for i in np.where(bad)[0]:
                ch=badch[i];obs[i,:,ch]=obs[i,0,ch]
                obs[i,:,ch+3]=0
        if fault in ('flagged','mixed'):
            # Invalid readings may contain arbitrary adversarial garbage; MUST be masked.
            invalid=q<.5
            for ch in range(3):
                obs[:,:,ch]=np.where(invalid[:,:,ch],rng.uniform(-100,150,(n,WINDOW)),obs[:,:,ch])
                obs[:,:,ch+3]=np.where(invalid[:,:,ch],rng.uniform(-80,80,(n,WINDOW)),obs[:,:,ch+3])
        episode_summary=np.stack((after[:,:,0].max(1),after[:,:,1].mean(1),after[:,:,2].max(1)),axis=1)
        summary=(episode_summary-CENTER)/SCALE
        body=(after[:,-1]-CENTER)/SCALE
        costs=np.array([0.,.055,.14],dtype='float32')
        exceed=np.maximum(future-LIMITS,0)/np.array([20.,8.,.24],dtype='float32')
        severity=exceed.max(axis=(2,3))
        best=(3*risk+.075*severity+costs).argmin(axis=1).astype('int64')
        current_risk=((after[:,-1,0]>59)|(after[:,-1,1]>41)|(after[:,-1,2]>.76)|(q.min(axis=(1,2))<1))
        attention=np.where(current_risk,0,np.where(external>.5,1,np.where(group_context[:,6]>.65,2,np.where(group_context[:,4]>.65,3,4)))).astype('int64')
        route=np.where(current_risk,0,np.where(group_context[:,3]>.65,1,np.where(group_context[:,4]>.7,3,2))).astype('int64')
        regulation=np.stack((group_context[:,4],1-risk.mean(axis=1),current_risk.astype('float32'),
                    group_context[:,6],risk.mean(axis=1),group_context[:,3],
                    1-np.clip((after[:,-1,2]-.3)/.6,0,1)),axis=-1).astype('float32')
        allobs.append(obs);allquality.append(q);allfuture.append((future-CENTER)/SCALE)
        allrisk.append(risk);alllong.append(longrisk)
        allcontext.append(group_context.copy());allaction.append(np.eye(3,dtype='float32')[acts[:,end-1]])
        allsummary.append(summary);allbody.append(body);allbest.append(best);allevent.append(external)
        allattn.append(attention);allroute.append(route);allreg.append(regulation)
        allreliability.append(q[:,-1,:].mean(axis=-1))
    group={'obs':allobs,'quality':allquality,'future':allfuture,'risk':allrisk,
           'risk_long':alllong,'context':allcontext,'executed_action':allaction,
           'summary':allsummary,'body':allbody,'best_action':allbest,'external':allevent,
           'attention':allattn,'route':allroute,'regulation':allreg,
           'reliability':allreliability}
    return {key:torch.from_numpy(np.stack(values,axis=1).copy()) for key,values in group.items()}

def slice_group(data,idx):return {k:v[idx] for k,v in data.items()}

def at(data,window_idx):return {k:v[:,window_idx] for k,v in data.items() if k!='risk_long'}