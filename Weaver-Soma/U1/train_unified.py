"""Reproducible joint training with independent simulated scenario splits.
Pure CPU synthetic data. No live user data, internet, or motor output.
"""
from pathlib import Path
import argparse,csv,json,time,sys,copy
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
from physics_simulator import make,LIMITS
from unified_model import WeaverSOMAUnified,joint_objective,STATE_CENTER,STATE_SCALE

OUT=Path(__file__).resolve().parent / 'results'
H=8

def make_data(n,seed,mode='normal',corrupt=True):
    obs,future,risk=make(n,seed,mode)
    rng=np.random.default_rng(seed+123456)
    context=np.zeros((n,8),dtype=np.float32)
    goal=rng.integers(0,3,n)
    context[np.arange(n),goal]=1.
    context[:,3]=rng.uniform(0,1,n) # speech presence / priority
    context[:,4]=rng.uniform(0,1,n) # attention novelty cue, *not* hidden hazard
    context[:,5]=rng.uniform(0,1,n) # simulated energy request
    context[:,6]=rng.uniform(0,1,n) # social task priority
    context[:,7]=rng.uniform(0,1,n) # compute pressure
    executed=rng.integers(0,3,n)
    action=np.eye(3,dtype=np.float32)[executed]
    quality=np.ones((n,H,3),dtype=np.float32)
    fail=rng.random((n,3))<.12
    for j in range(3):quality[np.where(fail[:,j])[0],:,j]=0.

    # Mechanical command effect, separate from a randomized outside disturbance.
    # The original simulator's action history is not available; this additional
    # synthetic measurement perturbation is not a physically validated controller.
    observed=obs.copy()
    cmd=executed.astype('float32')
    nudge=np.stack([-.75*cmd,-.08*cmd,-.03*cmd],axis=-1)
    observed[:,-1,:3]+=nudge
    observed[:,-1,3:6]+=nudge/.12
    external=(rng.random(n)<.34).astype(np.float32)
    idx=rng.integers(0,3,n)
    impulse=np.zeros((n,3),dtype=np.float32)
    sign=rng.choice([-1.,1.],size=n)
    amplitude=np.stack([rng.uniform(1.,7.,n),rng.uniform(.3,2.5,n),rng.uniform(.015,.11,n)],axis=-1)
    impulse[np.arange(n),idx]=amplitude[np.arange(n),idx]*sign*external
    observed[:,-1,:3]+=impulse
    observed[:,-1,3:6]+=impulse/.12
    # A missing sensor has no readable value; add adversarial arbitrary samples
    # in testing to verify the network masks these before any learned transform.
    observed[...,:3]=np.clip(observed[...,:3],[-5,5,-1],[110,80,1.4])

    # Preference/efficiency labels: choose safe action with minimal disruption;
    # if all actions violate, choose least integrated margin over threshold.
    exceed=np.maximum(future-LIMITS,0)/np.array([20.,8.,.24],dtype=np.float32)
    severity=exceed.max(axis=(2,3))
    costs=np.array([0.,.055,.14],dtype=np.float32)
    policy_score=3*risk+.075*severity+costs[None,:]
    best=policy_score.argmin(axis=1).astype(np.int64)
    # Recurrent history label as a compact episodic trace.
    hist=obs[:,:,:3]
    summary=np.stack([hist[:,:,0].max(axis=1),hist[:,:,1].mean(axis=1),hist[:,:,2].max(axis=1)],axis=-1)
    summary=(summary-np.array(STATE_CENTER,dtype='float32'))/np.array(STATE_SCALE,dtype='float32')
    body=(obs[:,-1,:3]-np.array(STATE_CENTER,dtype='float32'))/np.array(STATE_SCALE,dtype='float32')
    current_risk=((obs[:,-1,0]>59)|(obs[:,-1,1]>41)|(obs[:,-1,2]>.76)|fail.any(axis=1))
    # attention labels: safety / external / speech / memory / goal
    attention=np.where(current_risk,0,
        np.where(external>.5,1,np.where(context[:,6]>.65,2,np.where(context[:,4]>.65,3,4)))).astype('int64')
    # route advice: embodiment, realtime speech, interactive, background
    route=np.where(current_risk,0,np.where(context[:,3]>.65,1,np.where(context[:,4]>.7,3,2))).astype('int64')
    stability=(1-np.clip((obs[:,-1,2]-.3)/.6,0,1))
    hazard=risk.mean(axis=1)
    regulate=np.stack([context[:,4],1-hazard, current_risk.astype(np.float32),
                 context[:,6],hazard,context[:,3],stability],axis=-1).astype(np.float32)
    data={'obs':observed,'quality':quality,'context':context,'executed_action':action,
          'risk':risk,'future':(future-np.array(STATE_CENTER,dtype='float32'))/np.array(STATE_SCALE,dtype='float32'),
          'best_action':best,'external':external,'summary':summary,'body':body,
          'attention':attention,'route':route,'regulation':regulate,
          'reliability':quality[:,-1].mean(axis=1)}
    tensors={k:torch.from_numpy(v.copy()) for k,v in data.items()}
    return tensors


def concat_data(*datasets):
    return {key:torch.cat([d[key] for d in datasets],dim=0) for key in datasets[0]}


def forward_model(model,b):
    return model(b['obs'],b['quality'],b['context'],b['executed_action'])


def score(model,data,mode):
    model.eval()
    res={k:[] for k in ('risk_logits','policy_logits','external_logits','workspace_logits','compute_logits','episodic_summary','body_readout','regulation','reliability')}
    with torch.no_grad():
        for start in range(0,len(data['obs']),512):
            b={k:v[start:start+512] for k,v in data.items()}
            outputs=forward_model(model,b)
            for key in res:res[key].append(outputs[key].cpu())
    for k in res:res[k]=torch.cat(res[k]).numpy()
    risk_pred=torch.from_numpy(res['risk_logits']).sigmoid().numpy()
    y=data['risk'].numpy()
    ap=float(average_precision_score(y.flatten(),risk_pred.flatten()))
    chosen=np.argmax(res['policy_logits'],axis=1)
    idx=np.arange(len(chosen))
    oracle=y.min(axis=1)
    selected=y[idx,chosen]
    unavoidable=oracle.mean()*100
    avoidable=((selected>oracle)).mean()*100
    return {'mode':mode,'n_episodes':len(chosen),'risk_ap':ap,
         'policy_agreement':float(np.mean(chosen==data['best_action'].numpy())),
         'violation_pct':float(100*selected.mean()),'unavoidable_floor_pct':float(unavoidable),
         'avoidable_pct':float(avoidable),'withdraw_pct':float(100*np.mean(chosen==2)),
         'event_ap':float(average_precision_score(data['external'].numpy(),torch.sigmoid(torch.from_numpy(res['external_logits'])).numpy())),
         'workspace_accuracy':float(np.mean(res['workspace_logits'].argmax(1)==data['attention'].numpy())),
         'routing_accuracy':float(np.mean(res['compute_logits'].argmax(1)==data['route'].numpy())),
         'episodic_mae':float(np.mean(np.abs(res['episodic_summary']-data['summary'].numpy()))),
         'regulation_mae':float(np.mean(np.abs(res['regulation']-data['regulation'].numpy())))}


def invariance_check(model):
    b=make_data(64,98123,'normal');model.eval()
    # quality is independently declared; values on a missing channel are unobservable.
    b['quality'][:,:,1]=0
    b2={k:v.clone() for k,v in b.items()}
    b2['obs'][:,:,1]=999999
    b2['obs'][:,:,4]=-999999
    with torch.no_grad():
        a=forward_model(model,b)
        o=forward_model(model,b2)
    fields=['risk_logits','policy_logits','external_logits','workspace_logits','compute_logits','episodic_summary','regulation','state']
    diff=max(float((a[k]-o[k]).abs().max().item()) for k in fields)
    assert diff<=1e-6,diff
    return diff


def train(output=OUT,max_epochs=28,train_n=7800,val_n=1900):
    torch.set_num_threads(2)
    torch.manual_seed(73);np.random.seed(73)
    output.mkdir(parents=True,exist_ok=True)
    training=concat_data(make_data(train_n//2,110,'normal'),make_data(train_n-train_n//2,111,'changed'))
    validation=concat_data(make_data(val_n//2,210,'normal'),make_data(val_n-val_n//2,211,'changed'))
    model=WeaverSOMAUnified()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0003)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max_epochs,eta_min=.00025)
    best=float('inf');best_state=None;wait=0;history=[];best_epoch=0
    tick=time.perf_counter()
    for ep in range(1,max_epochs+1):
        model.train()
        rnd=torch.randperm(len(training['obs']))
        tot=0;cnt=0
        for idx in rnd.split(256):
            batch={k:v[idx] for k,v in training.items()}
            out=forward_model(model,batch)
            loss,_=joint_objective(out,batch)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.5)
            optimizer.step()
            tot+=loss.item()*len(idx);cnt+=len(idx)
        scheduler.step()
        model.eval();vloss=0
        with torch.no_grad():
            for ix in range(0,len(validation['obs']),512):
                v={k:t[ix:ix+512] for k,t in validation.items()}
                loss,_=joint_objective(forward_model(model,v),v)
                vloss+=loss.item()*len(v['obs'])
        vloss/=len(validation['obs'])
        history.append({'epoch':ep,'train_loss':tot/cnt,'validation_loss':vloss})
        if vloss<best-0.001:
            best=vloss;best_epoch=ep;wait=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:wait+=1
        if ep%4==0 or ep==1:print(f'epoch={ep:02d} train={tot/cnt:.4f} val={vloss:.4f} time={time.perf_counter()-tick:.1f}s',flush=True)
        if ep>=12 and wait>=7:break
    model.load_state_dict(best_state);model.eval()
    torch.save({'architecture':'WeaverSOMAUnified-v1','weights':best_state,'best_epoch':best_epoch,
                'train_seed':73,'train_modes':['normal','changed'],'note':'synthetic-only, no real device safety'},output/'weaver_soma_unified.pt')
    tests={mode:make_data(2200,10000+i,mode) for i,mode in enumerate(('normal','changed','extreme'))}
    scores=[score(model,tests[m],m) for m in tests]
    # Causal ablation probes: temporarily zero the contextual inputs and measure prediction/action changes.
    # These are probes, NOT causal proof of subjective consciousness.
    ablations=[]
    for mode,b in tests.items():
        with torch.no_grad():
            sub={k:v.clone() for k,v in b.items()}
            sub['context'].zero_()
        r=score(model,sub,mode+'_no_context')
        ablations.append(r)
    mask_diff=invariance_check(model)
    with (output/'train_curve.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    for filename,rows in [('test_metrics.csv',scores),('context_ablation.csv',ablations)]:
        with (output/filename).open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    meta={'best_epoch':best_epoch,'epochs_ran':ep,'total_parameters':sum(p.numel() for p in model.parameters()),
          'wall_seconds':round(time.perf_counter()-tick,2),'validation_loss':best,
          'missing_sensor_invariance_max_abs_diff':mask_diff,'train_count':len(training['obs']),
          'validation_count':len(validation['obs']),'each_test_count':2200,'test_metrics':scores,
          'ablation_metrics':ablations}
    (output/'manifest.json').write_text(json.dumps(meta,indent=2))
    print('RESULT',json.dumps(meta,indent=2),flush=True)
    return meta

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=28);p.add_argument('--train-n',type=int,default=7800);p.add_argument('--val-n',type=int,default=1900);p.add_argument('--out',default=str(OUT));args=p.parse_args()
    train(Path(args.out),args.epochs,args.train_n,args.val_n)
