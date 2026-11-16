"""WEAVER-SOMA U2 Stage 3: CPU group-robust training and pre-registered OOD holdouts.
All data synthetic. No motor output, crypto, or deployment.
Requires the separately recovered, SHA256-verified U1 checkpoint.
"""
from pathlib import Path
import sys, json, hashlib, time, copy
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'U1')]
from weaver_soma_u2 import WeaverSOMA100M,count_parameters
from unified_model import WeaverSOMAUnified,joint_objective
from train_unified import make_data,concat_data,forward_model
CP=HERE.parent/'U1'/'results'/'weaver_soma_unified.pt'
SHA='3eb0746806f3842ec2d8121a7bcf30f5351b1b26b70e2c53debdac2ea5030c8f'
OUT=HERE.parent/'results'
OUT.mkdir(exist_ok=True)

def augment(raw,seed,kind):
    rng=np.random.default_rng(seed)
    d={k:v.clone() for k,v in raw.items()}
    if kind=='clean':return d
    if kind=='noise':
        # Observational drift only; underlying physical future stays fixed.
        # Nominal channel noise is illustrative, not hardware calibrated.
        for j,sd in enumerate([2.,.32,.016,3.,1.0,.03]):
            d['obs'][:,:,j]+=torch.from_numpy(rng.normal(0,sd,d['obs'][:,:,j].shape).astype('float32'))
    elif kind=='flagged':
        # Independently flagged sensor dropout. Both readings and derivatives are masked in U1.
        j=rng.integers(0,3,len(d['obs']))
        for i,ch in enumerate(j):
            d['quality'][i,:,ch]=0.
            d['obs'][i,:,ch]=float(rng.uniform(-1000,1000))
            d['obs'][i,:,ch+3]=float(rng.uniform(-1000,1000))
        d['reliability']=d['quality'][:,-1,:].mean(dim=-1)
    elif kind=='silent':
        # Plausible stuck-at sensor fault, unflagged. Independent downstream
        # fault detection is essential; learned model cannot know this directly.
        j=rng.integers(0,3,len(d['obs']))
        vals=[38.,30.,.42]
        for i,ch in enumerate(j):
            d['obs'][i,:,ch]=vals[ch];d['obs'][i,:,ch+3]=0.
    else:raise ValueError(kind)
    # Labels for body readout and episodic history deliberately remain the true,
    # latent physical state. The model must learn under measurement distortion.
    return d

def get_predictions(model,d,batch_size=256):
    model.eval()
    preds={k:[] for k in ('risk_logits','future','policy_logits','body_readout','episodic_summary')}
    total_loss=0.
    with torch.inference_mode():
        for start in range(0,len(d['obs']),batch_size):
            b={k:v[start:start+batch_size] for k,v in d.items()}
            o=forward_model(model,b)
            loss,_=joint_objective(o,b);total_loss+=float(loss)*len(b['obs'])
            for k in preds:preds[k].append(o[k].cpu())
    return {k:torch.cat(v).numpy() for k,v in preds.items()},total_loss/len(d['obs'])

def ece(probs,labels,bins=10):
    p=probs.ravel();y=labels.ravel()
    edges=np.linspace(0,1,bins+1);result=0.
    for i in range(bins):
        inside=(p>=edges[i])&(p<(edges[i+1] if i<bins-1 else 1.00001))
        if inside.any():result+=float(inside.mean()*abs(y[inside].mean()-p[inside].mean()))
    return result

def evaluate(model,d,tag,kind):
    out,loss=get_predictions(model,d)
    probs=1/(1+np.exp(-np.clip(out['risk_logits'],-40,40)))
    y=d['risk'].numpy()
    cost=np.array([0.,.2,1.],dtype=np.float32)
    policy=(probs+0.02*cost).argmin(1)
    outcome=y[np.arange(len(policy)),policy]
    floor=y.min(1)
    scores={'scenario':tag,'sensor_condition':kind,'n':len(y),'joint_loss':loss,
        'risk_ap':float(average_precision_score(y.ravel(),probs.ravel())),
        'brier':float(np.mean((probs-y)**2)),'ece_10_bins':ece(probs,y),
        'avoidable_pct':float(np.mean(outcome>floor)*100),
        'total_violations_pct':float(outcome.mean()*100),
        'unavoidable_floor_pct':float(floor.mean()*100),
        'withdraw_pct':float(np.mean(policy==2)*100),
        'body_mae':float(np.mean(np.abs(out['body_readout']-d['body'].numpy()))),
        'future_mae':float(np.mean(np.abs(out['future']-d['future'].numpy())))}
    return scores

def load():
    assert hashlib.sha256(CP.read_bytes()).hexdigest()==SHA,'missing/wrong recovered weights'
    cp=torch.load(CP,map_location='cpu',weights_only=True)
    original=WeaverSOMAUnified().eval();original.load_state_dict(cp['weights'],strict=True)
    expanded=WeaverSOMA100M();assert expanded.load_u1_weights(str(CP))==91
    assert count_parameters(expanded)==100206197
    expanded.deep_cognitive_core.train_mode='adapter'
    for name,p in expanded.named_parameters():
        p.requires_grad_(name.startswith('deep_cognitive_core.adapter'))
    return original,expanded

def groups(seed,train):
    n=512 if train else 256
    a=make_data(n,seed,'normal')
    b=make_data(n,seed+1,'changed')
    return [augment(a,seed+2,'clean'),augment(b,seed+3,'clean'),
            augment(a,seed+4,'noise'),augment(b,seed+5,'noise'),
            augment(a,seed+6,'flagged'),augment(b,seed+7,'flagged')]

def run():
    start=time.time();torch.manual_seed(319);np.random.seed(319);torch.set_num_threads(2)
    old,model=load()
    tr=groups(12000,True)
    dev=groups(22000,False)
    test={
        'normal':make_data(1250,33001,'normal'),
        'changed':make_data(1250,33002,'changed'),
        'extreme':make_data(1250,33003,'extreme'),
    }
    # The independent test datasets are not touched until checkpoint selection.
    initial=[evaluate(old,x,f'dev_{i}','train_distribution') for i,x in enumerate(dev)]
    # Joint objective + risk calibration + teacher anchoring; optimize worst-group
    # performance on training batches to discourage overfitting to easy regimes.
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.00065)
    best_worst=max(x['joint_loss'] for x in initial)
    best_state=None;best_step=0;best_mean=float(np.mean([x['joint_loss'] for x in initial]))
    history=[]
    for step in range(1,241):
        model.train()
        per_group=[]
        for gi,data in enumerate(tr):
            ix=torch.randint(len(data['obs']),(24,))
            b={k:v[ix] for k,v in data.items()}
            out=forward_model(model,b)
            risk=F.binary_cross_entropy_with_logits(out['risk_logits'],b['risk'])
            # More balanced hazard ranking and future forecast consistency
            fut=F.smooth_l1_loss(out['future'],b['future'],beta=.3)
            body=F.smooth_l1_loss(out['body_readout'],b['body'],beta=.3)
            with torch.no_grad():teacher=forward_model(old,b)
            distill=F.mse_loss(out['risk_logits'],teacher['risk_logits'])
            # Combined, including U1 original objective to avoid only optimizing risk.
            base,_=joint_objective(out,b)
            per_group.append(base+.25*risk+.12*fut+.06*body+.12*distill)
        g=torch.stack(per_group)
        # Soft worst-group optimization + mean to retain overall capability.
        loss=.55*g.max()+.45*g.mean()
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
        opt.step()
        if step in (1,10,25,50,75,100,150,200,240):
            model.eval()
            dd=[evaluate(model,x,f'dev_{i}','train_distribution') for i,x in enumerate(dev)]
            worst=max(x['joint_loss'] for x in dd)
            mean=float(np.mean([x['joint_loss'] for x in dd]))
            history.append({'step':step,'worst_dev_loss':worst,'mean_dev_loss':mean,
                            'worst_dev_brier':max(x['brier'] for x in dd)})
            # Gate: improve worst and not worsen average by >.002, otherwise
            # keep exact U1 at best_step=0. No test labels used in selection.
            if worst < best_worst-1e-5 and mean < best_mean+.002:
                best_worst=worst;best_step=step
                best_state={k:v.detach().cpu().clone() for k,v in model.deep_cognitive_core.adapter.state_dict().items()}
                best_gate=model.deep_cognitive_core.adapter_gain.detach().cpu().clone()
            print(f'step={step} worst_dev={worst:.6f} mean_dev={mean:.6f} best={best_step}',flush=True)
    if best_step:
        model.deep_cognitive_core.adapter.load_state_dict(best_state)
        with torch.no_grad():model.deep_cognitive_core.adapter_gain.copy_(best_gate)
    else:
        with torch.no_grad():model.deep_cognitive_core.adapter_gain.zero_()
    # At this point, selection is frozen. Evaluate on fresh heldout seeds/modes.
    evaluations=[]
    for label,raw in test.items():
        for condition in ('clean','noise','flagged','silent'):
            d=augment(raw,37100+len(evaluations),condition)
            evaluations.append({'baseline':evaluate(old,d,label,condition),
                                'candidate':evaluate(model,d,label,condition)})
    dev_selected=[evaluate(model,x,f'dev_{i}','train_distribution') for i,x in enumerate(dev)]
    report={'experiment':'SOMA U2 Stage 3 CPU robust adapter with worst-group dev gate',
      'architecture':'original 100206197-parameter U2; frozen 100M core, trainable 2129-param adapter',
      'original_checkpoint_sha256':SHA,'train_episodes_per_group':512,
      'train_groups':6,'dev_episodes_per_group':256,'dev_groups':6,
      'heldout_episodes_per_mode':1250,'update_steps':240,
      'best_step':best_step,'baseline_dev_worst_joint_loss':max(x['joint_loss'] for x in initial),
      'selected_dev_worst_joint_loss':max(x['joint_loss'] for x in dev_selected),
      'baseline_dev_mean_joint_loss':float(np.mean([x['joint_loss'] for x in initial])),
      'selected_dev_mean_joint_loss':float(np.mean([x['joint_loss'] for x in dev_selected])),
      'training_curve':history,'independent_test':evaluations,'elapsed_sec':time.time()-start,
      'note':'Synthetic numerical experiments only, no deployment/physical safety/subjective consciousness claim'}
    (OUT/'stage3_results.json').write_text(json.dumps(report,indent=2))
    if best_step:
        torch.save({'version':'SOMA-U2-stage3-experimental-adapter',
         'u1_sha256':SHA,'best_step':best_step,'adapter_weights':best_state,
         'adapter_gain':best_gate},OUT/'stage3_adapter_experimental.pt')
    print('SUMMARY',json.dumps({k:v for k,v in report.items() if k not in ('training_curve','independent_test')},indent=2),flush=True)
    for t in evaluations:
        b=t['baseline'];c=t['candidate']
        print(f"{b['scenario']}:{b['sensor_condition']} U1 loss={b['joint_loss']:.4f} AP={b['risk_ap']:.4f} viol={b['avoidable_pct']:.2f}% | U2 loss={c['joint_loss']:.4f} AP={c['risk_ap']:.4f} viol={c['avoidable_pct']:.2f}%",flush=True)
if __name__=='__main__':run()

