"""Stage 5 WEAVER-SOMA: 100M-core CPU continuation on sequential sensor episodes.

Validates provenance, never consumes fake hardware data, never actuates motors.
The 100M core is trainable; prior-window BPTT is truncated for CPU memory.
This is research CONTINUATION, not foundation-model full-scale pretraining.
"""
from __future__ import annotations
import argparse,hashlib,json,time,random,sys,csv
from pathlib import Path
import torch,numpy as np
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
sys.path.append(str(Path(__file__).resolve().parent.parent / "U1"))
from temporal_model import StreamingWorldModel
from long_world import make_long,slice_group,at
from unified_model import joint_objective

BASE=Path(__file__).resolve().parent.parent
OUT=BASE/'results';OUT.mkdir(parents=True,exist_ok=True)
STAGE4=BASE/'U2'/'results'/'WEAVER_SOMA_U2_100M_Stage4_FULL_MODEL.pt'

def load_checkpoint(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('architecture')!='WeaverSOMA100M-stage4' or checkpoint.get('total_parameters')!=100206197:
        raise ValueError('Expected actual trained WeaverSOMA U2 Stage4 checkpoint')
    if checkpoint.get('synthetic_only') is not True or not checkpoint.get('no_motor_authority'):
        raise ValueError('Unexpected checkpoint provenance')
    return checkpoint

def build(n,seed,with_shift=False):
    regimes=['normal','changed'] if not with_shift else ['extreme','thermal_shift','motor_shift']
    conditions=('clean','noise','flagged','silent')
    return {f'{regime}/{fault}':make_long(n,seed+2713*i,regime,fault)
            for i,(regime,fault) in enumerate((regime,fault) for regime in regimes for fault in conditions)}

def score(model,data):
    model.eval();ys=[];ps=[];bl=[];long_y=[];long_p=[]
    with torch.inference_mode():
        loss_sum=0.;n=len(data['obs'])
        for i in range(0,n,16):
            idx=slice(i,i+16);b=slice_group(data,idx)
            o=model(b['obs'],b['quality'],b['context'],b['executed_action'])
            last=at(b,3)
            joint,_=joint_objective(o['last'],last)
            y=b['risk_long'][:,-1]
            bce=F.binary_cross_entropy_with_logits(o['long_hazard_logits'],y)
            body=F.smooth_l1_loss(o['memory_body'],last['body'])
            loss=joint+.40*bce+.06*body
            loss_sum+=float(loss)*len(b['obs'])
            ys.append(last['risk'].numpy());ps.append(o['last']['risk_logits'].sigmoid().numpy())
            long_y.append(y.numpy());long_p.append(o['long_hazard_logits'].sigmoid().numpy())
        y=np.concatenate(ys);p=np.concatenate(ps)
        ly=np.concatenate(long_y);lp=np.concatenate(long_p)
        decisions=(p+.02*np.array([0.,.2,1.],dtype='float32')).argmin(axis=1)
        selected=y[np.arange(len(y)),decisions]; oracle=y.min(axis=1)
        return {'loss':round(loss_sum/n,6),
                'long_brier':round(float(np.mean((ly-lp)**2)),6),
                'long_ap':round(float(average_precision_score(ly.ravel(),lp.ravel())),5),
                'long_prevalence':round(float(ly.mean()),5),
                'short_risk_ap':round(float(average_precision_score(y.ravel(),p.ravel())),5),
                'short_avoidable_pct':round(float((selected>oracle).mean()*100),4),
                'short_violations_pct':round(float(selected.mean()*100),4),
                'withdrawal_pct':round(float((decisions==2).mean()*100),4),
                'sample_n':n}

def evaluate(model,groups):return {k:score(model,v) for k,v in groups.items()}
def aggregate(scores):
    return {'avg_loss':round(float(np.mean([x['loss'] for x in scores.values()])),6),
            'worst_loss':round(float(max(x['loss'] for x in scores.values())),6),
            'avg_long_brier':round(float(np.mean([x['long_brier'] for x in scores.values()])),6)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=80)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--lr',type=float,default=8e-6)
    p.add_argument('--checkpoint',default=str(STAGE4))
    p.add_argument('--threads',type=int,default=3)
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    seed=20261116;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    now=time.monotonic()
    cp=load_checkpoint(args.checkpoint)
    ckpt_sha=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    m=StreamingWorldModel();m.core.load_state_dict(cp['weights'],strict=True)
    m.core.deep_cognitive_core.train_mode='deep'
    # Freeze the original 121k U1 tensors; optimize 100M core and temporal memory.
    for name,param in m.named_parameters():
        param.requires_grad_(name.startswith('core.deep_cognitive_core.') and
                             not name.startswith('core.deep_cognitive_core.adapter')
                             or name.startswith('temporal_cell.')
                             or name.startswith('long_hazard.')
                             or name.startswith('future_body.'))
    trainable=sum(p.numel() for p in m.parameters() if p.requires_grad)
    deep_count=sum(p.numel() for name,p in m.named_parameters() if p.requires_grad and name.startswith('core.deep_cognitive_core'))
    print(json.dumps({'stage':'5','core_total':sum(p.numel() for p in m.core.parameters()),
                      'actually_trainable':trainable,'deep_trainable':deep_count,
                      'steps':args.steps,'stage4_sha256':ckpt_sha}),flush=True)
    train=build(160,10000)
    dev=build(56,20000)
    holdout=build(100,30000,with_shift=True)
    initial=evaluate(m,dev)
    best_score=aggregate(initial)['worst_loss']+.5*aggregate(initial)['avg_loss']
    best_step=0;history=[{'step':0,'dev':aggregate(initial)}]
    # Holdout isn't consulted until checkpoint selection is frozen.
    optimizer=torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),lr=args.lr,weight_decay=0.00005,foreach=False)
    train_keys=list(train)
    for step in range(1,args.steps+1):
        m.train()
        key=train_keys[(step-1)%len(train_keys)]
        dataset=train[key]
        selection=torch.randint(0,len(dataset['obs']),(args.batch_size,))
        b=slice_group(dataset,selection)
        outputs=m(b['obs'],b['quality'],b['context'],b['executed_action'])
        last=at(b,3)
        loss,_=joint_objective(outputs['last'],last)
        long=F.binary_cross_entropy_with_logits(outputs['long_hazard_logits'],b['risk_long'][:,-1])
        mem=F.smooth_l1_loss(outputs['memory_body'],last['body'])
        objective=loss+.40*long+.06*mem
        optimizer.zero_grad(set_to_none=True);objective.backward()
        if step==1:
            nonzero=sum(p.numel() for name,p in m.named_parameters() if p.requires_grad and name.startswith('core.deep_cognitive_core') and p.grad is not None and torch.any(p.grad!=0))
            print('BACKPROP_100M_NONZERO',nonzero,flush=True)
            if nonzero < 100_000_000: raise RuntimeError('Deep 100M core did not participate in backpropagation; refusing to save checkpoint')
        torch.nn.utils.clip_grad_norm_((p for p in m.parameters() if p.requires_grad),1.)
        optimizer.step()
        if step%20==0 or step==1 or step==args.steps:
            val=evaluate(m,dev);agg=aggregate(val)
            trial=agg['worst_loss']+.5*agg['avg_loss']
            if trial<best_score-1e-4:
                best_score=trial;best_step=step
                # Persist only after actual development improvement.
                full={'architecture':'WeaverSOMA100M-stage5-sequential','synthetic_only':True,
                      'no_motor_authority':True,'pretrained_core_origin':'WeaverSOMA100M-stage4',
                      'source_stage4_sha256':ckpt_sha,'best_step':step,
                      'temporal_windows':4,'window_samples':8,
                      'weights':{k:v.detach().cpu() for k,v in m.state_dict().items()}}
                torch.save(full,OUT/'WEAVER_SOMA_U2_Stage5_Temporal_Research.pt')
            history.append({'step':step,'train_objective':round(float(objective.detach()),6),
                            'dev':agg,'best_step':best_step,'elapsed_s':round(time.monotonic()-now,1)})
            print('UPDATE',json.dumps(history[-1]),flush=True)
    # Freeze selected checkpoint. If no dev improvement, never claim promotion.
    if best_step:
        selected=torch.load(OUT/'WEAVER_SOMA_U2_Stage5_Temporal_Research.pt',map_location='cpu',weights_only=True)
        m.load_state_dict(selected['weights'],strict=True)
    final_dev=evaluate(m,dev)
    final_holdout=evaluate(m,holdout)
    # Causal ablation uses no older windows and tests if memory adds value.
    first_key='thermal_shift/flagged'
    d=holdout[first_key];m.eval()
    def ablation_brier(ablate):
        ys=[];ps=[]
        with torch.inference_mode():
            for i in range(0,len(d['obs']),16):
                b=slice_group(d,slice(i,i+16))
                z=m(b['obs'],b['quality'],b['context'],b['executed_action'],ablate_history=ablate)
                ys.append(b['risk_long'][:,-1].numpy())
                ps.append(z['long_hazard_logits'].sigmoid().numpy())
        y=np.concatenate(ys);p=np.concatenate(ps)
        return float(np.mean((y-p)**2))
    full_brier=ablation_brier(False);ablated_brier=ablation_brier(True)
    result={'description':'stage5 CPU-long-sequence continued pretraining research only',
            'real_hardware_frames':0,'simulated_training_trajectories':sum(len(x['obs']) for x in train.values()),
            'development_trajectories':sum(len(x['obs']) for x in dev.values()),
            'locked_test_trajectories':sum(len(x['obs']) for x in holdout.values()),
            'trajectory_frames':32,'stage4_checksum':ckpt_sha,
            'full_model_parameters':sum(p.numel() for p in m.parameters()),
            'trainable_parameters':trainable,'deep_core_trainable_parameters':deep_count,
            'steps':args.steps,'best_dev_step':best_step,'initial_dev':initial,'selected_dev':final_dev,
            'locked_test':final_holdout,
            'aggregate_initial_dev':aggregate(initial),'aggregate_selected_dev':aggregate(final_dev),
            'aggregate_locked_test':aggregate(final_holdout),
            'heldout_memory_ablation':{'group':first_key,'temporal_history_brier':full_brier,
                                       'without_older_windows_brier':ablated_brier},
            'elapsed_seconds':round(time.monotonic()-now,2),
            'limitations':'all synthetic; 32 sequential physical samples only; 24-step risk from physics; no camera/audio/text training; not physical safety or full-scale foundation pretraining'}
    (OUT/'stage5_results.json').write_text(json.dumps(result,indent=2))
    print('FINAL',json.dumps({k:v for k,v in result.items() if k not in ('initial_dev','selected_dev','locked_test')},indent=2),flush=True)

if __name__=='__main__':main()