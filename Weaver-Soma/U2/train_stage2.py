"""Weaver-SOMA U2 stage 2: reproduce the zero-gate CPU research trial.

Preserves the recovered U1 model exactly at initialization. Two separate
strategies: frozen 100M core + 2K adapter; frozen 100M core + deep scalar gate.
No actions are sent to hardware or production Weaver.
"""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import torch
from sklearn.metrics import average_precision_score

ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'U2'),str(ROOT/'U1')]
from unified_model import WeaverSOMAUnified,joint_objective
from weaver_soma_u2 import WeaverSOMA100M,count_parameters
from train_unified import make_data,forward_model,concat_data

CP=ROOT/'U1'/'results'/'weaver_soma_unified.pt'
SHA='3eb0746806f3842ec2d8121a7bcf30f5351b1b26b70e2c53debdac2ea5030c8f'
OUT=ROOT/'U2'/'results'

def evaluate(model,data):
    model.eval()
    total=0.;prob=[];n=len(data['obs'])
    with torch.inference_mode():
        for i in range(0,n,64):
            sub={k:v[i:i+64] for k,v in data.items()}
            pred=forward_model(model,sub)
            loss,_=joint_objective(pred,sub)
            total+=float(loss)*len(sub['obs'])
            prob.append(pred['risk_logits'].sigmoid())
    p=torch.cat(prob).numpy()
    y=data['risk'].numpy()
    actions=(p+.02*np.array([0.,.2,1.],dtype=np.float32)).argmin(axis=1)
    outcome=y[np.arange(n),actions]
    return {'loss':total/n,'risk_ap':float(average_precision_score(y.ravel(),p.ravel())),
            'total_violation_pct':float(outcome.mean()*100),
            'avoidable_pct':float((outcome>y.min(axis=1)).mean()*100),
            'withdraw_pct':float((actions==2).mean()*100)}

def main():
    if not CP.exists() or hashlib.sha256(CP.read_bytes()).hexdigest()!=SHA:
        raise RuntimeError('Missing or mismatched original trusted U1 checkpoint')
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(20261115)
    obj=torch.load(CP,map_location='cpu',weights_only=True)
    teacher=WeaverSOMAUnified().eval()
    teacher.load_state_dict(obj['weights'],strict=True)
    student=WeaverSOMA100M()
    assert student.load_u1_weights(str(CP))==91
    assert count_parameters(student)==100206197

    train=concat_data(make_data(512,2600,'normal'),make_data(512,2601,'changed'))
    dev=concat_data(make_data(192,2800,'normal'),make_data(192,2801,'changed'))
    holdouts={name:make_data(300,3000+i,name)
              for i,name in enumerate(('normal','changed','extreme'))}
    # Every original output must be identical, not merely close.
    teacher.eval();student.eval()
    with torch.inference_mode():
        b={k:v[:16] for k,v in dev.items()}
        a=forward_model(teacher,b);z=forward_model(student,b)
    diffs={k:float((a[k]-z[k]).abs().max()) for k in a}
    assert max(diffs.values())==0.,diffs

    baseline={tag:evaluate(teacher,data)
              for tag,data in [('dev',dev),*holdouts.items()]}

    # Strategy A: train only ~2k adapter parameters.
    for name,p in student.named_parameters():
        p.requires_grad_(name.startswith('deep_cognitive_core.adapter'))
    student.deep_cognitive_core.train_mode='adapter'
    optimizer=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],lr=.001)
    initial=evaluate(student,dev)['loss']
    best=initial;best_step=0
    best_adapter={k:v.detach().clone() for k,v in student.deep_cognitive_core.adapter.state_dict().items()}
    best_gain=student.deep_cognitive_core.adapter_gain.detach().clone()
    curve=[];start=time.monotonic()
    for step in range(1,161):
        student.train()
        indices=torch.randperm(len(train['obs']))[:32]
        batch={k:v[indices] for k,v in train.items()}
        loss,_=joint_objective(forward_model(student,batch),batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad],1.0)
        optimizer.step()
        if step in (1,5,10,20,40,80,120,160):
            current=evaluate(student,dev)
            curve.append({'step':step,**current})
            if current['loss']+1e-7<best:
                best=current['loss'];best_step=step
                best_adapter={k:v.detach().clone() for k,v in student.deep_cognitive_core.adapter.state_dict().items()}
                best_gain=student.deep_cognitive_core.adapter_gain.detach().clone()
    student.deep_cognitive_core.adapter.load_state_dict(best_adapter)
    with torch.no_grad(): student.deep_cognitive_core.adapter_gain.copy_(best_gain)
    adapter_scores={tag:evaluate(student,data)
                    for tag,data in [('dev',dev),*holdouts.items()]}

    # Strategy B: train the gate only; all ~100M deep weights remain frozen.
    torch.manual_seed(20261115)
    pilot=WeaverSOMA100M()
    pilot.load_u1_weights(str(CP))
    pilot.deep_cognitive_core.train_mode='deep'
    for name,p in pilot.named_parameters():
        p.requires_grad_(name=='deep_cognitive_core.residual_gain')
    opt=torch.optim.AdamW([pilot.deep_cognitive_core.residual_gain],lr=.0003)
    gate_hist=[];gate_best=evaluate(pilot,dev)['loss'];gate_step=0
    for step in range(1,13):
        pilot.train()
        indices=torch.randperm(len(train['obs']))[:16]
        batch={k:v[indices] for k,v in train.items()}
        loss,_=joint_objective(forward_model(pilot,batch),batch)
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step in (1,4,8,12):
            v=evaluate(pilot,dev)
            gate_hist.append({'step':step,**v})
            if v['loss']<gate_best:
                gate_best=v['loss'];gate_step=step

    report={'total_parameters':count_parameters(student),
            'trained_u1_tensors':91,'frozen_deep_core':True,
            'u1_checkpoint_sha256':SHA,
            'exact_initial_max_absolute_difference':max(diffs.values()),
            'train_episodes':len(train['obs']),'dev_episodes':len(dev['obs']),
            'fresh_episodes_per_regime':300,'baseline':baseline,
            'adapter_best_update':best_step,'adapter_curve':curve,
            'adapter_scores':adapter_scores,'gate_best_update':gate_step,
            'gate_curve':gate_hist,'runtime_s':round(time.monotonic()-start,2),
            'status':'Research only. Synthetic CPU trial; NOT physical robot safety.'}
    (OUT/'stage2_results.json').write_text(json.dumps(report,indent=2))
    # Separate experimental adapter artifact, never automatically deploy.
    if best_step>0:
        torch.save({'version':'WEAVER-SOMA-U2-stage2-adapter',
                    'original_checkpoint_sha256':SHA,'weights':best_adapter,
                    'adapter_gain':best_gain,'best_step':best_step},
                   OUT/'u2_adapter_only.pt')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
