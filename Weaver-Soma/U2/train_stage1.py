"""WEAVER-SOMA U2 stage-1 CPU training (synthetic-only, advisory-only).

Run from Weaver-Soma/U2:
    python train_stage1.py

Restore the original TRUSTED U1 weights in ../U1/results/weaver_soma_unified.pt.
Never send predictions from this experimental model to motors or signed intents.
"""
from pathlib import Path
import sys,hashlib,json,time
import numpy as np
import torch
from sklearn.metrics import average_precision_score
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(1,str(HERE.parent/'U1'))
from weaver_soma_u2 import WeaverSOMA100M,count_parameters
from unified_model import WeaverSOMAUnified,joint_objective
from train_unified import make_data,forward_model

TRUSTED_SHA256='3eb0746806f3842ec2d8121a7bcf30f5351b1b26b70e2c53debdac2ea5030c8f'
CHECKPOINT=HERE.parent/'U1'/'results'/'weaver_soma_unified.pt'
OUT=HERE/'results'
OUT.mkdir(exist_ok=True)

def score(model,data):
    model.eval();count=0;total=0.;pr=[]
    with torch.inference_mode():
        for i in range(0,len(data['obs']),16):
            b={k:v[i:i+16] for k,v in data.items()}
            o=forward_model(model,b)
            loss,_=joint_objective(o,b)
            count+=len(b['obs']); total+=float(loss)*len(b['obs'])
            pr.append(o['risk_logits'].sigmoid().numpy())
    risk=np.concatenate(pr,axis=0)
    actual=data['risk'].numpy()
    policy=risk.argmin(1)
    oracle=actual.min(1)
    violations=actual[np.arange(len(policy)),policy]
    return {'loss':total/count,
            'risk_ap':float(average_precision_score(actual.ravel(),risk.ravel())),
            'avoidable_pct':float((violations>oracle).mean()*100),
            'total_violation_pct':float(violations.mean()*100)}

def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261115)
    if not CHECKPOINT.exists():
        raise RuntimeError('Restore trusted U1 checkpoint to U1/results before training U2.')
    sha=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    if sha != TRUSTED_SHA256:
        raise ValueError('U1 SHA-256 mismatch: refusing to train from unexpected weights')
    original=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    teacher=WeaverSOMAUnified().eval()
    teacher.load_state_dict(original['weights'])
    student=WeaverSOMA100M()
    matching=student.load_u1_weights(str(CHECKPOINT))
    assert matching==91 and count_parameters(student)==100204067
    # Protect all 91 original tensors. Only new 100M shared core gets gradients.
    for name,param in student.named_parameters():
        param.requires_grad_(name.startswith('deep_cognitive_core.'))

    training=make_data(192,6042,'normal')
    validation=make_data(96,78042,'changed')
    TEST=make_data(96,89042,'extreme')
    baseline=score(teacher,validation)
    initial=score(student,validation)
    # Save the starting core so early stopping can restore epoch-zero best.
    best=initial['loss']; best_step=0
    best_core={k:v.cpu().clone() for k,v in student.deep_cognitive_core.state_dict().items()}
    optimizer=torch.optim.AdamW(
        (p for p in student.parameters() if p.requires_grad),
        lr=3e-6,weight_decay=0.0001)
    history=[];start=time.monotonic()
    for step in range(1,21):
        student.train()
        indices=torch.randperm(len(training['obs']))[:16]
        b={k:v[indices] for k,v in training.items()}
        out=forward_model(student,b)
        loss,_=joint_objective(out,b)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in student.parameters() if p.requires_grad),1.0)
        optimizer.step()
        if step in (1,2,4,8,12,16,20):
            current=score(student,validation)
            history.append({'step':step,'validation':current})
            if current['loss'] < best:
                best=current['loss'];best_step=step
                best_core={k:v.detach().cpu().clone()
                           for k,v in student.deep_cognitive_core.state_dict().items()}
            print(f"step {step}: val_loss={current['loss']:.6f}, best_step={best_step}",flush=True)
    student.deep_cognitive_core.load_state_dict(best_core)
    selected=score(student,validation)
    ood=score(student,TEST)
    report={'version':'WEAVER-SOMA U2 100M staged CPU test',
            'recovered_u1_tensors':matching,'parameters':count_parameters(student),
            'updates_run':20,'best_step':best_step,'u1_validation':baseline,
            'u2_initial_validation':initial,'selected_validation':selected,
            'selected_extreme_ood':ood,'history':history,
            'elapsed_s':round(time.monotonic()-start,2),
            'notes':'No motor action, no real sensors, no safety or consciousness evidence'}
    (OUT/'stage1_metrics.json').write_text(json.dumps(report,indent=2))
    # No 400MB weight file is published or promoted if best_step == 0.
    if best_step:
        torch.save({'checkpoint_version':'WEAVER-SOMA-U2-stage1',
                    'trusted_u1_sha256':TRUSTED_SHA256,
                    'core_weights':best_core,'best_step':best_step},
                    OUT/'u2_core_development.pt')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
