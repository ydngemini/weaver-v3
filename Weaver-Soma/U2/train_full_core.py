"""Stage 4: actual full-100M-deep-core CPU training (synthetic, research-only).

All 100M+ deep-core tensors receive AdamW gradient updates; original U1 weights remain frozen.
Checkpoint selection uses independently generated development data; extreme and future
holdout examples never influence model selection. No hardware/production integrations.
"""
from __future__ import annotations
import argparse,hashlib,json,random,time,copy,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'U1')]
from weaver_soma_u2 import WeaverSOMA100M,count_parameters
from unified_model import WeaverSOMAUnified,joint_objective
from train_unified import make_data,forward_model
from train_stage3 import augment

ROOT=HERE.parent
CKPT=ROOT/'U1'/'weaver_soma_unified.pt'
SHA='3eb0746806f3842ec2d8121a7bcf30f5351b1b26b70e2c53debdac2ea5030c8f'
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
REGIMES=('normal','changed')
CONDS=('clean','noise','flagged','silent')


def subset(b, ids):return {k:v[ids] for k,v in b.items()}


def make_groups(n, seed, regimes=REGIMES, conditions=CONDS):
    groups={}
    for i,reg in enumerate(regimes):
        raw=make_data(n,seed+100*i,reg)
        for j,kind in enumerate(conditions):
            groups[f'{reg}/{kind}']=augment(raw,seed+1000*i+7*j+500,kind)
    return groups


def evaluate(model,d,batch_size=128):
    model.eval()
    logits=[];total=0.0
    with torch.inference_mode():
        for a in range(0,len(d['obs']),batch_size):
            b=subset(d,slice(a,a+batch_size))
            outputs=forward_model(model,b)
            loss,_=joint_objective(outputs,b)
            total+=float(loss)*len(b['obs'])
            logits.append(outputs['risk_logits'].sigmoid().cpu())
    predicted=torch.cat(logits).numpy()
    y=d['risk'].numpy()
    pi=(predicted+.02*np.array([0,.2,1.],dtype='float32')).argmin(axis=1)
    selected=y[np.arange(len(pi)),pi]
    oracle=y.min(axis=1)
    return {'loss':round(total/len(y),7),
       'risk_ap':round(float(average_precision_score(y.ravel(),predicted.ravel())),6),
       'risk_brier':round(float(np.square(predicted-y).mean()),7),
       'avoidable_pct':round(float(100*(selected>oracle).mean()),4),
       'total_violations_pct':round(float(100*selected.mean()),4),
       'withdraw_pct':round(float(100*(pi==2).mean()),4),
       'unavoidable_floor_pct':round(float(100*oracle.mean()),4),
       'n':len(y)}


def eval_groups(model,grps):return {tag:evaluate(model,data) for tag,data in grps.items()}

def aggregate(metrics):
    values=[row['loss'] for row in metrics.values()]
    clean=[row['loss'] for tag,row in metrics.items() if tag.endswith('/clean')]
    return {'mean_loss':round(float(np.mean(values)),7),
            'worst_loss':round(float(max(values)),7),
            'clean_mean_loss':round(float(np.mean(clean)),7)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=240)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--interval',type=int,default=30)
    parser.add_argument('--lr',type=float,default=3e-5)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    torch.manual_seed(3127);np.random.seed(3127);random.seed(3127)
    start=time.time()
    if not CKPT.exists():raise FileNotFoundError(f'Restore original U1 at {CKPT}')
    if hashlib.sha256(CKPT.read_bytes()).hexdigest()!=SHA:raise ValueError('Checkpoint SHA-256 mismatch')
    cp=torch.load(CKPT,map_location='cpu',weights_only=True)
    teacher=WeaverSOMAUnified().eval();teacher.load_state_dict(cp['weights'],strict=True)
    for p in teacher.parameters():p.requires_grad_(False)
    torch.manual_seed(3127)
    student=WeaverSOMA100M();n_tensors=student.load_u1_weights(str(CKPT));
    assert n_tensors==91
    total_params=count_parameters(student)
    assert total_params==100206197,total_params
    for name,p in student.named_parameters():
        p.requires_grad_(name.startswith('deep_cognitive_core.') and not name.startswith('deep_cognitive_core.adapter'))
    with torch.no_grad():
        student.deep_cognitive_core.residual_gain.fill_(.02)
        student.deep_cognitive_core.adapter_gain.zero_()
    trainable=sum(p.numel() for p in student.parameters() if p.requires_grad)
    assert trainable>100_000_000,trainable
    student.deep_cognitive_core.train_mode='deep'
    print(json.dumps({'total_parameters':total_params,'actually_trainable':trainable,'restored_u1_tensors':n_tensors,'initial_residual_gain':0.02,'steps_requested':args.steps}),flush=True)

    train=make_groups(1024,41000)
    dev=make_groups(256,51000)
    heldout=make_groups(700,61000,('normal','changed','extreme'),CONDS)
    base_dev=eval_groups(teacher,dev)
    base_heldout=eval_groups(teacher,heldout)
    initial_dev=eval_groups(student,dev)
    print('BASE_DEV',aggregate(base_dev),'INIT_DEV',aggregate(initial_dev),flush=True)

    opt=torch.optim.AdamW((p for p in student.parameters() if p.requires_grad),lr=args.lr,weight_decay=.00005,foreach=False)
    train_tags=list(train)
    best_step=0
    best_metrics=aggregate(initial_dev)
    chosen_decrease=0
    history=[{'step':0,**best_metrics}]
    teacher_agg=aggregate(base_dev)
    best_score=best_metrics['worst_loss'] + .5*best_metrics['mean_loss']
    candidate_state=None
    checkpoint_path=OUT/'u2_fullcore_100m_best_core_fp32.pt'
    grad_count=0
    core_snapshot=student.deep_cognitive_core.blocks[0].ff[0].weight.detach().view(-1)[0].item()
    for step in range(1,args.steps+1):
        tag=train_tags[(step-1)%len(train_tags)]
        data=train[tag]
        ix=torch.randint(len(data['obs']),(args.batch_size,))
        b=subset(data,ix)
        student.train()
        o=forward_model(student,b)
        loss,_=joint_objective(o,b)
        with torch.no_grad(): ref=forward_model(teacher,b)
        loss=loss + .03*F.mse_loss(o['risk_logits'],ref['risk_logits']) + .005*F.mse_loss(o['state'],ref['state'])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step==2:
            grad_count=sum(p.numel() for name,p in student.named_parameters()
                 if p.requires_grad and p.grad is not None and torch.any(p.grad!=0).item())
        torch.nn.utils.clip_grad_norm_((p for p in student.parameters() if p.requires_grad),1.0)
        opt.step()
        if step==1 or step%args.interval==0 or step==args.steps:
            scores=eval_groups(student,dev)
            agg=aggregate(scores)
            trial_score=agg['worst_loss']+.5*agg['mean_loss']
            allowed_mean=teacher_agg['mean_loss']+.01
            if (trial_score<best_score-0.0001 and agg['mean_loss']<=allowed_mean):
                best_step=step;best_score=trial_score;best_metrics=agg
                chosen_decrease+=1
                torch.save({'architecture':'WeaverSOMA100M-v2-fullcore','stage':'research_only',
                    'u1_sha256':SHA,'best_step':step,'total_parameters':total_params,
                    'deep_core_fp32':{k:v.detach().cpu() for k,v in student.deep_cognitive_core.state_dict().items()},
                    'optimizer':'AdamW CPU', 'model_was_actuated':False},checkpoint_path)
            history.append({'step':step,'train_loss':round(float(loss.detach()),7),**agg,'best_step':best_step,'seconds':round(time.time()-start,1)})
            print('UPDATE',json.dumps(history[-1]),flush=True)

    all_core_changed=abs(float(student.deep_cognitive_core.blocks[0].ff[0].weight.detach().view(-1)[0])-core_snapshot)
    if not checkpoint_path.exists():
        checkpoint_path=OUT/'u2_fullcore_100m_last_unpromoted_fp32.pt'
        torch.save({'architecture':'WeaverSOMA100M-v2-fullcore','stage':'research_only_no_dev_gain',
                    'u1_sha256':SHA,'best_step':args.steps,'total_parameters':total_params,
                    'deep_core_fp32':{k:v.detach().cpu() for k,v in student.deep_cognitive_core.state_dict().items()},
                    'optimizer':'AdamW CPU','model_was_actuated':False},checkpoint_path)
        selection='NO_IMPROVEMENT_BASELINE_RETAINED'
    else:selection='BEST_DEV_RESEARCH_ONLY'
    chosen=WeaverSOMA100M()
    assert chosen.load_u1_weights(str(CKPT))==91
    selected=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    chosen.deep_cognitive_core.load_state_dict(selected['deep_core_fp32'],strict=True)
    chosen.deep_cognitive_core.train_mode='deep'
    chosen.eval()
    selected_dev=eval_groups(chosen,dev)
    final_heldout=eval_groups(chosen,heldout)
    gate={}
    for tag in final_heldout:
        a=base_heldout[tag];b=final_heldout[tag]
        gate[tag]={'loss_diff':round(b['loss']-a['loss'],6),
                   'risk_ap_diff':round(b['risk_ap']-a['risk_ap'],6),
                   'avoidable_pct_diff':round(b['avoidable_pct']-a['avoidable_pct'],5)}
    report={'experiment':'WEAVER-SOMA U2 Stage 4 full 100M CPU optimizer run',
            'source_checkpoint_sha256':SHA,'original_u1_tensors_preserved':91,
            'parameters_total':total_params,'parameters_updated_in_optimizer':trainable,
            'nonzero_gradient_parameters_after_step_2':int(grad_count),
            'example_deep_weight_abs_change':all_core_changed,
            'updates_executed':args.steps,'training_episodes_unique_physics':2048,
            'development_unique_physics':512,'heldout_unique_physics':2100,
            'training_conditions':list(CONDS),'regimes_used_for_training':list(REGIMES),
            'best_dev_step':best_step,'selection':selection,
            'checkpoint_path':str(checkpoint_path),'checkpoint_bytes':checkpoint_path.stat().st_size,
            'base_dev':base_dev,'base_heldout':base_heldout,
            'u2_dev_selected':selected_dev,'u2_heldout_selected':final_heldout,
            'delta_heldout':gate,'history':history,'seconds':round(time.time()-start,2),
            'real_sensors':False,'motor_permission':False,
            'note':'Synthetic-only. 100M core trained; model not physically validated, no production use or consciousness evidence.'}
    (OUT/'stage4_fullcore_metrics.json').write_text(json.dumps(report,indent=2))
    print('FINAL',json.dumps({k:v for k,v in report.items() if k not in ('base_dev','base_heldout','u2_dev_selected','u2_heldout_selected','history','delta_heldout')},indent=2),flush=True)
    print('DIFFERENCES',json.dumps(gate,indent=2),flush=True)


if __name__=='__main__':main()
