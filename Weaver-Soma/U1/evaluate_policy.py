"""Counterfactual decision rules over a frozen model; validation-selected only."""
import json,csv
from pathlib import Path
import numpy as np,torch
from train_unified import make_data,forward_model
from unified_model import WeaverSOMAUnified
HERE=Path(__file__).resolve().parent

def outputs(model,data):
    model.eval();risk=[];head=[]
    with torch.inference_mode():
        for i in range(0,len(data['obs']),512):
            batch={k:v[i:i+512] for k,v in data.items()}
            x=forward_model(model,batch)
            risk.append(x['risk_logits'].sigmoid())
            head.append(x['policy_logits'])
    return torch.cat(risk).numpy(),torch.cat(head).numpy()

def evaluate(policy,labels):
    oracle=labels.min(axis=1)
    chosen=labels[np.arange(len(policy)),policy]
    return {'violation_pct':float(chosen.mean()*100),
            'avoidable_pct':float(np.mean(chosen>oracle)*100),
            'withdraw_pct':float(np.mean(policy==2)*100),
            'ease_pct':float(np.mean(policy==1)*100)}

def main():
    torch.set_num_threads(2)
    cp=torch.load(HERE/'results/weaver_soma_unified.pt',map_location='cpu',weights_only=True)
    model=WeaverSOMAUnified();model.load_state_dict(cp['weights'])
    dv=[make_data(1700,777,'normal'),make_data(1700,778,'changed')]
    test=[('normal',make_data(2200,10000,'normal')),
          ('changed',make_data(2200,10001,'changed')),
          ('extreme',make_data(2200,10002,'extreme'))]
    # Select among risk-based action selection penalties ONLY on development data.
    candidates=[]
    for p in [0.,.02,.05,.10,.2,.35,.5,1.0]:
        overall=[]
        for d in dv:
            r,_=outputs(model,d)
            pi=(r + p*np.array([0.,.2,1.],dtype=np.float32)).argmin(axis=1)
            overall.append(evaluate(pi,d['risk'].numpy()))
        # use a cost tradeoff; pure safety would always withdraw where safer
        candidates.append({'penalty':p,'development_avoidable_pct':float(np.mean([x['avoidable_pct'] for x in overall])),
                           'development_withdraw_pct':float(np.mean([x['withdraw_pct'] for x in overall]))})
    # Pick minimum avoidable among those not trivially 100% withdraw.
    candidates.sort(key=lambda x:(x['development_avoidable_pct']+max(0,x['development_withdraw_pct']-65)*.05,x['development_withdraw_pct']))
    chosen=candidates[0]['penalty']
    rows=[]
    for name,d in test:
        risk,head=outputs(model,d)
        actual=d['risk'].numpy()
        ps={'learned_policy_head':head.argmax(1),'risk_based_frozen':(risk+chosen*np.array([0.,.2,1.],dtype=np.float32)).argmin(1),
            'always_withdraw':np.full(len(d['obs']),2,dtype=np.int64)}
        for k,policy in ps.items():rows.append({'regime':name,'strategy':k,'selected_cost_penalty':chosen,**evaluate(policy,actual)})
    with (HERE/'results/policy_alternatives.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (HERE/'results/policy_development.json').open('w') as f:json.dump({'candidates':candidates,'chosen_penalty':chosen},f,indent=2)
    print(json.dumps({'chosen_penalty':chosen,'candidates':candidates,'test':rows},indent=2))
if __name__=='__main__':main()
