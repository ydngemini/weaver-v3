"""Evaluate selected stage 3 adapter with fault-gating on completely unseen seeds.
No additional training, and no checkpoint selection on these test outcomes.
"""
import sys,json,hashlib,copy
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'U2'),str(ROOT/'U1')]
from weaver_soma_u2 import WeaverSOMA100M
from unified_model import WeaverSOMAUnified
from train_stage3 import CP,SHA,OUT,evaluate,augment
from train_unified import make_data,forward_model

def main():
    torch.set_num_threads(2)
    assert hashlib.sha256(CP.read_bytes()).hexdigest()==SHA
    cp=torch.load(CP,map_location='cpu',weights_only=True)
    ref=WeaverSOMAUnified().eval();ref.load_state_dict(cp['weights'])
    candidate=WeaverSOMA100M();candidate.load_u1_weights(str(CP))
    candidate.deep_cognitive_core.train_mode='adapter'
    experimental=torch.load(OUT/'stage3_adapter_experimental.pt',map_location='cpu',weights_only=True)
    candidate.deep_cognitive_core.adapter.load_state_dict(experimental['adapter_weights'])
    with torch.no_grad():candidate.deep_cognitive_core.adapter_gain.copy_(experimental['adapter_gain'])
    candidate.eval()
    # Unit test: with healthy quality, U1 and stage3 gated-U2 must be identical.
    raw=make_data(48,41000,'normal')
    healthy={k:v.clone() for k,v in raw.items()}
    healthy['quality'].fill_(1)
    with torch.inference_mode():
        y0=forward_model(ref,healthy);y1=forward_model(candidate,healthy)
    exact={k:float((y0[k]-y1[k]).abs().max()) for k in y0}
    assert max(exact.values())==0,exact
    # Tests are independent from Stage3 training/selection AND Stage3 first test seeds.
    results=[]
    for i,regime in enumerate(('normal','changed','extreme')):
        raw=make_data(1800,51000+i,regime)
        for j,condition in enumerate(('clean','noise','flagged','silent')):
            d=augment(raw,52000+10*i+j,condition)
            b=evaluate(ref,d,regime,condition)
            u=evaluate(candidate,d,regime,condition)
            results.append({'regime':regime,'condition':condition,'u1':b,'u2_health_gated':u})
            print(regime,condition,'U1',round(b['joint_loss'],4),'U2',round(u['joint_loss'],4),
                  'avoidable',round(b['avoidable_pct'],3),round(u['avoidable_pct'],3),flush=True)
    report={'status':'heldout validation of previously selected Stage3 adapter, no retraining',
     'exact_output_difference_with_all_sensors_healthy':max(exact.values()),
     'test_episodes_per_regime':1800,'synthetic_only':True,'test':results}
    (OUT/'health_gate_heldout.json').write_text(json.dumps(report,indent=2))
    print('HEALTH-GATE OUTPUTS EXACT',max(exact.values()))
if __name__=='__main__':main()

