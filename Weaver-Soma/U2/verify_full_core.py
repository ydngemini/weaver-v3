"""Stage 4 independent verification of trained 100M U2 core.
Research-only. No hardware control, production deployment, or signed intents.
"""
from pathlib import Path
import hashlib,json,sys
import torch

ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'U2'),str(ROOT/'U1')]
from weaver_soma_u2 import WeaverSOMA100M,count_parameters
SHA='3eb0746806f3842ec2d8121a7bcf30f5351b1b26b70e2c53debdac2ea5030c8f'

def main():
    torch.set_num_threads(3)
    original_path=ROOT/'U1'/'weaver_soma_unified.pt'
    assert hashlib.sha256(original_path.read_bytes()).hexdigest()==SHA
    metrics=json.loads((ROOT/'results'/'stage4_fullcore_metrics.json').read_text())
    ckpt_path=ROOT/'results'/'u2_fullcore_100m_best_core_fp32.pt'
    ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=True)
    assert ckpt['u1_sha256']==SHA
    assert ckpt['architecture']=='WeaverSOMA100M-v2-fullcore'
    torch.manual_seed(3127)
    untrained=WeaverSOMA100M().eval()
    untrained.load_u1_weights(str(original_path))
    initial={k:v.clone() for k,v in untrained.deep_cognitive_core.state_dict().items()
             if 'blocks.' in k and k.endswith('weight')}
    trained=WeaverSOMA100M().eval()
    assert trained.load_u1_weights(str(original_path))==91
    trained.deep_cognitive_core.load_state_dict(ckpt['deep_core_fp32'],strict=True)
    assert count_parameters(trained)==100206197
    changes={k:float((trained.deep_cognitive_core.state_dict()[k]-v).abs().mean())
             for k,v in initial.items()}
    assert len(changes)==60 and all(x>0 for x in changes.values())
    assert metrics['nonzero_gradient_parameters_after_step_2']==100082241
    x=torch.randn(2,8,7)
    q=torch.ones(2,8,3)
    q[1,:,1]=0
    context=torch.rand(2,8)
    action=torch.tensor([[0.,1.,0.],[1.,0.,0.]])
    with torch.no_grad():
        baseline=trained(x,q,context,action)
        corrupted=x.clone()
        corrupted[1,:,1]=1000000
        corrupted[1,:,4]=-1000000
        masked=trained(corrupted,q,context,action)
    for k in ('risk_logits','state','future','policy_logits'):
        assert torch.isfinite(baseline[k]).all(),k
        assert torch.allclose(baseline[k][1],masked[k][1],atol=1e-6),k
    print(json.dumps({
        'status':'PASS',
        'core_weights_changed':len(changes),
        'nonzero_gradient_parameters':metrics['nonzero_gradient_parameters_after_step_2'],
        'checkpoint_bytes':ckpt_path.stat().st_size,
        'missing_sensor_invariance':'PASS'
    },indent=2))

if __name__=='__main__':
    main()
