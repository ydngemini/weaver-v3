"""CPU-only behavioral tests. No external services or motor action."""
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT.parent/'U1')]
from unified_model import WeaverSOMAUnified
from weaver_soma_u2 import WeaverSOMA100M

def test_gate_is_exact_for_fully_healthy_sensors():
    torch.manual_seed(42)
    base=WeaverSOMAUnified().eval()
    bigger=WeaverSOMA100M(core_dim=64,depth=1,ff_dim=128).eval()
    result=bigger.load_state_dict(base.state_dict(),strict=False)
    assert all(k.startswith('deep_cognitive_core.') for k in result.missing_keys)
    bigger.deep_cognitive_core.train_mode='adapter'
    with torch.no_grad():bigger.deep_cognitive_core.adapter_gain.fill_(0.7)
    obs=torch.randn(3,8,7);quality=torch.ones(3,8,3)
    context=torch.randn(3,8)
    action=torch.tensor([[0.,1.,0.]]).repeat(3,1)
    with torch.inference_mode():
        original=base(obs,quality,context,action)
        candidate=bigger(obs,quality,context,action)
    for key in original:
        assert torch.equal(original[key],candidate[key]),key

def test_health_gate_changes_only_flagged_examples():
    torch.manual_seed(44)
    original=WeaverSOMAUnified().eval()
    bigger=WeaverSOMA100M(core_dim=64,depth=1,ff_dim=128).eval()
    bigger.load_state_dict(original.state_dict(),strict=False)
    bigger.deep_cognitive_core.train_mode='adapter'
    with torch.no_grad():bigger.deep_cognitive_core.adapter_gain.fill_(1.0)
    x=torch.randn(2,8,7);q=torch.ones(2,8,3);q[1, :, 1]=0
    c=torch.randn(2,8);a=torch.tensor([[0.,1.,0.],[0.,1.,0.]])
    with torch.inference_mode():
        base=original(x,q,c,a)
        altered=bigger(x,q,c,a)
    assert torch.equal(base['state'][0],altered['state'][0])
    assert not torch.equal(base['state'][1],altered['state'][1])

def test_missing_sensor_never_leaks_values():
    torch.manual_seed(45)
    bigger=WeaverSOMA100M(core_dim=64,depth=1,ff_dim=128).eval()
    bigger.deep_cognitive_core.train_mode='adapter'
    with torch.no_grad():bigger.deep_cognitive_core.adapter_gain.fill_(.8)
    obs=torch.randn(1,8,7);q=torch.ones(1,8,3);q[:,:,1]=0
    bad=obs.clone();bad[:,:,1]=1e6;bad[:,:,4]=-1e6
    context=torch.randn(1,8)
    with torch.inference_mode():
        before=bigger(obs,q,context)['risk_logits']
        after=bigger(bad,q,context)['risk_logits']
    assert torch.equal(before,after)

