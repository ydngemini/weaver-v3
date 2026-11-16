"""Structure and backward-compatibility tests, NOT a model quality evaluation."""
import torch
from unified_model import WeaverSOMAUnified, masked_inputs
from weaver_soma_u2 import WeaverSOMA100M, count_parameters


def main():
    torch.manual_seed(17)
    torch.set_num_threads(4)
    old=WeaverSOMAUnified().eval()
    new=WeaverSOMA100M().eval()
    assert count_parameters(new)==100206197, count_parameters(new)
    x=torch.randn(1,8,7); q=torch.ones(1,8,3); c=torch.randn(1,8)
    a=torch.tensor([[0.,1.,0.]])
    with torch.no_grad():
        y0=old(x,q,c,a); y1=new(x,q,c,a)
    assert y0.keys()==y1.keys()
    for key in y0:
        assert y0[key].shape==y1[key].shape,(key,y0[key].shape,y1[key].shape)
        assert torch.isfinite(y1[key]).all(),key

    # Mask test: no temperature observation or derivative can leak to output.
    q[:,:,1]=0
    corrupted=x.clone(); corrupted[:,:,1]=9999; corrupted[:,:,4]=-9999
    with torch.no_grad():
        s1=new(x,q,c,a)['risk_logits']
        s2=new(corrupted,q,c,a)['risk_logits']
    assert torch.allclose(s1,s2,atol=1e-6), 'sensor leakage'
    # With both neural residual gates initialized to zero, the newly grown
    # network produces exactly U1's existing outputs when original U1 weights
    # are restored. This is tested after the state_dict copy below.

    # Verify newly grown core can consume matching older state dict.
    # This uses UNTRAINED U1 weights only to test schema, not to suggest
    # the original trained checkpoint was recovered.
    result=new.load_state_dict(old.state_dict(),strict=False)
    assert not result.unexpected_keys
    assert all(k.startswith('deep_cognitive_core.') for k in result.missing_keys)
    old.eval(); new.eval()
    with torch.inference_mode():
        y0=old(x,q,c,a); y1=new(x,q,c,a)
    for key in y0:
        assert torch.equal(y0[key],y1[key]), key
    assert new.deep_cognitive_core.residual_gain.item()==0
    assert new.deep_cognitive_core.adapter_gain.item()==0
    print('PASS: output contract; finite outputs; missing-sensor isolation; U1 state_dict schema')


if __name__=='__main__':main()
