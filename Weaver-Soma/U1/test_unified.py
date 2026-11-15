"""pytest -q test_unified.py; tests never actuate anything."""
import numpy as np,torch
from pathlib import Path
from train_unified import make_data,forward_model,invariance_check
from unified_model import WeaverSOMAUnified
from weaver_adapter import WeaverSOMABridge
ROOT=Path(__file__).resolve().parent
CKPT=str(ROOT/'results'/'weaver_soma_unified.pt')

def test_one_model_checkpoint():
    item=torch.load(CKPT,map_location='cpu',weights_only=True)
    assert item['architecture']=='WeaverSOMAUnified-v1'
    m=WeaverSOMAUnified();m.load_state_dict(item['weights'])
    assert sum(p.numel() for p in m.parameters())==121827
    assert len(m.experts)==5

def test_outputs_and_normalization():
    b=make_data(22,31)
    net=WeaverSOMAUnified();out=forward_model(net,b)
    assert out['risk_logits'].shape==(22,3)
    assert out['future'].shape==(22,3,6,3)
    assert out['body_tokens'].shape==(22,4)
    assert out['workspace_weights'].shape==(22,5)
    assert torch.allclose(out['workspace_weights'].sum(-1),torch.ones(22),atol=1e-5)

def test_mask_invariance():
    model=WeaverSOMABridge(CKPT).model
    assert invariance_check(model)==0

def test_exact_weaver_mesh_schema():
    b=make_data(1,55);q=b['quality'][0].numpy()
    view,latent=WeaverSOMABridge(CKPT).infer(
       b['obs'][0].numpy(),q,b['context'][0].numpy(),b['executed_action'][0].numpy())
    assert view.action_proposal in {'hold','ease','withdraw'}
    assert len(latent)==64
    msg=WeaverSOMABridge.to_mesh_observation(q,age_ms=25)
    assert set(msg)=={'observed_at_ms','sensors'}
    assert set(msg['sensors'])=={'body'}
    assert set(msg['sensors']['body'])=={'confidence','sample_age_ms'}
    assert 0<=msg['sensors']['body']['confidence']<=1
    assert WeaverSOMABridge.to_hub_vector(latent).shape==(256,)
    assert set(WeaverSOMABridge.to_memory_metadata(view))=={'kind','tags','reward','surprise','risk','success'}
    req=WeaverSOMABridge.safe_intent_for_observation({'body_revision':8,'world_revision':3})
    assert req['actions']==[{'type':'observe','sensor':'body'}]
    assert 'integrity' not in req # only Weaver's own IntentCompiler signs

def test_invalid_channels_rejected():
    try:WeaverSOMABridge.to_mesh_observation(np.ones((2,3),dtype='float32'))
    except ValueError:pass
    else:raise AssertionError('invalid shape accepted')
    try:WeaverSOMABridge.to_mesh_observation(np.full((8,3),.5,dtype='float32'))
    except ValueError:pass
    else:raise AssertionError('ambiguous sensor health accepted')
