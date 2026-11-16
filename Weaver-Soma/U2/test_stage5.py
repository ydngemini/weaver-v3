import sys,json,hashlib
from pathlib import Path
import torch,numpy as np,pytest
HERE=Path(__file__).resolve().parent
sys.path.append(str(HERE.parent/'U1'))
from temporal_model import StreamingWorldModel
from long_world import make_long
from sensor_stream import CaptureSession
from packet import FingerFrame,NEW_ADC,NEW_TEMP,NEW_ENCODER

def test_long_episode_shape_and_temporal_dependence():
    a=make_long(4,42,'normal');b=make_long(4,42,'normal')
    assert a['obs'].shape==(4,4,8,7)
    assert a['quality'].shape==(4,4,8,3)
    assert a['risk_long'].shape==(4,4,3)
    assert torch.equal(a['obs'],b['obs'])
    assert not torch.equal(a['obs'][:,0],a['obs'][:,1]),'must contain a truly evolving episode'

def test_generated_faults_never_unmask_corruption():
    a=make_long(3,11,'changed','flagged')
    assert set(torch.unique(a['quality']).tolist()).issubset({0.,1.})
    assert torch.isfinite(a['obs']).all()

def test_hardware_provenance_fail_closed():
    base={'node_id':3,'expected_can_id':515,'calibration_digest':'0'*64,
          'trusted_clock_offset_ms':0}
    with pytest.raises(ValueError):CaptureSession(base,source='hardware')
    c=CaptureSession(base,source='synthetic')
    assert c.meta['is_physical_measurement'] is False

def test_sensor_packet_to_window():
    base={'node_id':3,'expected_can_id':515,'calibration_digest':'a'*64,
          'trusted_clock_offset_ms':0}
    s=CaptureSession(base,source='synthetic')
    for j in range(190):
        f=FingerFrame(3,j+1,1000+j*10,(20,30,10,15),2000,2001,
                       3100,3100,5000,NEW_ADC|NEW_ENCODER|(NEW_TEMP if j%13==0 else 0))
        r=s.ingest(f.encode(),can_id=515,received_ms=1000+j*10+7)
        assert r['pressure_kpa'] is None
    obs,q,meta=s.latest_u2_window(2890)
    assert obs.shape==(8,7) and q.shape==(8,3)
    assert q[:,0].sum()==0,'uncalibrated force must NEVER be published as measured pressure'

def test_model_thirty_two_frame_forward_with_small_core():
    torch.set_num_threads(2)
    m=StreamingWorldModel()
    # Override only for this fast unit test; integration uses the real 100M core.
    from weaver_soma_u2 import WeaverSOMA100M
    m.core=WeaverSOMA100M(core_dim=64,depth=1,ff_dim=128)
    b=make_long(2,45,'normal')
    with torch.no_grad():o=m(b['obs'],b['quality'],b['context'],b['executed_action'])
    assert o['long_hazard_logits'].shape==(2,3)
    assert o['last']['risk_logits'].shape==(2,3)


def test_small_deep_core_receives_gradient():
    from weaver_soma_u2 import WeaverSOMA100M
    from unified_model import joint_objective
    torch.manual_seed(52)
    m=StreamingWorldModel()
    m.core=WeaverSOMA100M(core_dim=64,depth=1,ff_dim=128)
    with torch.no_grad():m.core.deep_cognitive_core.residual_gain.fill_(.025)
    d=make_long(2,55,'normal')
    o=m(d['obs'],d['quality'],d['context'],d['executed_action'])
    b={k:v[:,-1] for k,v in d.items() if k!='risk_long'}
    loss,_=joint_objective(o['last'],b)
    loss.backward()
    grad=m.core.deep_cognitive_core.blocks[0].ff[0].weight.grad
    assert grad is not None and torch.any(grad!=0),'U2 deep-core bypass detected'

def test_four_window_sensor_stream_to_trained_model_interface():
    base={'node_id':3,'expected_can_id':515,'calibration_digest':'a'*64,
          'trusted_clock_offset_ms':0}
    s=CaptureSession(base,source='synthetic')
    for j in range(455):
        f=FingerFrame(3,j+1,1000+j*10,(20,30,10,15),2000+j//20,2001+j//20,
                       3100+j//13,3100+j//13,5000,
                       NEW_ADC|NEW_ENCODER|(NEW_TEMP if j%13==0 else 0))
        s.ingest(f.encode(),can_id=515,received_ms=f.capture_ms+7)
    obs,q,meta=s.latest_u2_sequence((2100,3060,4020,4980))
    assert obs.shape==(4,8,7) and q.shape==(4,8,3)
    assert q[:,:,0].sum()==0  # real pressure still requires force AND area calibration
    assert q[:,:,2].sum()>0