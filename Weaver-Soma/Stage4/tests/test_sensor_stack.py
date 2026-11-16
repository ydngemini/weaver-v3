"""Offline CPU tests of deterministic sensor gateway, not physical certification."""
import dataclasses
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sensors import SensorPacket, HealthMonitor, SOMASensorWindow, DEFAULT_SPECS
from sensors.integrity import ChannelSpec

BASE=10_000_000_000
DT=100_000_000

def packet(channel,seq,value,now=None):
    return SensorPacket.create(DEFAULT_SPECS[channel],seq,BASE+seq*DT if now is None else now,value)

def healthy_cycle(n,seq):
    return {n:packet(n,seq,{'pressure':24.,'temperature':33.,'joint':.36}[n]+.003*seq) for n in DEFAULT_SPECS}

def test_warmup_then_true():
    m=HealthMonitor()
    for i in range(3):
        status=m.inspect(packet('temperature',i,33+i*.01),BASE+i*DT)
        assert status.trusted==(i>=1)
        assert status.reasons==(('reacquiring',) if i==0 else ())

def test_corrupt_crc_and_recovery():
    m=HealthMonitor()
    for i in range(2):m.inspect(packet('pressure',i,27+i),BASE+i*DT)
    bad=dataclasses.replace(packet('pressure',2,29),checksum=42)
    r=m.inspect(bad,BASE+2*DT)
    assert not r.trusted and 'checksum' in r.reasons
    assert not m.inspect(packet('pressure',3,30),BASE+3*DT).trusted
    assert m.inspect(packet('pressure',4,31),BASE+4*DT).trusted

def test_sequence_replay_stale_future():
    m=HealthMonitor()
    for i in range(2):m.inspect(packet('joint',i,.3+i*.01),BASE+i*DT)
    r=m.inspect(packet('joint',1,.5),BASE+2*DT)
    assert not r.trusted and 'sequence_replay' in r.reasons
    r=m.inspect(packet('joint',3,.32,BASE-1_000_000_000),BASE+3*DT)
    assert 'stale_timestamp' in r.reasons
    r=m.inspect(packet('joint',4,.32,BASE+10*DT),BASE+4*DT)
    assert 'future_timestamp' in r.reasons

def test_nonfinite_range_and_spoofed_identity():
    m=HealthMonitor()
    for i in range(2):m.inspect(packet('temperature',i,30+i*.02),BASE+i*DT)
    for value in (float('nan'),float('inf'),150):
        r=m.inspect(packet('temperature',4, value),BASE+4*DT)
        assert not r.trusted
    other=dataclasses.replace(packet('temperature',5,33),sensor_id='unknown')
    r=m.inspect(other,BASE+5*DT)
    assert not r.trusted and 'sensor_id_mismatch' in r.reasons

def test_stationary_is_not_fault_without_witness():
    m=HealthMonitor()
    for i in range(6):
        status=m.inspect(packet('pressure',i,35.),BASE+i*DT)
    assert status.trusted

def test_stuck_requires_independent_evidence():
    m=HealthMonitor()
    for i in range(2):m.inspect(packet('pressure',i,35.+i*2),BASE+i*DT)
    frozen=packet('pressure',2,37.)
    r=m.inspect(frozen,BASE+2*DT,witness_change=2.)
    assert not r.trusted and 'stuck_with_witness' in r.reasons

def test_redundant_reference_can_detect_plausible_bias():
    m=HealthMonitor()
    for i in range(2):m.inspect(packet('joint',i,.4+i*.02),BASE+i*DT)
    pkt=packet('joint',2,.62)
    status=m.inspect(pkt,BASE+2*DT,independent_reference=.44)
    assert not status.trusted and 'redundancy_disagreement' in status.reasons
    solo=HealthMonitor()
    for i in range(2):solo.inspect(packet('joint',i,.4+i*.02),BASE+i*DT)
    assert solo.inspect(pkt,BASE+2*DT).trusted  # unverified plausible bias is invisible

def test_window_shapes_quality_derivatives_and_missing():
    a=SOMASensorWindow()
    for i in range(3):
        frame=a.push(healthy_cycle('pressure',i)|healthy_cycle('temperature',i)|healthy_cycle('joint',i),BASE+i*DT,independently_verified_contact=True)
    assert frame.obs.shape==(8,7)
    assert frame.quality.shape==(8,3)
    assert frame.quality[-1].tolist()==[1.,1.,1.]
    assert frame.obs[-1,6]==1.
    assert np.isfinite(frame.obs).all()
    partial=a.push({'pressure':packet('pressure',3,24.5),'joint':packet('joint',3,.38)},BASE+3*DT)
    # Asynchronous, slower thermal sensor is fresh for 250ms; no false fault.
    assert not partial.stop_requested
    assert partial.quality[-1].tolist()==[1.,1.,1.]
    assert partial.obs[-1,4]==0  # held temp never invents a derivative
    assert partial.obs[-1,6]==0
    assert 'fresh_cache' in partial.reasons['temperature']
    # Missing longer than sensor TTL -> fail closed.
    stale=a.push({'pressure':packet('pressure',7,24.5),'joint':packet('joint',7,.38)},BASE+7*DT)
    assert stale.stop_requested
    assert stale.quality[-1].tolist()==[1.,0.,1.]
    assert stale.obs[-1,1]==0 and stale.obs[-1,4]==0
    assert 'missing' in stale.reasons['temperature']

def test_boundary_risk_not_confused_with_sensor_failure():
    a=SOMASensorWindow(safety_boundaries=(60.,43.,.8))
    for i in range(3):
        frame=a.push({c:packet(c,i,{'pressure':75.,'temperature':35.,'joint':.4}[c]) for c in DEFAULT_SPECS},BASE+i*DT)
    assert frame.stop_requested
    assert frame.quality[-1].tolist()==[1.,1.,1.]
    assert 'illustrative_boundary_exceeded' in frame.reasons['supervisor']

def test_bad_frame_order_rejected():
    a=SOMASensorWindow()
    a.push({},BASE)
    with pytest.raises(ValueError):a.push({},BASE)

def test_existing_torch_model_missing_channel_invariance():
    # Integration contract: masked U1/U2 cannot see a missing temp's values.
    sys.path.insert(0,'/mnt/data/weaver_soma_stage3/U2')
    from unified_model import WeaverSOMAUnified
    net=WeaverSOMAUnified().eval()
    a=SOMASensorWindow()
    for i in range(8):
        inputs={c:packet(c,i,{'pressure':30.,'temperature':33.,'joint':.3}[c]) for c in DEFAULT_SPECS}
        if i>=5: inputs.pop('temperature')
        result=a.push(inputs,BASE+i*DT)
    x=torch.from_numpy(result.obs).unsqueeze(0)
    q=torch.from_numpy(result.quality).unsqueeze(0)
    assert q[0,-1,1]==0
    altered=x.clone();altered[0,-1,1]=1e6;altered[0,-1,4]=-1e6
    context=torch.zeros(1,8)
    with torch.inference_mode():
        y=net(x,q,context)['risk_logits']
        z=net(altered,q,context)['risk_logits']
    assert torch.equal(y,z)
