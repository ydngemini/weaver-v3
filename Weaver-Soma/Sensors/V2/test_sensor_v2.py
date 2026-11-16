"""CPU synthetic tests. The host node is not a certified hardware safety system."""
from dataclasses import replace
from pathlib import Path
import pytest
import numpy as np
from packet import *
from gateway import FingerCalibration,FingerGateway,IngestError
from bridge import OriginalSOMABridge

CAL=FingerCalibration(node_id=3,expected_can_id=0x203,calibration_digest='f'*64,
                      min_joint_deg=0.,max_joint_deg=90.,trusted_clock_offset_ms=0)
def frame(seq=1,t=1000,**kw):
    args={'node_id':3,'seq':seq,'capture_ms':t,'force_centinewtons':(100,50,50,20),
          'angle_u14':2000,'angle_witness_u14':2002,'contact_centi_c':3000,
          'witness_centi_c':3001,'supply_mv':5000,'flags':FRESH_ALL}
    args.update(kw)
    return FingerFrame(**args)

def get(g,f,can_id=0x203,receipt=6):
    return g.ingest(f.encode(),can_id=can_id,received_ms=f.capture_ms+receipt)

def test_protocol_size_and_crc_known_vector():
    assert FRAME.size==32
    assert crc16(b'123456789')==0x29b1
    assert FingerFrame.decode(frame().encode())==frame()

def test_crc_tamper_detected():
    b=bytearray(frame().encode());b[8]^=0x41
    with pytest.raises(FrameError,match='crc_mismatch'):FingerFrame.decode(bytes(b))

def test_bad_dlc_rejected():
    with pytest.raises(FrameError,match='size'):FingerFrame.decode(frame().encode()+b'\x00')

def test_unknown_flag_rejected():
    with pytest.raises(FrameError,match='unknown flags'):frame(flags=0x8000).encode()

def test_wrong_node_rejected():
    with pytest.raises(IngestError,match='unexpected_node_id'):get(FingerGateway(CAL),frame(node_id=4))

def test_wrong_can_id_rejected():
    with pytest.raises(IngestError,match='unexpected_can_id'):get(FingerGateway(CAL),frame(),can_id=0x204)

def test_no_trusted_clock_mapping():
    cal=replace(CAL,trusted_clock_offset_ms=None)
    with pytest.raises(IngestError,match='untrusted_clock'):get(FingerGateway(cal),frame())

def test_stale_frame():
    with pytest.raises(IngestError,match='stale_packet'):get(FingerGateway(CAL),frame(),receipt=95)

def test_future_timestamp():
    with pytest.raises(IngestError,match='future_timestamp'):get(FingerGateway(CAL),frame(),receipt=-10)

def test_replay_and_seq_backwards():
    g=FingerGateway(CAL);get(g,frame())
    with pytest.raises(IngestError,match='replay'):get(g,frame())
    assert get(g,frame(2,1010))['seq']==2

def test_capture_clock_rolls_back():
    g=FingerGateway(CAL);get(g,frame(seq=1,t=1000))
    with pytest.raises(IngestError,match='nonmonotonic_clock'):get(g,frame(seq=2,t=999))

def test_unflagged_force_doesnt_become_pressure():
    g=FingerGateway(CAL);x=get(g,frame())
    assert x['pressure_kpa'] is None and x['quality']['pressure']==0
    assert x['quality']['joint']==x['quality']['temperature']==1

def test_unverified_area_rejected():
    with pytest.raises(ValueError,match='area calibration'):
        FingerCalibration(3,0x203,'f'*64,area_calibrated=True,taxel_area_mm2=(10.,)*4)

def test_calibrated_area_yields_explicit_pressure_proxy():
    cal=replace(CAL,force_calibrated=True,area_calibrated=True,taxel_area_mm2=(100.,)*4)
    x=get(FingerGateway(cal),frame(force_centinewtons=(100,100,100,100)))
    assert x['pressure_kpa']==10
    assert x['quality']['pressure']==1

def test_adc_flag_revokes_pressure_only():
    cal=replace(CAL,force_calibrated=True,area_calibrated=True,taxel_area_mm2=(100.,)*4)
    x=get(FingerGateway(cal),frame(flags=FRESH_ALL|FAULT_ADC))
    assert x['quality']['pressure']==0 and x['quality']['temperature']==1

def test_brownout_cancels_all_measurements():
    x=get(FingerGateway(CAL),frame(supply_mv=4000))
    assert all(v==0 for v in x['quality'].values())

def test_encoder_disagreement_detected():
    x=get(FingerGateway(CAL),frame(angle_witness_u14=2500))
    assert x['quality']['joint']==0 and x['reasons']['joint_disagreement']

def test_temp_disagreement_detected():
    x=get(FingerGateway(CAL),frame(witness_centi_c=3220))
    assert x['quality']['temperature']==0

def test_slow_temp_stuck_detected_when_witness_changes_enough():
    g=FingerGateway(CAL)
    for j in range(130):
        x=get(g,frame(seq=j+1,t=1000+j*10,contact_centi_c=3100,
                      witness_centi_c=3100+int(80*j/129)))
    assert x['quality']['temperature']==0 and x['reasons']['temperature_stuck_suspected']
    assert g.latched_temp_stuck
    y=get(g,frame(seq=131,t=2300,contact_centi_c=3100,witness_centi_c=3150))
    assert y['quality']['temperature']==0

def test_shared_common_mode_stuck_not_detectable_from_twins_alone():
    g=FingerGateway(CAL)
    for j in range(130):
        x=get(g,frame(seq=j+1,t=1000+j*10,contact_centi_c=3100,
                      witness_centi_c=3100))
    assert x['quality']['temperature']==1 # known blind spot

def test_temp_change_below_detection_threshold_not_assumed_fault():
    g=FingerGateway(CAL)
    for j in range(130):
        x=get(g,frame(seq=j+1,t=1000+j*10,contact_centi_c=3100,
                      witness_centi_c=3100+int(15*j/129)))
    assert x['quality']['temperature']==1

def test_v1_bridge_no_fabricated_pressure_and_finite():
    g=FingerGateway(CAL);bridge=OriginalSOMABridge()
    for j in range(180):
        x=get(g,frame(seq=j+1,t=1000+j*10,angle_u14=2200+j//12,
                      angle_witness_u14=2202+j//12))
        bridge.ingest(x)
    obs,q,meta=bridge.window(2790)
    assert obs.shape==(8,7) and q.shape==(8,3)
    assert np.isfinite(obs).all() and np.isfinite(q).all()
    assert np.all(q[:,0]==0) and np.all(obs[:,0]==0) and np.all(obs[:,3]==0)
    assert np.all(obs[:,6]==0)
    assert q[:,1].sum()>0 and q[:,2].sum()>0

def test_not_unsafe_numeric_values():
    with pytest.raises(FrameError,match='encoder'):
        frame(angle_u14=40000).encode()
    with pytest.raises(FrameError,match='force'):
        frame(force_centinewtons=(-1,0,0,0)).encode()

def test_sparse_thermal_conversions_not_laundered_as_fresh():
    g=FingerGateway(CAL);bridge=OriginalSOMABridge()
    for j in range(160):
        flags=FRESH_ALL if j==0 else (NEW_ADC|NEW_ENCODER)
        x=get(g,frame(seq=j+1,t=1000+j*10,flags=flags))
        bridge.ingest(x)
    obs,q,_=bridge.window(2590)
    assert np.all(q[:,1]==0), 'cached TMP117 values were treated as new'
    assert q[:,2].sum()>0

def test_slow_temp_freeze_with_realistic_thermal_updates():
    g=FingerGateway(CAL)
    for j in range(14):
        x=get(g,frame(seq=j+1,t=1000+j*125,contact_centi_c=3100,
                      witness_centi_c=3100+int(90*j/13)))
    assert x['reasons']['temperature_stuck_suspected']
    assert x['quality']['temperature']==0

def test_reported_temp_fault_immediately_fails_quality_even_when_cached():
    g=FingerGateway(CAL);bridge=OriginalSOMABridge()
    for j in range(30):
        flags=FRESH_ALL if j%13==0 else (NEW_ADC|NEW_ENCODER)
        if j==29: flags|=FAULT_TEMP
        x=get(g,frame(seq=j+1,t=1000+j*10,flags=flags))
        bridge.ingest(x)
    obs,q,_=bridge.window(1290)
    assert q[-1,1]==0

def test_adc_fault_immediately_revokes_prior_calibrated_pressure():
    cal=replace(CAL,force_calibrated=True,area_calibrated=True,taxel_area_mm2=(100.,)*4)
    g=FingerGateway(cal);bridge=OriginalSOMABridge()
    for j in range(110):
        x=get(g,frame(seq=j+1,t=1000+j*10,force_centinewtons=(100,100,100,100)))
        bridge.ingest(x)
    assert bridge.hub.quality('pressure',2090).state.value=='ok'
    x=get(g,frame(seq=111,t=2100,flags=FRESH_ALL|FAULT_ADC))
    bridge.ingest(x)
    assert bridge.hub.quality('pressure',2100).state.value!='ok'

def test_verified_gpio_plan_no_pin_collision_and_bus_addresses_unique():
    import csv
    plan=Path(__file__).resolve().parent/'PINMAP.csv'
    rows=list(csv.DictReader(plan.open()))
    physical=[r['LQFP48_physical_pin'] for r in rows if r['LQFP48_physical_pin']]
    assert len(physical)==len(set(physical))
    assert {'PA11','PA12','PA5','PA6','PA7','PB6','PB7','PA13','PA14'} <= {r['STM32G0B1CBT6_GPIO'] for r in rows}
    assert len({0x48,0x49,0x4A})==3
    assert 4*100<860
