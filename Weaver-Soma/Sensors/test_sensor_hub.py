"""SOMA-SENSE CPU-only regression tests: no hardware, no robot control."""
import numpy as np
from sensor_hub import SensorHub,Sample,State,SPECS

def emit(h,channel,value,t,seq,source_ok=True):
    return h.ingest(Sample(channel,value,SPECS[channel].unit,t,t+2,seq,SPECS[channel].expected_calibration,source_ok))

def populate(h,missing=None,changes=None):
    changes=changes or {}
    for idx,t in enumerate(range(0,2001,20)):
        vals={'pressure':20+.002*t,'pressure_ref':20.1+.002*t,
              'temperature':32.+.001*t,'temperature_ref':32.1+.001*t,
              'joint':.25+.0001*t,'joint_ref':.251+.0001*t}
        vals.update({k:(v(t) if callable(v) else v) for k,v in changes.items()})
        for ch,v in vals.items():
            if ch==missing:continue
            accepted=emit(h,ch,v,t,idx)
            assert accepted or not np.isfinite(v)

def test_clean_window():
    h=SensorHub();populate(h)
    x,q,meta=h.to_soma_window(1920)
    assert x.shape==(8,7) and q.shape==(8,3)
    assert q.sum()==24 and np.isfinite(x).all()
    assert (x[:,6]==1).all()
    assert all(not x for x in meta['issues'])

def test_independently_detected_stuck_channel():
    h=SensorHub();populate(h,changes={'joint':lambda t:.25 if t>=900 else .25+.0001*t})
    obs,q,_=h.to_soma_window(1920)
    assert (q[-5:,2]==0).all(),q[:,2]
    assert (obs[:,2]==0).any()

def test_unreported_drift():
    h=SensorHub();populate(h,changes={'temperature':lambda t:32.+.001*t+4.0 if t>=700 else 32.+.001*t})
    assert h.quality('temperature',1500).state is State.SUSPECT
    _,q,_=h.to_soma_window(1920)
    assert q[:,1].sum()==0

def test_fault_lag_and_missing_samples():
    h=SensorHub();populate(h,missing='pressure')
    x,q,_=h.to_soma_window(1920)
    assert not q[:,0].any()
    assert not x[:,0].any() and not x[:,3].any() and not x[:,6].any()

def test_replay_rejected():
    h=SensorHub();assert emit(h,'joint',.4,100,10)
    assert not emit(h,'joint',.5,110,10)
    assert not emit(h,'joint',.5,90,11)
    assert emit(h,'joint',.5,120,11)

def test_corrupt_metadata_and_nonfinite_rejected():
    h=SensorHub();bad=Sample('temperature',float('nan'),'C',100,102,1,'temp-cal-v1')
    assert not h.ingest(bad)
    bad=Sample('temperature',20,'F',100,102,1,'temp-cal-v1')
    assert not h.ingest(bad)
    bad=Sample('temperature',20,'C',100,102,1,'old-cal')
    assert not h.ingest(bad)

def test_stale_and_self_reported_fault():
    h=SensorHub();assert emit(h,'joint',.3,100,1)
    assert h.quality('joint',300).state is State.INVALID
    assert emit(h,'joint',.3,320,2,False)
    assert h.quality('joint',320).state is State.INVALID

def test_suspicion_not_yet_proof():
    h=SensorHub();populate(h,missing='joint_ref',changes={'joint':lambda t:.36})
    assert h.quality('joint',1920).state is State.OK

def test_max_rate_spike():
    h=SensorHub()
    assert emit(h,'pressure',20,100,1)
    assert emit(h,'pressure_ref',20,100,1)
    assert emit(h,'pressure',80,120,2)
    assert h.quality('pressure',120).state is State.SUSPECT

def test_quality_masked_values_stay_finite():
    h=SensorHub();populate(h,changes={'pressure':lambda t:float('inf') if t==1000 else 22.})
    x,q,_=h.to_soma_window(1920)
    assert np.isfinite(x).all() and np.isfinite(q).all()

def test_rejected_packet_quarantines_channel_until_fresh_valid_sample():
    h=SensorHub(); assert emit(h,'joint',.3,100,1)
    assert not emit(h,'joint',float('nan'),110,2)
    assert h.quality('joint',112).state is State.INVALID
    assert emit(h,'joint',.31,120,2)
    assert h.quality('joint',120).state is State.OK

def test_out_of_order_packet_does_not_replace_previous_good_sample():
    h=SensorHub(); assert emit(h,'pressure',18.,100,1)
    assert not emit(h,'pressure',50.,80,2)
    assert h.quality('pressure',103).state is State.INVALID
    assert emit(h,'pressure',18.1,120,2)
    assert h.quality('pressure',120).state is State.OK
