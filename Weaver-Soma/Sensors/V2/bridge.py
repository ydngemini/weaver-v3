"""Strict bridge: V2 readings -> original V1 SensorHub (never fabricate kPa)."""
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
# GitHub layout: Weaver-Soma/Sensors/V2/ next to Sensors/sensor_hub.py.
# Local research layout: soma_sensor_v2/ next to soma_sensor_v1/.
CANDIDATES=(HERE.parent,HERE.parent/'soma_sensor_v1')
for folder in CANDIDATES:
    if (folder/'sensor_hub.py').exists():
        sys.path.insert(0,str(folder)); break
else:
    raise ImportError('SOMA-SENSE V1 sensor_hub.py is required next to V2 or in sibling soma_sensor_v1/')
from sensor_hub import SensorHub,Sample

class OriginalSOMABridge:
    def __init__(self):
        self.hub=SensorHub()
    def ingest(self,reading:dict):
        t=reading['capture_ms'];seq=reading['seq'];q=reading['quality'];fresh=reading['fresh']
        values=[('pressure',reading['pressure_kpa'] if reading['pressure_kpa'] is not None else 0.,'kPa',bool(q['pressure'])),
                ('joint',reading['joint_fraction'],'fraction',bool(q['joint'])),
                ('joint_ref',reading['witness_joint_fraction'],'fraction',bool(q['joint'])),
                ('temperature',reading['contact_c'],'C',bool(q['temperature'])),
                ('temperature_ref',reading['witness_c'],'C',bool(q['temperature']))]
        for ch,val,unit,good in values:
            source='pressure' if ch=='pressure' else ('temperature' if 'temperature' in ch else 'joint')
            if not fresh[source] and good:
                # Preserve acquisition time; do not turn a cached reading into
                # a new measurement at each CAN-FD transmit timestamp.
                continue
            expected=self.hub.specs[ch].expected_calibration
            self.hub.ingest(Sample(ch,float(val),unit,int(t),int(t),int(seq),expected,source_ok=good))
    def window(self,end_ms:int):
        obs,quality,meta=self.hub.to_soma_window(end_ms=end_ms,period_ms=120,frames=8)
        assert obs.shape==(8,7) and quality.shape==(8,3)
        assert np.isfinite(obs).all() and np.isfinite(quality).all()
        return obs,quality,meta