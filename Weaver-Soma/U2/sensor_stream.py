"""Bench / physical capture gateway for SOMA-SENSE V2 -> U2.

Only physical mode writes research data when separately provisioned actual
calibration and hardware evidence are present. CRC does NOT authenticate hardware.
It never trains from a packet by itself and never issues motor actions.
"""
from __future__ import annotations
from pathlib import Path
import json,sys,time,hashlib,datetime
import numpy as np
from typing import Any

THIS=Path(__file__).resolve().parent
SENSORS=THIS.parent/'Sensors'
sys.path[:0]=[str(SENSORS/'V2'),str(SENSORS)]
from gateway import FingerCalibration,FingerGateway,IngestError
from bridge import OriginalSOMABridge

class CaptureSession:
    def __init__(self,provision:dict[str,Any],*,source:str,output:Path|None=None):
        if source not in ('synthetic','hardware'):raise ValueError('explicit source required')
        self.source=source
        # Hardware mode cannot be spoofed by selecting a CLI flag alone.
        if source=='hardware':
            required=('operator_signed_off','contact_area_mm2_verified',
                      'force_calibration_verified','timebase_provenance_verified',
                      'device_identity_verified','reference_sensor_calibration_verified')
            if not all(provision.get(k) is True for k in required):
                raise ValueError('hardware capture requires independently verified calibration & provenance checklist')
            if not provision.get('calibration_reference_id'):
                raise ValueError('hardware capture requires a traceable calibration reference ID')
        cal=FingerCalibration(
            node_id=int(provision['node_id']),
            expected_can_id=int(provision['expected_can_id']),
            calibration_digest=str(provision['calibration_digest']),
            trusted_clock_offset_ms=int(provision['trusted_clock_offset_ms']) if provision.get('trusted_clock_offset_ms') is not None else None,
            force_calibrated=bool(provision.get('force_calibration_verified',False)),
            area_calibrated=bool(provision.get('contact_area_mm2_verified',False)),
            taxel_area_mm2=tuple(provision['taxel_area_mm2']) if provision.get('taxel_area_mm2') else None)
        self.gateway=FingerGateway(cal)
        self.bridge=OriginalSOMABridge()
        self.meta={'source':source,'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'calibration_digest':cal.calibration_digest,'calibration_reference_id':provision.get('calibration_reference_id'),
                   'is_physical_measurement':source=='hardware'}
        self.frames=0
        self.output=Path(output) if output else None
        if self.output:
            self.output.parent.mkdir(parents=True,exist_ok=True)
            if self.output.exists():raise FileExistsError('never overwrite an existing capture')
            self.output.write_text(json.dumps({'type':'session',**self.meta})+'\n')

    def ingest(self,payload:bytes,*,can_id:int,received_ms:int)->dict:
        frame=self.gateway.ingest(payload,can_id=can_id,received_ms=received_ms)
        self.bridge.ingest(frame)
        self.frames+=1
        record={'type':'sensor_sample','capture_ms':int(frame['capture_ms']),
                'sequence':int(frame['seq']),'quality':frame['quality'],'fresh':frame['fresh'],
                'calibration_digest':frame['calibration_digest'],
                'pressure_kpa':frame['pressure_kpa'],'temperature_c':frame['contact_c'],
                'joint_fraction':frame['joint_fraction'],'reasons':frame['reasons']}
        if self.output:
            with self.output.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        return record

    def latest_u2_window(self,end_ms:int):
        obs,q,meta=self.bridge.window(end_ms)
        assert obs.shape==(8,7) and q.shape==(8,3)
        return obs,q,meta

    def latest_u2_sequence(self,ends_ms:tuple[int,...]):
        if len(ends_ms)!=4 or any(b<=a for a,b in zip(ends_ms,ends_ms[1:])):
            raise ValueError('require four increasing complete window end times')
        windows=[self.latest_u2_window(t) for t in ends_ms]
        return np.stack([w[0] for w in windows]),np.stack([w[1] for w in windows]),[w[2] for w in windows]

    @staticmethod
    def write_example_provision(path:Path):
        """NO real calibration claims. Hardware mode will refuse this template."""
        payload={'node_id':3,'expected_can_id':515,'calibration_digest':'0'*64,
                 'trusted_clock_offset_ms':0,'operator_signed_off':False,
                 'contact_area_mm2_verified':False,'force_calibration_verified':False,
                 'device_identity_verified':False,'timebase_provenance_verified':False,
                 'reference_sensor_calibration_verified':False,
                 'calibration_reference_id':None,'taxel_area_mm2':None}
        Path(path).write_text(json.dumps(payload,indent=2)+'\n')