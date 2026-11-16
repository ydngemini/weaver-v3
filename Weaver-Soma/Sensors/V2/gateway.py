"""Bench-only conversion of verified sensor frames to SOMA V1/V2 input.

No cryptographic device identity implementation. The caller MUST provision a
trusted (node_id, CAN identifier, calibration digest, clock mapping) out of band;
this module only checks expected values for consistency. Never an actuator guard.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import math
from typing import Optional
from packet import (FingerFrame,FrameError,FAULT_ADC,FAULT_TEMP,
                    FAULT_ENCODER,FAULT_SUPPLY,FAULT_CALIBRATION,NEW_ADC,NEW_TEMP,NEW_ENCODER)

@dataclass(frozen=True)
class FingerCalibration:
    node_id:int
    expected_can_id:int
    calibration_digest:str
    min_joint_deg:float=0.
    max_joint_deg:float=90.
    force_calibrated:bool=False
    area_calibrated:bool=False
    taxel_area_mm2:tuple[float,float,float,float]|None=None
    # Node milliseconds must be mapped to gateway monotonic milliseconds by a
    # TRUSTED synchronization procedure, not simply trusting unsigned metadata.
    trusted_clock_offset_ms:int|None=None

    def __post_init__(self):
        if self.node_id<1 or self.node_id>250:raise ValueError('invalid node_id')
        if self.expected_can_id<0 or self.expected_can_id>0x1fffffff:
            raise ValueError('invalid CAN arbitration ID')
        if len(self.calibration_digest)!=64 or any(x not in '0123456789abcdef' for x in self.calibration_digest):
            raise ValueError('expected lowercase 64 character SHA256 calibration digest')
        if not (0<=self.min_joint_deg<self.max_joint_deg<=360):raise ValueError('invalid joint stops')
        if self.area_calibrated and not self.force_calibrated:
            raise ValueError('area calibration alone cannot make force valid')
        if self.area_calibrated and (self.taxel_area_mm2 is None or len(self.taxel_area_mm2)!=4 or
            any(not math.isfinite(x) or x<=0 for x in self.taxel_area_mm2)):
            raise ValueError('validated positive contact area required')

class IngestError(ValueError):pass

class FingerGateway:
    def __init__(self,cal:FingerCalibration):
        self.cal=cal
        self.last_seq:Optional[int]=None
        self.last_capture:Optional[int]=None
        self.last_thermal:deque[tuple[int,int,int]]=deque(maxlen=300)
        self.latched_temp_stuck=False  # needs trusted local recommissioning
        self.rejected: list[str]=[]
        self.accepted=0

    def ingest(self,payload:bytes,*,can_id:int,received_ms:int)->dict:
        try:
            if can_id != self.cal.expected_can_id:raise IngestError('unexpected_can_id')
            if self.cal.trusted_clock_offset_ms is None:raise IngestError('untrusted_clock')
            frame=FingerFrame.decode(payload)
            if frame.node_id!=self.cal.node_id:raise IngestError('unexpected_node_id')
            capture_mapped=frame.capture_ms+self.cal.trusted_clock_offset_ms
            if capture_mapped>received_ms+3:raise IngestError('future_timestamp')
            if received_ms-capture_mapped>90:raise IngestError('stale_packet')
            if self.last_seq is not None and frame.seq<=self.last_seq:
                raise IngestError('replay_or_restart_requires_recommission')
            if self.last_capture is not None and frame.capture_ms<=self.last_capture:
                raise IngestError('nonmonotonic_clock')
            if self.last_capture is not None and frame.capture_ms-self.last_capture>150:
                # Do not use stale old values to extrapolate missing samples.
                # The next accepted frame is still usable, but gap is flagged.
                gap=True
            else:gap=False
        except (FrameError,IngestError) as e:
            self.rejected.append(str(e))
            raise IngestError(str(e)) from e

        self.last_seq=frame.seq
        self.last_capture=frame.capture_ms
        self.accepted+=1
        bad_supply=bool((frame.flags&(FAULT_SUPPLY|FAULT_CALIBRATION)) or not 4750<=frame.supply_mv<=5250)
        fault_force=bad_supply or bool(frame.flags&FAULT_ADC) or not self.cal.force_calibrated
        fault_temp=bad_supply or bool(frame.flags&FAULT_TEMP)
        fault_joint=bad_supply or bool(frame.flags&FAULT_ENCODER)

        # An independent low-dimensional corroboration check: contact and
        # witness temp must be co-located, not simply at two different body sites.
        if frame.flags&NEW_TEMP:
            self.last_thermal.append((frame.capture_ms,frame.contact_centi_c,frame.witness_centi_c))
        while self.last_thermal and frame.capture_ms-self.last_thermal[0][0]>1500:
            self.last_thermal.popleft()
        stuck_temperature=False
        if len(self.last_thermal)>=5:
            p0,p1=self.last_thermal[0],self.last_thermal[-1]
            if p1[0]-p0[0]>=800:
                drift_main=(p1[1]-p0[1])/100.
                drift_witness=(p1[2]-p0[2])/100.
                if abs(drift_main)<.035 and abs(drift_witness)>.65:
                    stuck_temperature=True
        if abs(frame.contact_centi_c-frame.witness_centi_c)>150:
            # Separate assemblies are not equivalent thermal witnesses.
            # This criterion is only appropriate after co-location validation.
            fault_temp=True
        if stuck_temperature:self.latched_temp_stuck=True
        if self.latched_temp_stuck:fault_temp=True
        if any(not 0<=x<=15000 for x in frame.force_centinewtons):fault_force=True

        def joint_frac(raw):
            degrees=raw*(360./16384.)
            # A small mechanical tolerance is allowed; out-of-range is not
            # silently clamped into a valid reading.
            if degrees<self.cal.min_joint_deg-2 or degrees>self.cal.max_joint_deg+2:
                return 0.,False
            return (degrees-self.cal.min_joint_deg)/(self.cal.max_joint_deg-self.cal.min_joint_deg),True
        primary,primary_ok=joint_frac(frame.angle_u14)
        witness,witness_ok=joint_frac(frame.angle_witness_u14)
        joint_good=not fault_joint and primary_ok and witness_ok and abs(primary-witness)<=.055

        pressure_kpa=None
        if (self.cal.force_calibrated and self.cal.area_calibrated
            and self.cal.taxel_area_mm2 and not fault_force):
            # Pressure proxy, only valid for a precharacterized effective patch.
            force_n=sum(frame.force_centinewtons)/100.
            active_area=sum(a for a,v in zip(self.cal.taxel_area_mm2,frame.force_centinewtons) if v>0)
            if active_area>0:pressure_kpa=1000*force_n/active_area
            if pressure_kpa is None or pressure_kpa>100 or pressure_kpa<0:
                fault_force=True;pressure_kpa=None

        return {'capture_ms':capture_mapped,'seq':frame.seq,'gap':gap,
                'force_centinewtons':frame.force_centinewtons,
                'pressure_kpa':pressure_kpa,'joint_fraction':primary,
                'witness_joint_fraction':witness,
                'contact_c':frame.contact_centi_c/100,
                'witness_c':frame.witness_centi_c/100,
                'fresh':{'pressure':bool(frame.flags&NEW_ADC),
                         'temperature':bool(frame.flags&NEW_TEMP),
                         'joint':bool(frame.flags&NEW_ENCODER)},
                'quality':{'pressure':int(pressure_kpa is not None and not fault_force),
                           'temperature':int(not fault_temp),
                           'joint':int(joint_good)},
                'reasons':{'temperature_stuck_suspected':self.latched_temp_stuck,
                           'joint_disagreement':abs(primary-witness)>.055,
                           'supply_fault':bad_supply,
                           'frame_gap':gap},
                'calibration_digest':self.cal.calibration_digest}