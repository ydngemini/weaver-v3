"""SOMA-SENSE v1: a fail-closed, advisory-only sensor validation and U1/U2 bridge.

Zero actuator access, no trained fault classifier. Thresholds are simulated bench
research defaults; real hardware requires qualification and calibration.
"""
from __future__ import annotations
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import math
import numpy as np

class State(str, Enum):
    OK = 'ok'
    SUSPECT = 'suspect'
    INVALID = 'invalid'

@dataclass(frozen=True)
class ChannelSpec:
    channel: str
    unit: str
    minimum: float
    maximum: float
    max_age_ms: int
    expected_calibration: str
    partner: Optional[str] = None
    max_partner_disagreement: Optional[float] = None
    max_rate_per_second: Optional[float] = None
    # Diagnostics only; NOT independently certified physical safety limits.

@dataclass(frozen=True)
class Sample:
    channel: str
    value: float
    unit: str
    capture_ms: int
    received_ms: int
    seq: int
    calibration_id: str
    source_ok: bool = True

@dataclass(frozen=True)
class ChannelHealth:
    state: State
    value: float
    age_ms: int
    reasons: tuple[str, ...]
    source_capture_ms: Optional[int] = None

SPECS = {
    'pressure': ChannelSpec('pressure','kPa',0.,100.,90,'pressure-cal-v1','pressure_ref',6.,150.),
    'pressure_ref': ChannelSpec('pressure_ref','kPa',0.,100.,90,'pressure-ref-cal-v1','pressure',6.,150.),
    'temperature': ChannelSpec('temperature','C',-10.,75.,350,'temp-cal-v1','temperature_ref',1.5,30.),
    'temperature_ref': ChannelSpec('temperature_ref','C',-10.,75.,350,'temp-ref-cal-v1','temperature',1.5,30.),
    'joint': ChannelSpec('joint','fraction',0.,1.,90,'joint-cal-v1','joint_ref',.055,3.5),
    'joint_ref': ChannelSpec('joint_ref','fraction',0.,1.,90,'joint-ref-cal-v1','joint',.055,3.5),
}
PRIMARY = ('pressure','temperature','joint')

class SensorHub:
    def __init__(self, specs: Optional[dict[str, ChannelSpec]] = None, max_samples: int = 500):
        self.specs = dict(SPECS if specs is None else specs)
        self.samples: dict[str, deque[Sample]] = {k:deque(maxlen=max_samples) for k in self.specs}
        self.rejections: list[tuple[str,str,int]] = []
        self.quarantine: dict[str,tuple[int,str]] = {}

    def ingest(self, s: Sample) -> bool:
        if s.channel not in self.specs:
            self.rejections.append((s.channel,'unknown_channel',s.seq));return False
        spec = self.specs[s.channel]
        prior = self.samples[s.channel][-1] if self.samples[s.channel] else None
        problems=[]
        if s.unit != spec.unit: problems.append('unit_mismatch')
        if s.calibration_id != spec.expected_calibration: problems.append('calibration_mismatch')
        if not math.isfinite(s.value): problems.append('nonfinite')
        if s.capture_ms < 0 or s.received_ms < 0 or s.capture_ms > s.received_ms+3:
            problems.append('invalid_timing')
        if s.received_ms - s.capture_ms > max(3*spec.max_age_ms,1000):
            problems.append('delayed_packet')
        if prior is not None:
            if s.seq <= prior.seq:problems.append('replay_or_reset_seq')
            if s.capture_ms <= prior.capture_ms:problems.append('nonmonotonic_capture')
        if problems:
            self.rejections.extend((s.channel,x,s.seq) for x in problems)
            self.quarantine[s.channel] = (s.received_ms, problems[0])
            return False
        self.quarantine.pop(s.channel,None)
        self.samples[s.channel].append(s)
        return True

    def _prior(self, channel:str, at_ms:int) -> Optional[Sample]:
        items=self.samples[channel]
        if not items:return None
        capture=[s.capture_ms for s in items]
        pos=bisect_right(capture,at_ms)-1
        return items[pos] if pos>=0 else None

    def quality(self, channel:str, at_ms:int) -> ChannelHealth:
        sp=self.specs[channel]
        s=self._prior(channel,at_ms)
        if s is None:
            return ChannelHealth(State.INVALID,0.,-1,('no_sample',))
        age=at_ms-s.capture_ms
        reasons=[]
        if age>sp.max_age_ms: reasons.append('stale')
        if channel in self.quarantine and self.quarantine[channel][0]<=at_ms:
            reasons.append('packet_rejected_'+self.quarantine[channel][1])
        if not s.source_ok:reasons.append('source_reported_fault')
        if not (sp.minimum <= s.value <= sp.maximum):reasons.append('out_of_range')
        if reasons:return ChannelHealth(State.INVALID,0.,age,tuple(reasons),s.capture_ms)
        # Unit, sequence, clock and calibration checks occurred at ingestion.
        # A plausible stuck reading is not detectable without corroboration or excitation.
        if sp.partner and sp.max_partner_disagreement is not None:
            ref=self._prior(sp.partner,at_ms)
            if ref is not None:
                ref_spec=self.specs[sp.partner]
                if (ref.source_ok and abs(ref.capture_ms-s.capture_ms)<=min(sp.max_age_ms,ref_spec.max_age_ms)
                    and at_ms-ref.capture_ms<=ref_spec.max_age_ms
                    and ref_spec.minimum<=ref.value<=ref_spec.maximum):
                    if abs(ref.value-s.value)>sp.max_partner_disagreement:
                        reasons.append('independent_disagreement')
            # Missing partner is an uncertainty, NOT evidence of sensor correctness.
        # Very rapid changes can be a real impact. Treat as suspect, never 'proof' of failure.
        recent=[x for x in self.samples[channel] if x.capture_ms<=at_ms]
        if len(recent)>=2 and sp.max_rate_per_second is not None:
            # use last two distinct samples, even if the first is stale
            before=recent[-2];dt=(s.capture_ms-before.capture_ms)/1000.
            if dt>0 and abs(s.value-before.value)/dt > sp.max_rate_per_second:
                reasons.append('implausible_rate')
        status=State.SUSPECT if reasons else State.OK
        return ChannelHealth(status,s.value if not reasons else 0.,age,tuple(reasons),s.capture_ms)

    def snapshot(self, at_ms:int) -> dict[str,ChannelHealth]:
        return {name:self.quality(name,at_ms) for name in self.specs}

    def to_soma_window(self, end_ms:int, *, period_ms:int=120, frames:int=8):
        """Returns finite obs[8,7] and conservative q[8,3] for existing U1/U2.

        Derivatives are zero and quality=0 if either timestamp's sample is invalid.
        U1 contact flag is unmasked internally, so it must be zero if pressure is invalid.
        """
        if period_ms<=0 or frames<2:raise ValueError('invalid window specification')
        stamps=[end_ms-(frames-i)*period_ms for i in range(frames+1)]
        hist=[self.snapshot(t) for t in stamps]
        obs=np.zeros((frames,7),dtype=np.float32)
        quality=np.zeros((frames,3),dtype=np.float32)
        reason_log=[]
        for i in range(frames):
            issues={}
            for j,ch in enumerate(PRIMARY):
                old=hist[i][ch];new=hist[i+1][ch]
                if old.state is State.OK and new.state is State.OK:
                    quality[i,j]=1.
                    obs[i,j]=new.value
                    obs[i,j+3]=(new.value-old.value)/(period_ms/1000.)
                else:
                    issues[ch]=tuple(sorted(set(old.reasons+new.reasons))) or ('not_verified',)
            if quality[i,0]>0:
                obs[i,6]=float(obs[i,0]>8.)
            reason_log.append(issues)
        assert np.isfinite(obs).all() and np.isfinite(quality).all()
        return obs,quality,{'timestamps_ms':stamps[1:],'issues':reason_log,'independent_checks':'timestamp,sequence,unit,calibration,bounds,pair_consistency'}