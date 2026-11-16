"""SOMA-SENSE V2 wire frame. Synthetic/bench-only protocol, no hardware control.

CAN-FD: 32 data bytes, little endian, fixed point. CAN-FD's link CRC protects the
bus; this CRC-16 is *additional accidental-corruption detection*, NOT authentication.
The trusted gateway must separately provision device identity & calibration.
"""
from __future__ import annotations
from dataclasses import dataclass
import struct
from typing import ClassVar

VERSION=2
PAYLOAD=struct.Struct('<BBHII4HHHhhH')
FRAME=struct.Struct('<BBHII4HHHhhHH')
assert PAYLOAD.size==30 and FRAME.size==32

FAULT_ADC=1<<0
FAULT_TEMP=1<<1
FAULT_ENCODER=1<<2
FAULT_SUPPLY=1<<3
FAULT_CALIBRATION=1<<4
NEW_ADC=1<<8
NEW_TEMP=1<<9
NEW_ENCODER=1<<10
FRESH_ALL=NEW_ADC|NEW_TEMP|NEW_ENCODER
KNOWN_FLAGS=FAULT_ADC|FAULT_TEMP|FAULT_ENCODER|FAULT_SUPPLY|FAULT_CALIBRATION|FRESH_ALL

class FrameError(ValueError):
    pass

def crc16(data:bytes)->int:
    """CRC-16/CCITT-FALSE init 0xffff, polynomial 0x1021, xorout 0x0000."""
    crc=0xffff
    for b in data:
        crc^=b<<8
        for _ in range(8):
            crc=((crc<<1)^0x1021 if crc&0x8000 else crc<<1)&0xffff
    return crc

@dataclass(frozen=True)
class FingerFrame:
    node_id:int
    seq:int
    capture_ms:int
    force_centinewtons:tuple[int,int,int,int]
    angle_u14:int
    angle_witness_u14:int
    contact_centi_c:int
    witness_centi_c:int
    supply_mv:int
    flags:int=0
    version:ClassVar[int]=VERSION

    def _fields(self):
        if not 1<=self.node_id<=250:raise FrameError('node_id outside provisioned range')
        if not 0<=self.seq<=0xffffffff:raise FrameError('invalid seq')
        if not 0<=self.capture_ms<=0xffffffff:raise FrameError('invalid capture_ms')
        if len(self.force_centinewtons)!=4:raise FrameError('requires four calibrated taxels')
        if any(not 0<=x<=65535 for x in self.force_centinewtons):raise FrameError('force out of range')
        if any(not 0<=x<16384 for x in (self.angle_u14,self.angle_witness_u14)):
            raise FrameError('encoder code must be 14-bit')
        if any(not -32768<=x<=32767 for x in (self.contact_centi_c,self.witness_centi_c)):
            raise FrameError('temperature out of wire range')
        if not 0<=self.supply_mv<=65535:raise FrameError('supply outside wire range')
        if self.flags&~KNOWN_FLAGS:raise FrameError('unknown flags')
        return (VERSION,self.node_id,self.flags,self.seq,self.capture_ms,
                *self.force_centinewtons,self.angle_u14,self.angle_witness_u14,
                self.contact_centi_c,self.witness_centi_c,self.supply_mv)

    def encode(self)->bytes:
        payload=PAYLOAD.pack(*self._fields())
        return payload+struct.pack('<H',crc16(payload))

    @classmethod
    def decode(cls,data:bytes)->'FingerFrame':
        if len(data)!=FRAME.size:raise FrameError('incorrect CAN-FD payload size')
        if crc16(data[:-2])!=struct.unpack_from('<H',data,30)[0]:
            raise FrameError('crc_mismatch')
        fields=FRAME.unpack(data)
        if fields[0]!=VERSION:raise FrameError('unsupported_version')
        obj=cls(node_id=fields[1], flags=fields[2],seq=fields[3],capture_ms=fields[4],
                force_centinewtons=tuple(fields[5:9]),angle_u14=fields[9],
                angle_witness_u14=fields[10],contact_centi_c=fields[11],
                witness_centi_c=fields[12],supply_mv=fields[13])
        obj._fields()
        return obj
