"""CPU-only targeted wire/sensor fault injection. Not a reliability estimate."""
import json
from pathlib import Path
import random
from dataclasses import replace
from packet import FingerFrame,FRESH_ALL,NEW_ADC,NEW_ENCODER,NEW_TEMP,FAULT_TEMP,FAULT_ADC
from gateway import FingerGateway,FingerCalibration,IngestError

CAL=FingerCalibration(3,0x203,'e'*64,trusted_clock_offset_ms=0,
                      force_calibrated=True,area_calibrated=True,taxel_area_mm2=(100.,)*4)

def frame(seq,t,**extra):
    kwargs=dict(node_id=3,seq=seq,capture_ms=t,force_centinewtons=(100,)*4,
        angle_u14=1000,angle_witness_u14=1000,contact_centi_c=3100,
        witness_centi_c=3100,supply_mv=5000,flags=FRESH_ALL)
    kwargs.update(extra)
    return FingerFrame(**kwargs)

def case(name,seed):
    rng=random.Random(seed);g=FingerGateway(CAL);out=None
    if name=='clean':
        for j in range(20):
            out=g.ingest(frame(j+1,1000+j*10).encode(),can_id=0x203,received_ms=1006+j*10)
        return False,'no_fault'
    if name=='noisy_nominal':
        for j in range(20):
            delta=rng.randint(-5,5)
            f=frame(j+1,1000+j*10,angle_u14=1000+delta,angle_witness_u14=1001+delta,
                    contact_centi_c=3100+rng.randint(-8,8),witness_centi_c=3100+rng.randint(-8,8))
            out=g.ingest(f.encode(),can_id=0x203,received_ms=1006+j*10)
        return False,'no_fault'
    start=frame(1,1000)
    g.ingest(start.encode(),can_id=0x203,received_ms=1006)
    f=frame(2,1010)
    if name=='crc_corruption':
        data=bytearray(f.encode());data[13]^=4;data=bytes(data)
    elif name=='replay':data=start.encode()
    elif name=='stale':
        try:g.ingest(f.encode(),can_id=0x203,received_ms=1190)
        except IngestError:return True,'rejected'
        return False,'missed'
    elif name=='can_id_swap':
        try:g.ingest(f.encode(),can_id=0x204,received_ms=1016)
        except IngestError:return True,'rejected'
        return False,'missed'
    elif name=='brownout':f=replace(f,supply_mv=4000);data=f.encode()
    elif name=='encoder_witness_disagreement':
        f=replace(f,angle_witness_u14=1450);data=f.encode()
    elif name=='thermal_witness_disagreement':
        f=replace(f,witness_centi_c=3360);data=f.encode()
    elif name=='adc_reported_fault':f=replace(f,flags=FRESH_ALL|FAULT_ADC);data=f.encode()
    elif name=='temp_reported_fault':f=replace(f,flags=FRESH_ALL|FAULT_TEMP);data=f.encode()
    elif name=='temperature_freeze_visible_to_witness':
        # Nearly 1.5s of legitimate 8Hz TMP117 conversions, main frozen.
        for j in range(2,14):
            t=1000+(j-1)*125
            f=frame(j,t,contact_centi_c=3100,witness_centi_c=3100+int(j*7.5))
            out=g.ingest(f.encode(),can_id=0x203,received_ms=t+6)
        return bool(out['reasons']['temperature_stuck_suspected']),'temporal_cross_check'
    elif name=='temperature_freeze_below_detection':
        for j in range(2,14):
            t=1000+(j-1)*125
            f=frame(j,t,contact_centi_c=3100,witness_centi_c=3100+int(j*2))
            out=g.ingest(f.encode(),can_id=0x203,received_ms=t+6)
        return bool(out['reasons']['temperature_stuck_suspected']),'undetected_by_design'
    elif name=='both_thermal_sensors_frozen':
        for j in range(2,14):
            t=1000+(j-1)*125
            f=frame(j,t,contact_centi_c=3100,witness_centi_c=3100)
            out=g.ingest(f.encode(),can_id=0x203,received_ms=t+6)
        return bool(out['reasons']['temperature_stuck_suspected']),'common_mode_blindspot'
    elif name=='forge_valid_crc_and_identity':
        # Without authenticated transport, a spoof can remain numerically plausible.
        data=f.encode()
        out=g.ingest(data,can_id=0x203,received_ms=1016)
        return False,'not_authenticated'
    else:raise ValueError(name)
    try:out=g.ingest(data,can_id=0x203,received_ms=1016)
    except IngestError:return True,'rejected'
    detected=out['quality']!=dict.fromkeys(out['quality'],1)
    return detected,'quality_rule'

def main():
    names=['clean','noisy_nominal','crc_corruption','replay','stale','can_id_swap',
           'brownout','encoder_witness_disagreement','thermal_witness_disagreement',
           'adc_reported_fault','temp_reported_fault','temperature_freeze_visible_to_witness',
           'temperature_freeze_below_detection','both_thermal_sensors_frozen',
           'forge_valid_crc_and_identity']
    results=[]
    for i,name in enumerate(names):
        flags=[case(name,1000+3*i+k)[0] for k in range(100)]
        results.append({'scenario':name,'trials':100,'flagged':sum(flags),
                        'flagged_rate':sum(flags)/100,
                        'note':'scripted detection only; not hardware validation'})
    out={'label':'SYNTHETIC FAULT INJECTION: 1,500 targeted trials, not population sensitivity',
         'cases':results,'limitations':[
             'clean/noise example set not a field false-positive rate',
             'correlated frozen twins remain undetectable without external stimuli',
             'small temperature changes can evade drift monitor',
             'CRC/CAN arbitration ID do not authenticate node identity',
             'multiple isolated scenarios chosen to exercise implemented rules, not real failure distribution']}
    path=Path(__file__).resolve().parent/'fault_results.json'
    path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
