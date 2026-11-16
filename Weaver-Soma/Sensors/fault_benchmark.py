"""Reproducible CPU synthetic fault-injection diagnostic. No real sensor accuracy claims."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sensor_hub import SensorHub,Sample,SPECS

SCENARIOS = (
    'clean','normal_sensor_noise', 'stuck_joint', 'stuck_pressure', 'stuck_temperature',
    'temperature_drift', 'pressure_dropout','replayed_joint','bad_calibration',
    'source_reported_joint_fault', 'joint_out_of_range','shared_common_mode_joint_stuck'
)
FAULTS=set(SCENARIOS)-{'clean','normal_sensor_noise'}

def trial(scenario:str,seed:int):
    rng=np.random.default_rng(seed)
    hub=SensorHub()
    offset_p=rng.normal(0,.3); offset_t=rng.normal(0,.15); offset_j=rng.normal(0,.01)
    for idx,t in enumerate(range(0,1981,20)):
        pressure=18+.006*t+offset_p
        temp=28+.0012*t+offset_t
        joint=.15+.00022*t+offset_j
        vals={
            'pressure':pressure+rng.normal(0,.025),
            'pressure_ref':pressure+.1+rng.normal(0,.02),
            'temperature':temp+rng.normal(0,.012),
            'temperature_ref':temp+.1+rng.normal(0,.012),
            'joint':joint+rng.normal(0,.001),
            'joint_ref':joint+.001+rng.normal(0,.001),
        }
        if scenario=='normal_sensor_noise':
            vals['pressure']+=rng.normal(0,.7)
            vals['temperature']+=rng.normal(0,.13)
            vals['joint']+=rng.normal(0,.006)
        if t>=900:
            if scenario=='stuck_joint': vals['joint']=.15+.00022*900+offset_j
            if scenario=='stuck_pressure': vals['pressure']=18+.006*900+offset_p
            if scenario=='stuck_temperature': vals['temperature']=28+.0012*900+offset_t
            if scenario=='temperature_drift': vals['temperature']+=4.
            if scenario=='joint_out_of_range':vals['joint']=1.2
            if scenario=='shared_common_mode_joint_stuck':
                vals['joint']=.15+.00022*900+offset_j
                vals['joint_ref']=vals['joint']+.001
        for ch,value in vals.items():
            if t>=900 and ch=='pressure' and scenario=='pressure_dropout': continue
            seq=idx
            if t>=900 and ch=='joint' and scenario=='replayed_joint':seq=44
            calibration=SPECS[ch].expected_calibration
            if t>=900 and ch=='temperature' and scenario=='bad_calibration':calibration='tampered-cal'
            ok=not (t>=900 and ch=='joint' and scenario=='source_reported_joint_fault')
            hub.ingest(Sample(ch,float(value),SPECS[ch].unit,t,t+2,seq,calibration,ok))
    obs,q,log=hub.to_soma_window(1920)
    monitored=any(q[-3:,j].min()==0 for j in range(3))
    # Source-only is an intentionally WEAK strawman, not an industry baseline.
    source_only=any(not sample.source_ok for x in ('pressure','temperature','joint') for sample in hub.samples[x] if sample.capture_ms>=1700)
    return {'scenario':scenario,'ground_truth_fault':scenario in FAULTS,'monitored_fault':bool(monitored),
            'source_only_fault':bool(source_only),'rejected_packets':len(hub.rejections),
            'end_sensor_quality':q[-1].tolist(),'issues_last':log['issues'][-1],
            'obs_finite':bool(np.isfinite(obs).all())}

def main(n_per_case=100):
    rows=[]
    for s in SCENARIOS:
        tr=[trial(s,923000+i) for i in range(n_per_case)]
        rows.append({'scenario':s,'fault':s in FAULTS,'trials':n_per_case,
                     'monitor_detection_rate':round(sum(x['monitored_fault'] for x in tr)/n_per_case,4),
                     'source_only_detection_rate':round(sum(x['source_only_fault'] for x in tr)/n_per_case,4),
                     'finite_rate':sum(x['obs_finite'] for x in tr)/n_per_case})
    return {'label':'SYNTHETIC SELF-CHECK ONLY (not calibrated sensor test)', 'seed_start':923000,
         'trials_per_case':n_per_case,'cases':rows,
         'caveats':['Uncorroborated slow drift can remain undetected',
                    'Both reference sensors drifting together can evade these rules',
                    'False alarms are possible during real rapid contact; thresholds are research-only',
                    'No hardware, clock synchronization, bus electromagnetic noise or physical safety certification tested']}

if __name__=='__main__':
    result=main()
    (Path(__file__).parent/'benchmark_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
