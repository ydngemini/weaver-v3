"""Seeded synthetic fault-injection benchmark, CPU-only, never robot certification."""
import argparse
import dataclasses
import json
from pathlib import Path
import numpy as np
from sensors import SensorPacket, HealthMonitor, DEFAULT_SPECS

BASE = 12_000_000_000
DT = 100_000_000


def packet(channel, seq, value, timestamp=None):
    return SensorPacket.create(DEFAULT_SPECS[channel], seq, BASE+seq*DT if timestamp is None else timestamp, value)


def baseline(channel, jitter):
    return {"pressure": 35.0 + jitter, "temperature": 34.0 + jitter*0.04,
            "joint": 0.41 + jitter*0.002}[channel]


def trial(channel, kind, seed):
    rng=np.random.default_rng(seed)
    m=HealthMonitor()
    prior=[]
    # Independent clean warmup; no model predictions or future labels in monitor.
    for k in range(5):
        v=baseline(channel, float(rng.normal(0,.2)))
        p=packet(channel,k,v)
        r=m.inspect(p,BASE+k*DT)
        assert r.trusted if k>=1 else not r.trusted
        prior.append(v)
    k=5; t=BASE+k*DT
    good_value=baseline(channel,float(rng.normal(0,.2)))
    p=packet(channel,k,good_value)
    opts={}
    if kind=="checksum_corruption":
        p=dataclasses.replace(p,checksum=p.checksum^0x100)
    elif kind=="sequence_replay":
        p=packet(channel,4,prior[-1])
    elif kind=="stale_time":
        p=packet(channel,k,good_value,t-800_000_000)
    elif kind=="out_of_range":
        p=packet(channel,k,DEFAULT_SPECS[channel].valid_max+100)
    elif kind=="stuck_with_independent_witness":
        p=packet(channel,k,prior[-1])
        opts["witness_change"]=DEFAULT_SPECS[channel].witness_min_change*3
    elif kind=="plausible_bias_with_redundancy":
        delta=DEFAULT_SPECS[channel].independent_tolerance*1.8
        # For pressure the gain is 14.4kPa in 100ms; within permissive max-rate.
        p=packet(channel,k,good_value+delta)
        opts["independent_reference"]=good_value
    elif kind=="plausible_bias_without_redundancy":
        # Same plausible drift, but without reference or independent excitation.
        delta=DEFAULT_SPECS[channel].independent_tolerance*0.7
        p=packet(channel,k,good_value+delta)
    elif kind=="clean":
        pass
    else:raise ValueError(kind)
    r=m.inspect(p,t,**opts)
    return r


def run(seed=301, per_case=250):
    faults=["checksum_corruption","sequence_replay","stale_time","out_of_range",
            "stuck_with_independent_witness","plausible_bias_with_redundancy"]
    blind_spot="plausible_bias_without_redundancy"
    groups={}
    for kind in ["clean",*faults,blind_spot]:
        total=0;rejected=0;reasons={}
        for i,channel in enumerate(DEFAULT_SPECS):
            for trial_i in range(per_case):
                report=trial(channel,kind,seed + 100_000*i + 17_000*trial_i)
                total+=1
                if not report.trusted:rejected+=1
                for why in report.reasons:
                    reasons[why]=reasons.get(why,0)+1
        groups[kind]={"total":total,"rejected":rejected,"rejection_fraction":round(rejected/total,5),"reasons":reasons}
    positives=sum(groups[k]['rejected'] for k in faults)
    pcount=sum(groups[k]['total'] for k in faults)
    report={"scenario":"synthetic deterministic injected faults with independent reference only where specified",
            "seed":seed,"channels":list(DEFAULT_SPECS),"per_channel_per_scenario":per_case,
            "known_detectable_injection_recall":round(positives/pcount,6),
            "known_fault_events":pcount,"known_faults_detected":positives,
            "clean_false_alarm_fraction":groups['clean']['rejection_fraction'],
            "blind_spot":"Plausible smooth sensor bias without independent witness is not reliably detectable.",
            "results":groups,
            "warning":"Synthetic packets and trusted-channel assumptions; no evidence of real device reliability or physical robot safety."}
    return report

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('--cases',type=int,default=250)
    ap.add_argument('--output',default='fault_benchmark_results.json')
    args=ap.parse_args()
    report=run(per_case=args.cases)
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
    for k,v in report['results'].items():print(k,v['rejected'],'/',v['total'],'rejected')
