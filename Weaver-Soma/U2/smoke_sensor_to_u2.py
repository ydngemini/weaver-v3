"""Synthetic SOMA-SENSE V2 CAN-FD packets -> trained U2 Stage5, no motors."""
from pathlib import Path
import json,sys,time,numpy as np,torch
HERE=Path(__file__).resolve().parent
sys.path.append(str(HERE.parent/'U1'))
from temporal_model import StreamingWorldModel
from sensor_stream import CaptureSession
from packet import FingerFrame,NEW_ADC,NEW_TEMP,NEW_ENCODER

CHECKPOINT=HERE.parent/'results'/'WEAVER_SOMA_U2_Stage5_Temporal_Research.pt'

def main():
    torch.set_num_threads(3)
    c=CaptureSession({'node_id':3,'expected_can_id':0x203,'calibration_digest':'a'*64,
                      'trusted_clock_offset_ms':0},source='synthetic')
    for i in range(460):
        flags=NEW_ADC|NEW_ENCODER|(NEW_TEMP if i%13==0 else 0)
        f=FingerFrame(node_id=3,seq=i+1,capture_ms=1000+i*10,
                force_centinewtons=(23,17,12,8),angle_u14=2000+i//18,
                angle_witness_u14=2001+i//18,
                contact_centi_c=3000+i//13,witness_centi_c=3000+i//13,
                supply_mv=4998,flags=flags)
        c.ingest(f.encode(),can_id=0x203,received_ms=f.capture_ms+7)
    obs,q,metadata=c.latest_u2_sequence((2100,3060,4020,4980))
    assert obs.shape==(4,8,7) and q.shape==(4,8,3)
    assert q[:,:,0].sum()==0, 'cannot use force as pressure without calibrated area'
    cp=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    assert cp['architecture']=='WeaverSOMA100M-stage5-sequential'
    net=StreamingWorldModel();net.load_state_dict(cp['weights']);net.eval()
    frames=torch.from_numpy(obs[None]);health=torch.from_numpy(q[None])
    context=torch.zeros((1,4,8));action=torch.zeros((1,4,3));action[:,:,0]=1
    latencies=[]
    with torch.inference_mode():
        for _ in range(12):
            t=time.perf_counter();o=net(frames,health,context,action)
            latencies.append((time.perf_counter()-t)*1000)
    assert o['long_hazard_logits'].shape==(1,3)
    assert torch.isfinite(o['long_hazard_logits']).all()
    for key,val in o['last'].items():assert torch.isfinite(val.float()).all(),key
    report={'source':'synthetic CAN-FD packets, NOT actual hardware',
            'sensor_packets':460,'physical_samples':0,
            'sequence_shape':list(obs.shape),'quality_shape':list(q.shape),
            'pressure_valid_samples':int(q[:,:,0].sum()),
            'temperature_valid_samples':int(q[:,:,1].sum()),
            'joint_valid_samples':int(q[:,:,2].sum()),
            'CPU_latency_ms_median_12_repetitions':round(float(np.median(latencies)),3),
            'CPU_latency_ms_max_12_repetitions':round(float(max(latencies)),3),
            'outputs_finite':True,'autonomous_motor_control':False,
            'note':'Does not qualify real-time deadline or physical safety; no device signatures.'}
    (HERE.parent/'results').mkdir(exist_ok=True)
    (HERE.parent/'results'/'v2_sensor_to_stage5_smoke.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
