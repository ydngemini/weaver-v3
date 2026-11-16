"""Synthetic V2 frames to the ACTUAL recovered U1 checkpoint. No motors or network."""
from __future__ import annotations
import hashlib,sys,json,os
from pathlib import Path
import numpy as np
import torch
from packet import FingerFrame,NEW_ADC,NEW_TEMP,NEW_ENCODER
from gateway import FingerGateway,FingerCalibration
from bridge import OriginalSOMABridge

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=Path(os.environ.get('SOMA_U1_CHECKPOINT',str(ROOT/'U1'/'results'/'weaver_soma_unified.pt')))
OLD_MODEL=Path(os.environ.get('SOMA_U1_SOURCE_DIR',str(ROOT/'U1')))
EXPECTED='3eb0746806f3842ec2d8121a7bcf30f5351b1b26b70e2c53debdac2ea5030c8f'

def smoke(checkpoint:Path=CHECKPOINT,model_directory:Path=OLD_MODEL):
    if not checkpoint.is_file():raise FileNotFoundError('Recover original U1 checkpoint first')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=EXPECTED:
        raise ValueError('Recovered U1 weight checksum mismatch')
    if not (model_directory/'unified_model.py').exists():
        raise FileNotFoundError('Original U1 model source must be available')
    sys.path.insert(0,str(model_directory))
    from unified_model import WeaverSOMAUnified
    obj=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if obj.get('architecture')!='WeaverSOMAUnified-v1':raise ValueError('wrong model')
    net=WeaverSOMAUnified().eval();net.load_state_dict(obj['weights'],strict=True)
    gateway=FingerGateway(FingerCalibration(3,0x203,'a'*64,trusted_clock_offset_ms=0))
    bridge=OriginalSOMABridge()
    # Simulated sensor pod, 100Hz CAN-FD; real TMP117 new sample ~8Hz.
    for j in range(190):
        flags=NEW_ADC|NEW_ENCODER
        if j%13==0: flags|=NEW_TEMP
        f=FingerFrame(node_id=3,seq=j+1,capture_ms=1000+j*10,
            force_centinewtons=(30,40,0,20),angle_u14=2000+j//15,
            angle_witness_u14=2001+j//15,contact_centi_c=3100+(j//13)*2,
            witness_centi_c=3100+(j//13)*2,supply_mv=5000,flags=flags)
        bridge.ingest(gateway.ingest(f.encode(),can_id=0x203,received_ms=f.capture_ms+7))
    obs,q,meta=bridge.window(2890)
    with torch.inference_mode():
        outputs=net(torch.from_numpy(obs[None]),torch.from_numpy(q[None]),
                    torch.zeros((1,8)),torch.tensor([[1.,0.,0.]]))
    for key,tensor in outputs.items():
        assert torch.isfinite(tensor.float()).all(),key
    assert float(q[:,0].sum())==0.0,'never invent pressure without area calibration'
    result={'checkpoint_sha256':EXPECTED,'original_u1_parameters':sum(p.numel() for p in net.parameters()),
        'total_frames':190,'pressure_quality_sum':float(q[:,0].sum()),
        'thermal_quality_sum':float(q[:,1].sum()),'joint_quality_sum':float(q[:,2].sum()),
        'risk_probs':outputs['risk_logits'].sigmoid().numpy()[0].tolist(),
        'shape_obs':list(obs.shape),'shape_quality':list(q.shape),
        'note':'Synthetic bridge shape check ONLY, no physical safety or model accuracy measured'}
    (Path(__file__).resolve().parent/'u1_smoke_results.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':print(json.dumps(smoke(),indent=2))
