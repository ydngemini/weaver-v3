"""Verify sensor monitor with the separately recovered original U1 checkpoint.

No actuator calls; does not publish or modify weights. Requires a trusted local
file whose sha256 matches the original training run.
"""
from __future__ import annotations
import argparse,hashlib,sys
from pathlib import Path
import numpy as np
import torch
from sensor_hub import SensorHub,Sample,SPECS

EXPECTED='3eb0746806f3842ec2d8121a7bcf30f5351b1b26b70e2c53debdac2ea5030c8f'

def main(checkpoint,model_module_dir):
    ckpt=Path(checkpoint)
    actual=hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if actual!=EXPECTED:raise ValueError('Checkpoint hash mismatch')
    sys.path.insert(0,str(Path(model_module_dir).resolve()))
    from unified_model import WeaverSOMAUnified
    obj=torch.load(ckpt,map_location='cpu',weights_only=True)
    assert obj.get('architecture')=='WeaverSOMAUnified-v1'
    net=WeaverSOMAUnified().eval()
    net.load_state_dict(obj['weights'],strict=True)

    hub=SensorHub()
    for i,t in enumerate(range(0,2001,20)):
        p=27+.002*t; temp=32.+.0008*t; joint=.3+.0001*t
        sensors={'pressure':p,'pressure_ref':p+.1,'temperature':temp,
                 'temperature_ref':temp+.1,'joint':joint,'joint_ref':joint+.001}
        for ch,v in sensors.items():
            hub.ingest(Sample(ch,v,SPECS[ch].unit,t,t+1,i,SPECS[ch].expected_calibration))
    obs,q,_=hub.to_soma_window(1920)
    if not (q==1).all():raise AssertionError('Expected clean sensor quality')
    context=torch.zeros(1,8);action=torch.tensor([[1.,0.,0.]])
    with torch.inference_mode():
        before=net(torch.from_numpy(obs)[None],torch.from_numpy(q)[None],context,action)

    # Create a plausible unreported thermal drift. A colocated independent
    # temperature witness continues normally, triggering a conservative mask.
    for i,t in enumerate(range(2040,2481,20),start=101):
        p=27+.002*t;temp=32.+.0008*t;joint=.3+.0001*t
        sensors={'pressure':p,'pressure_ref':p+.1,'temperature':temp+8,
                 'temperature_ref':temp+.1,'joint':joint,'joint_ref':joint+.001}
        for ch,v in sensors.items():
            hub.ingest(Sample(ch,v,SPECS[ch].unit,t,t+1,i,SPECS[ch].expected_calibration))
    corrupted,quality,_=hub.to_soma_window(2400)
    assert (quality[-3:,1]==0).all(),quality[:,1]
    assert np.isfinite(corrupted).all()
    with torch.inference_mode():
        after=net(torch.from_numpy(corrupted)[None],torch.from_numpy(quality)[None],context,action)
    assert torch.isfinite(after['risk_logits']).all()
    print('checkpoint_sha256',actual)
    print('model_parameters',sum(p.numel() for p in net.parameters()))
    print('before_quality',q[-1].tolist(),'after_quality',quality[-1].tolist())
    print('risk_before',before['risk_logits'].sigmoid().numpy().round(4).tolist())
    print('risk_after',after['risk_logits'].sigmoid().numpy().round(4).tolist())
    print('PASS: monitored sensor masks accepted by original trained U1')
    print('No actuator action or physical safety evaluation performed.')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--model-dir',required=True)
    args=parser.parse_args()
    main(args.checkpoint,args.model_dir)
