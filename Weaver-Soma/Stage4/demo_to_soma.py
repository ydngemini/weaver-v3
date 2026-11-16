"""Run synthetic integrity-checked readings into recovered U1 PyTorch model.
No hardware commands or networking. No robot safety claims.
"""
from pathlib import Path
import hashlib
import sys
import numpy as np
import torch
from sensors import SOMASensorWindow, SensorPacket, DEFAULT_SPECS


def run(checkpoint_path: str):
    path=Path(checkpoint_path)
    expected='3eb0746806f3842ec2d8121a7bcf30f5351b1b26b70e2c53debdac2ea5030c8f'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError('unexpected U1 checkpoint digest')
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'weaver_soma_stage3'/'U1'))
    from unified_model import WeaverSOMAUnified
    model=WeaverSOMAUnified().eval()
    obj=torch.load(path,map_location='cpu',weights_only=True)
    model.load_state_dict(obj['weights'],strict=True)
    assembler=SOMASensorWindow()  # illustrative safety boundary omitted
    BASE=50_000_000_000
    for tick in range(15):
        t=BASE+tick*100_000_000
        values={'pressure':30+0.15*tick,'temperature':34.0+0.01*tick,'joint':.35+.002*tick}
        packets={c:SensorPacket.create(spec,tick,t,values[c]) for c,spec in DEFAULT_SPECS.items()}
        if tick==10:
            # A single-device data corruption at an otherwise plausible value.
            from dataclasses import replace
            packets['temperature']=replace(packets['temperature'],checksum=packets['temperature'].checksum^1)
        sample=assembler.push(packets,t,independently_verified_contact=(tick>=3))
        x=torch.from_numpy(sample.obs).unsqueeze(0)
        q=torch.from_numpy(sample.quality).unsqueeze(0)
        context=torch.zeros(1,8)
        with torch.inference_mode():out=model(x,q,context)
        if tick in (0,1,9,10,11,12,14):
            print({'tick':tick, 'trusted':sample.quality[-1].astype(int).tolist(),
                   'stop_requested':sample.stop_requested,
                   'reason_temperature':sample.reasons['temperature'],
                   'risk_probabilities':np.round(out['risk_logits'].sigmoid()[0].numpy(),4).tolist()})
    print('No action taken. Independent supervisor must enforce any stop request.')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True,help='Verified recovered U1 .pt on local disk')
    run(p.parse_args().checkpoint)
