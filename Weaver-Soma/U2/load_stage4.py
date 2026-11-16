"""Load the verified WEAVER-SOMA U2 Stage 4 unified research checkpoint.
No hardware/robot runtime access or signed intent authorization.
"""
from pathlib import Path
import argparse
import torch
from weaver_soma_u2 import WeaverSOMA100M,count_parameters

def load(path: str):
    obj=torch.load(path,map_location='cpu',weights_only=True)
    if obj.get('architecture')!='WeaverSOMA100M-stage4' or obj.get('format_version')!=1:
        raise ValueError('Not a WEAVER-SOMA U2 Stage4 unified checkpoint')
    if obj.get('total_parameters')!=100206197 or not obj.get('no_motor_authority'):
        raise ValueError('Unexpected model metadata')
    model=WeaverSOMA100M().eval()
    model.load_state_dict(obj['weights'],strict=True)
    assert count_parameters(model)==100206197
    return model

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint',nargs='?',default=str(
        Path(__file__).resolve().parent.parent/'results'/'WEAVER_SOMA_U2_100M_Stage4_FULL_MODEL.pt'))
    args=parser.parse_args()
    torch.set_num_threads(3)
    model=load(args.checkpoint)
    observations=torch.randn(1,8,7)
    quality=torch.ones(1,8,3)
    context=torch.zeros(1,8)
    action=torch.tensor([[0.,1.,0.]])
    with torch.inference_mode():
        out=model(observations,quality,context,action)
    print('WEAVER-SOMA U2 research parameters:',count_parameters(model))
    print('Synthetic example risk predictions:',out['risk_logits'].sigmoid().tolist())
    print('No robot actuators or production services connected.')

if __name__=='__main__':
    main()
