"""Independent Stage5 checkpoint integrity and locked post-selection comparison."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import torch,numpy as np
HERE=Path(__file__).resolve().parent
sys.path.append(str(HERE.parent/'U1'))
from temporal_model import StreamingWorldModel
from train_stage5 import score,load_checkpoint
from long_world import make_long

STAGE4=HERE/'results'/'WEAVER_SOMA_U2_100M_Stage4_FULL_MODEL.pt'
STAGE5=HERE.parent/'results'/'WEAVER_SOMA_U2_Stage5_Temporal_Research.pt'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage4',default=str(STAGE4))
    parser.add_argument('--checkpoint',default=str(STAGE5))
    args=parser.parse_args()
    torch.set_num_threads(3)
    a=load_checkpoint(args.stage4)
    b=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    assert b['architecture']=='WeaverSOMA100M-stage5-sequential'
    assert b['source_stage4_sha256']==hashlib.sha256(Path(args.stage4).read_bytes()).hexdigest()
    assert b['no_motor_authority'] is True and b['synthetic_only'] is True
    torch.manual_seed(20261116)
    old=StreamingWorldModel();old.core.load_state_dict(a['weights'],strict=True)
    new=StreamingWorldModel();new.load_state_dict(b['weights'],strict=True)
    old.eval();new.eval()
    # Stage4 foundation and U1-derived peripheral layers must be identical.
    base_equal=all(torch.equal(a['weights'][k],b['weights']['core.'+k])
                   for k in a['weights'] if not k.startswith('deep_cognitive_core.'))
    assert base_equal,'U1 foundation changed'
    changed=[]
    for k,v in a['weights'].items():
        if k.startswith('deep_cognitive_core.'):
            out=b['weights']['core.'+k]
            if not torch.equal(v,out): changed.append((k,(v-out).abs().max().item()))
    assert len(changed)>90,'U2 deep core did not genuinely train'
    # Mutation invariant: masked channel inputs including their derivatives
    # must produce identical predictions even with adversarial corrupt values.
    x=make_long(3,123,'changed','flagged')
    quality=x['quality'].clone();quality[:,:,:,1]=0
    x1=x['obs'].clone();x2=x1.clone()
    x2[:,:,:,1]=98765.;x2[:,:,:,4]=-45678.
    with torch.inference_mode():
        z=new(x1,quality,x['context'],x['executed_action'])
        corrupted=new(x2,quality,x['context'],x['executed_action'])
    diff=max(float((z['last'][key]-corrupted['last'][key]).abs().max()) for key in z['last'])
    diff=max(diff,float((z['long_hazard_logits']-corrupted['long_hazard_logits']).abs().max()))
    assert diff<=1e-6,diff
    # Post-selection holdout from seeds matching stage5's sealed test groups.
    comparisons={}
    for regime,fault,i in [('thermal_shift','flagged',6),('motor_shift','clean',8),('extreme','silent',3)]:
        # build(with_shift=True) enumerates extreme x4, thermal_shift x4, motor_shift x4.
        dataset=make_long(100,30000+2713*i,regime,fault)
        comparisons[f'{regime}/{fault}']={'stage4_with_untrained_memory':score(old,dataset),
                                         'stage5_trained':score(new,dataset)}
    report={'stage5_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            'checkpoint_bytes':Path(args.checkpoint).stat().st_size,
            'total_parameters':sum(p.numel() for p in new.parameters()),
            'core_parameters':sum(p.numel() for p in new.core.parameters()),
            'changed_deep_core_tensors':len(changed),
            'max_deep_tensor_change':max(x[1] for x in changed),
            'original_u1_layers_bit_exact':base_equal,
            'adversarial_masked_thermal_max_output_difference':diff,
            'locked_comparisons':comparisons,
            'physical_observations':0,
            'warning':'synthetic-only physical models; no hardware control, no consciousness or physical safety claim'}
    out=HERE.parent/'results'/'stage5_verification.json'
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()