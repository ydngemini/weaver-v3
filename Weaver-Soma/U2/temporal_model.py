"""Single jointly trainable WEAVER-SOMA U2 + temporal state across 32 samples.

The existing 100M U2 model is not replaced. A compact GRU summarizes older
windows while the 100M core is fine-tuned on the newest window. No motor control.
"""
from __future__ import annotations
import torch
from torch import nn
from weaver_soma_u2 import WeaverSOMA100M

class StreamingWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.core=WeaverSOMA100M()
        self.temporal_cell=nn.GRUCell(64+8+3,64)
        self.long_hazard=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,3))
        self.future_body=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,3))

    def forward(self,obs,quality,context,actions,*,ablate_history=False):
        """Inputs [B,4,8,7], [B,4,8,3], [B,4,8], [B,4,3].

        Truncated backprop through the earlier 3 core invocations limits CPU RAM;
        memory cell + final 100M core share a single optimizer and checkpoint.
        """
        if obs.ndim!=4 or obs.shape[1:]!=(4,8,7):raise ValueError('Expected 32 frames of sensor observations')
        if quality.shape!=(obs.shape[0],4,8,3):raise ValueError('Sensor quality must match every sample')
        h=obs.new_zeros(obs.shape[0],64)
        for i in range(4):
            # Earlier encoder history is frozen *per forward* for bounded memory,
            # but its shared weights ARE trained when processing the latest window.
            if i<3:
                if ablate_history:continue
                with torch.no_grad():
                    old=self.core(obs[:,i],quality[:,i],context[:,i],actions[:,i])
                z=old['state'].detach()
            else:
                final=self.core(obs[:,i],quality[:,i],context[:,i],actions[:,i])
                z=final['state']
            h=self.temporal_cell(torch.cat((z,context[:,i],actions[:,i]),dim=-1),h)
        merged=torch.cat((final['state'],h),dim=-1)
        return {'last':final,'long_hazard_logits':self.long_hazard(merged),
                'memory_body':self.future_body(merged),'memory':h}