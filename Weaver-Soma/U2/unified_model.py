"""Weaver-SOMA U1: ONE multi-task trained neural core, research only.
Never use for physical motor control or to establish phenomenal consciousness.

Single nn.Module and checkpoint; explicit deterministic infrastructure stays outside.
"""
import torch
from torch import nn
import torch.nn.functional as F

CHANNEL_CENTER = (31.,34.,.48,0.,0.,0.,0.)
CHANNEL_SCALE = (22.,12.,.24,16.,9.,.16,1.)
STATE_CENTER = (31.,34.,.48)
STATE_SCALE = (22.,12.,.24)
HORIZON=6
N_ACTIONS=3


def masked_inputs(obs:torch.Tensor, quality:torch.Tensor):
    """Fully block invalid modalities in value AND derivative; health always explicit."""
    scale=obs.new_tensor(CHANNEL_SCALE)
    center=obs.new_tensor(CHANNEL_CENTER)
    normalized=(obs-center)/scale
    full_mask=torch.cat([quality,quality,torch.ones_like(quality[:,:,:1])],dim=-1)
    return normalized*full_mask,quality


class DiscreteBodyLexicon(nn.Module):
    """Internal compositional code; integers are not words, meanings require probes."""
    def __init__(self,dim=32,slots=4,words=8):
        super().__init__()
        self.slots=slots; self.words=words
        self.assign=nn.ModuleList([nn.Linear(dim//slots,words) for _ in range(slots)])
        self.codebook=nn.Parameter(torch.randn(slots,words,dim//slots)*.06)
    def forward(self,x):
        chunks=x.chunk(self.slots,dim=-1)
        codes=[];ids=[];usages=[]
        for i,c in enumerate(chunks):
            logits=self.assign[i](c)
            probs=logits.softmax(-1)
            # train with straight-through, decode with argmax
            if self.training:
                onehot=F.gumbel_softmax(logits,tau=.85,hard=True,dim=-1)
            else:
                onehot=F.one_hot(logits.argmax(-1),self.words).to(c.dtype)
            codes.append(onehot @ self.codebook[i])
            ids.append(onehot.argmax(-1))
            usages.append(((probs.mean(0)-1/self.words)**2).sum())
        return torch.cat(codes,-1),torch.stack(ids,-1),torch.stack(usages).mean()


class WeaverSOMAUnified(nn.Module):
    """Three sensory encoders -> recurrent body schema -> global workspace + shared heads.

    One trained checkpoint for all learned functions. Hard reflex/transport/crypto not learned.
    Input obs[B,8,7], quality[B,8,3], context[B,8], executed action[B,3].
    5 cognitive experts run inside one differentiable torch model.
    """
    def __init__(self,width=64,latent=64):
        super().__init__()
        self.width=width;self.latent_dim=latent
        self.skin=nn.Sequential(nn.Linear(3,24),nn.SiLU(),nn.Linear(24,24),nn.SiLU())
        self.thermal=nn.Sequential(nn.Linear(3,24),nn.SiLU(),nn.Linear(24,24),nn.SiLU())
        self.kinematic=nn.Sequential(nn.Linear(3,24),nn.SiLU(),nn.Linear(24,24),nn.SiLU())
        self.time_encoder=nn.GRU(input_size=24*3+1+3+8+3,hidden_size=width,batch_first=True)
        self.latent=nn.Sequential(nn.Linear(width,latent),nn.LayerNorm(latent),nn.Tanh())
        self.memory_gate=nn.Linear(latent,latent)
        self.memory_value=nn.Linear(latent,latent)
        self.lex=DiscreteBodyLexicon(dim=32,slots=4,words=8)
        self.lex_prep=nn.Linear(latent,32)
        self.lex_restore=nn.Linear(32,latent)
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(latent,64),nn.SiLU(),nn.Linear(64,latent)) for _ in range(5)])
        self.workspace_gate=nn.Linear(latent+8,5)
        self.fusion=nn.Sequential(nn.Linear(latent*3,latent),nn.SiLU(),nn.LayerNorm(latent))
        # shared broadcast state connects ALL cognitive heads
        self.counterfactual=nn.Sequential(nn.Linear(latent+3,96),nn.SiLU(),nn.Linear(96,HORIZON*3))
        self.risk=nn.Sequential(nn.Linear(latent+3+3,64),nn.SiLU(),nn.Linear(64,1))
        self.action_policy=nn.Linear(latent,3)
        self.event=nn.Linear(latent,1)
        self.episodic_summary=nn.Linear(latent,3)
        self.body_readout=nn.Linear(latent,3)
        self.workspace_task=nn.Linear(latent,5)
        self.compute_advice=nn.Linear(latent,4)
        self.regulation=nn.Sequential(nn.Linear(latent,32),nn.SiLU(),nn.Linear(32,7),nn.Sigmoid())
        self.confidence=nn.Sequential(nn.Linear(latent,1),nn.Sigmoid())
        self.semantic_memory_key=nn.Linear(latent,32)

    def refine_shared_state(self, common, z, internal_memory, broadcast, context, token_z):
        # Preserve existing U1 outputs while allowing a deeper shared trunk.
        return common

    def forward(self,obs,quality,context,executed_action=None,ablate=None):
        if executed_action is None:
            executed_action=torch.zeros(obs.shape[0],3,device=obs.device,dtype=obs.dtype)
        norm,q=masked_inputs(obs,quality)
        inputs=[]
        for idx,encoder in enumerate((self.skin,self.thermal,self.kinematic)):
            pair=torch.cat([norm[:,:,[idx,idx+3]],q[:,:,idx:idx+1]],-1)
            inputs.append(encoder(pair)*q[:,:,idx:idx+1])
        t=obs.size(1)
        x=torch.cat([*inputs,norm[:,:,6:7],q,context.unsqueeze(1).expand(-1,t,-1),executed_action.unsqueeze(1).expand(-1,t,-1)],dim=-1)
        _,hidden=self.time_encoder(x)
        z=self.latent(hidden[-1])
        # gated ephemeral internal memory, not external persistent storage
        internal_memory=torch.sigmoid(self.memory_gate(z))*torch.tanh(self.memory_value(z))
        if ablate=="memory":internal_memory=torch.zeros_like(internal_memory)
        token_z,token_ids,lex_regularizer=self.lex(self.lex_prep(z))
        expert_states=torch.stack([expert(z) for expert in self.experts],dim=1)
        weights=self.workspace_gate(torch.cat([z,context],-1)).softmax(dim=-1)
        broadcast=(weights.unsqueeze(-1)*expert_states).sum(1)
        if ablate=="workspace":broadcast=torch.zeros_like(broadcast)
        if ablate=="lexicon":token_z=torch.zeros_like(token_z)
        common=self.fusion(torch.cat([z,internal_memory,broadcast+self.lex_restore(token_z)],dim=-1))
        common=self.refine_shared_state(common,z,internal_memory,broadcast,context,token_z)
        B=obs.size(0)
        actions=torch.eye(3,device=obs.device,dtype=obs.dtype).unsqueeze(0).expand(B,-1,-1)
        states=common.unsqueeze(1).expand(-1,3,-1)
        futures=self.counterfactual(torch.cat([states,actions],-1)).reshape(B,3,HORIZON,3)
        boundary=((futures*obs.new_tensor(STATE_SCALE)+obs.new_tensor(STATE_CENTER)-obs.new_tensor([62.,43.,.82])) /obs.new_tensor(STATE_SCALE)).amax(dim=2)
        risk_logits=self.risk(torch.cat([states,actions,boundary],-1)).squeeze(-1)
        return {'state':common,'risk_logits':risk_logits,'future':futures,
                'policy_logits':self.action_policy(common),'external_logits':self.event(common).squeeze(-1),
                'episodic_summary':self.episodic_summary(common),'body_readout':self.body_readout(common),
                'workspace_logits':self.workspace_task(common),'compute_logits':self.compute_advice(common),
                'regulation':self.regulation(common),'reliability':self.confidence(common).squeeze(-1),
                'memory_key':self.semantic_memory_key(common),'workspace_weights':weights,
                'body_tokens':token_ids,'lex_regularizer':lex_regularizer}


def joint_objective(out,batch):
    bce=F.binary_cross_entropy_with_logits(out['risk_logits'],batch['risk'])
    traj=F.smooth_l1_loss(out['future'],batch['future'],beta=.3)
    policy=F.cross_entropy(out['policy_logits'],batch['best_action'])
    event=F.binary_cross_entropy_with_logits(out['external_logits'],batch['external'])
    summary=F.smooth_l1_loss(out['episodic_summary'],batch['summary'])
    body=F.smooth_l1_loss(out['body_readout'],batch['body'])
    workspace=F.cross_entropy(out['workspace_logits'],batch['attention'])
    route=F.cross_entropy(out['compute_logits'],batch['route'])
    regulate=F.mse_loss(out['regulation'],batch['regulation'])
    reliability=F.mse_loss(out['reliability'],batch['reliability'])
    total=(bce+.22*traj+.27*policy+.16*event+.16*summary+.12*body+
           .13*workspace+.10*route+.12*regulate+.06*reliability+.008*out['lex_regularizer'])
    return total,{'risk':bce.item(),'future':traj.item(),'policy':policy.item(),
                  'external':event.item(),'memory':summary.item(),'workspace':workspace.item()}