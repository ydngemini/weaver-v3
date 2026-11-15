"""A STRICTLY READ-ONLY prototype adapter for Weaver v3/v20 interfaces.

Based on repository contracts (2026-11-15):
weaver-v3/CascadeProjects/windsurf-project/weaver_cognition_mesh.py
weaver-v3/CascadeProjects/windsurf-project/weaver_neural_fabric.py
weaver/Cognition and WorldModel modules.

NO direct actuation, calls, signing, online updates, self-modification, or persistence.
"""
from dataclasses import dataclass
from typing import Any,Optional
import time
import numpy as np
import torch
from unified_model import WeaverSOMAUnified

ACTION_NAMES=('hold','ease','withdraw')
ROUTE_NAMES=('embodiment','realtime','interactive','background')
WORKSPACE_NAMES=('safety','external_observation','social','memory','goal')

@dataclass(frozen=True)
class SafeCortexView:
    action_proposal: str
    action_risks: tuple[float,float,float]
    sensor_reliability: float
    external_event_score: float
    workspace: str
    compute_advice: str
    latent_words: tuple[int,int,int,int]

class WeaverSOMABridge:
    """Never turn model proposals into hardware commands or signed intents.
    One frozen network is the entire learned cognitive core.
    """
    def __init__(self,checkpoint:str,action_cost_penalty:float=.02):
        obj=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if obj.get('architecture')!='WeaverSOMAUnified-v1':
            raise ValueError('unsupported/unknown model checkpoint')
        self.model=WeaverSOMAUnified()
        self.model.load_state_dict(obj['weights'],strict=True)
        self.model.eval()
        self.action_cost_penalty=float(action_cost_penalty)
        if not 0<=self.action_cost_penalty<=.2:
            raise ValueError('action cost penalty must be bounded')

    @torch.inference_mode()
    def infer(self,obs:np.ndarray,quality:np.ndarray,context:np.ndarray,executed_action:np.ndarray)->tuple[SafeCortexView,np.ndarray]:
        if obs.shape!=(8,7) or quality.shape!=(8,3) or context.shape!=(8,) or executed_action.shape!=(3,):
            raise ValueError('invalid sensor/context shape')
        if (not np.isfinite(obs).all() or not np.isfinite(context).all() or not np.isfinite(quality).all()
            or not np.isfinite(executed_action).all()):
            raise ValueError('nonfinite input')
        if not np.logical_or(quality==0,quality==1).all():
            raise ValueError('health flags must be independently determined and binary')
        inputs=[torch.from_numpy(v.astype('float32')).unsqueeze(0) for v in (obs,quality,context,executed_action)]
        out=self.model(*inputs)
        risk=out['risk_logits'].sigmoid()[0].numpy()
        cost=self.action_cost_penalty*np.array([0,.2,1.],dtype=np.float32)
        choice=int(np.argmin(risk+cost))
        confidence=float(quality[-1].mean()) # hardware reporting quality, NOT model certainty
        view=SafeCortexView(action_proposal=ACTION_NAMES[choice],action_risks=tuple(float(x) for x in risk),
               sensor_reliability=confidence,external_event_score=float(out['external_logits'][0].sigmoid()),
               workspace=WORKSPACE_NAMES[int(out['workspace_logits'][0].argmax())],
               compute_advice=ROUTE_NAMES[int(out['compute_logits'][0].argmax())],
               latent_words=tuple(int(i) for i in out['body_tokens'][0].tolist()))
        return view,out['state'][0].detach().numpy()

    @staticmethod
    def to_mesh_observation(quality:np.ndarray,*,age_ms:int=0)->dict[str,Any]:
        """Conforms to CognitionMesh.observe / SituationalGraph.observe.
        There are only 4 registered SENSOR_CHANNELS; do NOT invent 'thermal'.
        """
        if quality.shape!=(8,3) or not np.isfinite(quality).all():
            raise ValueError('quality must have [8,3] finite samples')
        if not np.isin(quality,[0,1]).all():
            raise ValueError('quality must be binary')
        age_ms=int(age_ms)
        if not 0<=age_ms<=4800:
            raise ValueError('age outside accepted v3 body channel range')
        return {'observed_at_ms':int(time.time()*1000),
                'sensors':{'body':{'confidence':float(quality[-1].mean()),'sample_age_ms':age_ms}}}

    @staticmethod
    def to_memory_metadata(view:SafeCortexView)->dict[str,Any]:
        """Inputs for 'CognitionMesh.memory.record()'. Metadata only; no private/raw data."""
        prob=max(view.action_risks)
        return {'kind':'observation','tags':['body','safety'], 'reward':0.,
                'surprise':round(float(view.external_event_score),4),
                'risk':round(float(prob),4),'success':True}

    @staticmethod
    def to_hub_vector(latent:np.ndarray)->np.ndarray:
        """An AkashicHub(dim=256)-compatible observation; use hub.write('soma', v).
        The vectors are operational learned representations, not conscious thoughts.
        """
        if latent.shape!=(64,) or not np.isfinite(latent).all():
            raise ValueError('expected finite 64-dim latent')
        out=np.zeros(256,dtype=np.float32)
        out[:64]=latent.astype(np.float32)
        return out

    @staticmethod
    def safe_intent_for_observation(awareness:dict[str,Any])->dict[str,Any]:
        """Unsigned REQUEST for 'IntentCompiler.compile'; NEVER forge signed capsules.
        Not a physical movement; requests refreshed body observation.
        """
        return {'goal':'refresh validated body telemetry',
                'priority':'embodiment',
                'actions':[{'type':'observe','sensor':'body'}],
                'ttl_ms':5000,
                'preconditions':{'body_revision':int(awareness.get('body_revision',0)),
                                 'world_revision':int(awareness.get('world_revision',0)),
                                 'requires_awake':True,'max_duration_ms':2000}}

    @staticmethod
    def to_legacy_world_state(view:SafeCortexView)->dict[str,Any]:
        """Optional structured state for v20 WorldModel.update_state(**state).
        Not written automatically, not a reimplementation of its DB or 'soul'.
        """
        return {'soma_synthetic_model_summary':{
                'sensor_reliability':view.sensor_reliability,
                'external_event_score':view.external_event_score,
                'action_advice_only':view.action_proposal,
                'model_version':'weaver-soma-u1'}}
