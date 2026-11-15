"""Local demonstration, no calls to hosted services or robot motors."""
import json
from pathlib import Path
from train_unified import make_data
from weaver_adapter import WeaverSOMABridge

folder=Path(__file__).resolve().parent
b=make_data(1,914,'normal')
bridge=WeaverSOMABridge(str(folder/'results/weaver_soma_unified.pt'))
q=b['quality'][0].numpy()
view,latent=bridge.infer(b['obs'][0].numpy(),q,b['context'][0].numpy(),b['executed_action'][0].numpy())
print('MODEL ADVICE ONLY:',view)
print('WEAVER V3 OBSERVATION:',json.dumps(bridge.to_mesh_observation(q,age_ms=0),indent=2))
print('WEAVER V3 MEMORY METADATA:',bridge.to_memory_metadata(view))
print('WEAVER AKASHIC HUB VECTOR SHAPE:',bridge.to_hub_vector(latent).shape)
print('UNSIGNED OBSERVE-ONLY REQUEST:',json.dumps(bridge.safe_intent_for_observation({'body_revision':1,'world_revision':1}),indent=2))
print('NO SIGNING, EXECUTION, MEMORY PERSISTENCE OR NETWORK CALL PERFORMED.')
