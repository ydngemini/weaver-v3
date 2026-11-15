# WEAVER-SOMA U2: 100,204,067-parameter CPU architecture

U2 **extends U1**, retaining its sensor encoders, recurrent body memory, five experts, shared-workspace representations, counterfactual forecasts and output signatures. A ten-block 1,024-dimensional attention/MLP cognitive trunk is added before the shared output heads. It is one trainable PyTorch module, not a separate AI service.

## Current state

- Parameter count: 100,204,067. Architecture and CPU forward/backward tests have succeeded locally.
- The **trained U1 epoch-24 checkpoint has been recovered and verified** (91 matching base tensors).
- The original U1 tensors can be loaded by WeaverSOMA100M.load_u1_weights(trusted_path).
- The additional U2 deep-cognitive-core weights are **randomly initialized and untrained**. Compatibility is not proof of improved capabilities.
- This is a numerical CPU research model, not a pretrained language model, complete general intelligence, or physical robot controller.

## Quick start

From this directory with torch installed:

~~~bash
python test_u2.py
python weaver_soma_u2.py --smoke-test
python weaver_soma_u2.py --train-step
~~~

Restoring original learned weights:

~~~python
from weaver_soma_u2 import WeaverSOMA100M
model = WeaverSOMA100M()
matched_tensors = model.load_u1_weights("../U1/results/weaver_soma_unified.pt")
assert matched_tensors == 91
~~~

The restored original file is not kept in the public repository. Its SHA-256 is:
3eb0746806f3842ec2d8121a7bcf30f5351b1b26b70e2c53debdac2ea5030c8f.

Recommended next stage: freeze U1's existing trained layers; train the new shared U2 trunk on the same synthetic data while measuring validation loss; progressively unfreeze only after stability checks; compare action errors, sensor-failure robustness and calibration on held-out episodes. Do not claim the trained U1 metrics for the untrained U2 model.

Never let model predictions sign intents, command motors, override an emergency stop or bypass Weaver's runtime safety guards.
