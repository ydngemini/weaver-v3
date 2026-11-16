# Stage 5 CPU reproduction and tests

From the repository root:

```bash
python -m pip install -r Weaver-Soma/U2/requirements_stage5.txt
PYTHONPATH=Weaver-Soma/U1:Weaver-Soma/U2:Weaver-Soma/Sensors:Weaver-Soma/Sensors/V2 python -m pytest -q Weaver-Soma/U2/test_stage5.py Weaver-Soma/Sensors/V2/test_sensor_v2.py
```

Expected results for the archived and reassembled reference sources: **35/35 tests passing** (7 Stage 5 and 28 V2 sensor tests). This was confirmed locally, not through GitHub Actions.

To reproduce the continuation training, separately restore the trusted, approximately 401 MB **trained Stage 4 U2** checkpoint to `Weaver-Soma/U2/results/WEAVER_SOMA_U2_100M_Stage4_FULL_MODEL.pt`. That weight file is intentionally excluded from this public repository. Then from the repo root:

```bash
cd Weaver-Soma/U2
python train_stage5.py --steps 80 --batch-size 4 --threads 3
python verify_stage5.py --stage4 ./results/WEAVER_SOMA_U2_100M_Stage4_FULL_MODEL.pt --checkpoint ../results/WEAVER_SOMA_U2_Stage5_Temporal_Research.pt
python smoke_sensor_to_u2.py
```

The training/verification code **cannot execute successfully without those real checkpoints**. It must not quietly initialize random 100M weights and claim reproduction. The earlier research metrics are from synthetic data only. Calibration and actual physical measurements have NOT been completed, and the model must not control actuators.
