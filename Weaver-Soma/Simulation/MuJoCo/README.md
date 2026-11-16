# Weaver-SOMA | MuJoCo assisted biped gait and sensor simulation

This repository experiment uses a real MuJoCo rigid-body engine. It is better suited for contact forces, ground friction, joint positions and inertial sensors than Blender's animation workflow. Blender can later produce meshes and visual assets.

**The robot is externally harness-assisted, controlled by hand-written joint targets, not autonomous learned walking.** No real finger electronics, temperature measurements, physical calibration or motor safety validation exists here.

## Sensors represented

- Six actuated leg joints with joint encoders and torque commands
- Six-axis inertial measurements: 3-axis gyro plus 3-axis accelerometer
- Eight simulated foot-taxel normal contact forces from the MuJoCo solver
- Temperature generated analytically as a **synthetic proxy**, not heat-transfer physics
- Independently flagged missing force and temperature channels
- Friction changes and slowed motors for unfamiliar dynamics tests

## Install and run

~~~bash
python -m pip install -r requirements.txt
python -m pytest -q test_biped.py
python run_biped.py --mode nominal --duration 5.76 --out results --draw
python run_biped.py --mode slippery --duration 5.76 --out results
python run_biped.py --mode slow_motors --duration 5.76 --out results
python run_biped.py --mode faulty --duration 5.76 --out results
~~~

The GitHub Actions runner installs the physics engine and publishes sensor traces, metrics, a trajectory visualization, and an animated GIF as build artifacts.

## U2 shape interface

The simulator records 120ms sampled seven-channel SOMA observations (pressure in kPa from a declared simulated contact area, analytic temperature C, normalized knee, three time derivatives and contact bit) and three independent channel-health flags. Results pack into **4 x 8 x 7** and **4 x 8 x 3** history windows, compatible with the Stage 5 data contract. This **does not load** the trained 100M checkpoint or authorize any actuator. Actual six-joint motor commands must replace the all-zero placeholder action tokens before causal/action training.

The independently checked physics outputs remain synthetic ground truth, not physical robot certification. Neural outputs are prohibited from controlling the motor or bypassing Weaver's signed execution boundaries.
