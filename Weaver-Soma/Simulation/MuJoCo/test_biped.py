"""Headless MuJoCo physics tests. No robot control or learned gait claim."""
import numpy as np
import mujoco
from run_biped import HERE,simulate,to_u2

def test_mujoco_model_build():
    m=mujoco.MjModel.from_xml_path(str(HERE / 'biped.xml'))
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    assert m.nu==6
    assert m.nq==9
    assert m.nsensordata>=20

def test_assisted_gait_produces_solver_contacts():
    m,rows=simulate(mode="nominal",duration=4.08,seed=15)
    assert m["observations"]>=32
    assert m["u2_32_frame_windows"]>=1
    assert m["max_contact_force_N"]>40, "no meaningful force-bearing contact"
    assert m["root_displacement_m"]>.2, "carriage did not produce forward travel"
    assert m["left_contact_frames"]>=10
    assert m["right_contact_frames"]>=10
    assert m["minimum_pelvis_height_m"]>.65
    assert all(np.isfinite(r["pelvis_height_m"]) for r in rows)
    assert max(r["L_knee_rad"] for r in rows)-min(r["L_knee_rad"] for r in rows)>.2

def test_soma_windows_quality_and_no_hidden_temperature_leak():
    m,rows=simulate(mode="faulty",duration=4.08,seed=22)
    obs,health,windows,quality=to_u2(rows,"faulty")
    assert windows.ndim==4 and windows.shape[-3:]==(4,8,7)
    assert quality.shape[-3:]==(4,8,3)
    assert m["masked_modalities"]>0
    assert np.all(obs[health[:,1]==0,1]==0)
    assert np.all(obs[health[:,1]==0,4]==0)
    assert np.isfinite(windows).all()

def test_unfamiliar_friction():
    m,rows=simulate(mode="slippery",duration=4.08,seed=22)
    assert m["physics_engine"]=="MuJoCo"
    assert len(rows)>=32
    assert m["real_sensor_records"]==0
