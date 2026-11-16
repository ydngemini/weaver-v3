"""MuJoCo physically simulated *harness-assisted* biped walking and senses.
No unassisted balance, real thermal dynamics, motor authority, or physical sensor data.
"""
from pathlib import Path
import argparse,json,math
import numpy as np
import mujoco

HERE=Path(__file__).resolve().parent
JOINTS=[f"{s}_{j}" for s in "LR" for j in ("hip","knee","ankle")]
PERIOD=.12
AREA_M2=.0012  # explicitly assumed simulated contact area, NOT a real calibration

def mid(model,typ,name):
    i=mujoco.mj_name2id(model,typ,name)
    if i<0: raise ValueError("missing model component "+name)
    return i

def joint(model,data,name):
    i=mid(model,mujoco.mjtObj.mjOBJ_JOINT,name)
    return float(data.qpos[model.jnt_qposadr[i]]),float(data.qvel[model.jnt_dofadr[i]])

def sensed(model,data,name):
    i=mid(model,mujoco.mjtObj.mjOBJ_SENSOR,name)
    a,b=int(model.sensor_adr[i]),int(model.sensor_dim[i])
    return data.sensordata[a:a+b].copy().tolist()

def control(model,data,t,mode):
    for i,name in enumerate(JOINTS):
        side=name[0]; part=name.split("_")[1]
        phase=2*math.pi*.82*t+(math.pi if side=="R" else 0)
        swing=max(math.sin(phase),0.)
        desired={"hip":-.29*math.sin(phase)+.03,
                 "knee":.10+.58*swing,
                 "ankle":-.16*math.sin(phase)-.31*swing}[part]
        kp,kd,limit={"hip":(55,4,38),"knee":(58,4,32),"ankle":(35,2.4,19)}[part]
        q,v=joint(model,data,name)
        power=.42 if mode=="slow_motors" else 1.
        data.ctrl[i]=np.clip(power*(kp*(desired-q)-kd*v),-limit,limit)
    data.qfrc_applied[:]=0
    x,z,pitch=data.qpos[:3]; vx,vz,rate=data.qvel[:3]
    # EXTERNAL support carriage: these are NOT learned actions.
    data.qfrc_applied[0]=24*((.14 if t>.4 else 0)-vx)
    data.qfrc_applied[1]=-720*z-62*vz
    data.qfrc_applied[2]=-95*pitch-10*rate

def taxels(model,data,foot_name):
    gid=mid(model,mujoco.mjtObj.mjOBJ_GEOM,foot_name+"_foot_geom")
    bid=mid(model,mujoco.mjtObj.mjOBJ_BODY,foot_name+"_foot")
    values=np.zeros(4)
    for i in range(data.ncon):
        c=data.contact[i]
        if gid not in (int(c.geom1),int(c.geom2)):continue
        forces=np.zeros(6)
        mujoco.mj_contactForce(model,data,i,forces)
        normal=max(0.,float(forces[0]))
        R=np.asarray(data.xmat[bid]).reshape(3,3)
        local=R.T@(np.asarray(c.pos)-np.asarray(data.xpos[bid]))
        index=(2 if local[0]>.065 else 0)+(1 if local[1]>0 else 0)
        values[index]+=normal
    return values

def to_u2(samples,mode):
    obs=np.zeros((len(samples),7),dtype=np.float32)
    quality=np.ones((len(samples),3),dtype=np.float32)
    for i,s in enumerate(samples):
        f=s["left_force_N"]+s["right_force_N"]
        p=f/AREA_M2/1000. # kPa from assumed surface patch, synthetic only
        temp=s["temperature_simulated_C"]
        jointfrac=np.clip((s["L_knee_rad"]+s["R_knee_rad"])/2/1.4,0,1)
        obs[i,:3]=[p,temp,jointfrac]
        obs[i,6]=float(f>.2)
        if i:obs[i,3:6]=(obs[i,:3]-obs[i-1,:3])/PERIOD
        if mode=="faulty" and i%17==5:
            quality[i,1]=0
        if mode=="faulty" and i%29==8:
            quality[i,0]=0
    for i in range(len(obs)):
        for j in range(3):
            if quality[i,j]==0:
                obs[i,j]=obs[i,j+3]=0
            if i and quality[i-1,j]==0:obs[i,j+3]=0
    assert np.isfinite(obs).all()
    if len(obs)<32:raise ValueError("need 32 SOMA history samples")
    windows=np.stack([obs[i:i+32].reshape(4,8,7)
                      for i in range(0,len(obs)-31,8)])
    masks=np.stack([quality[i:i+32].reshape(4,8,3)
                    for i in range(0,len(obs)-31,8)])
    return obs,quality,windows,masks

def simulate(mode="nominal",duration=5.76,seed=19,out_dir=None,draw=False):
    if mode not in ("nominal","slippery","slow_motors","faulty"):raise ValueError(mode)
    if not 3.96<=duration<=20:raise ValueError("4 sec minimum for 32-frame U2 sequence")
    model=mujoco.MjModel.from_xml_path(str(HERE/"biped.xml"))
    data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    if mode=="slippery":
        gi=mid(model,mujoco.mjtObj.mjOBJ_GEOM,"floor")
        model.geom_friction[gi,0]=.18
    rng=np.random.default_rng(seed)
    pelvis_body=mid(model,mujoco.mjtObj.mjOBJ_BODY,"pelvis")
    bodyids={s:[mid(model,mujoco.mjtObj.mjOBJ_BODY,n)
                for n in ("pelvis",s+"_thigh",s+"_shank",s+"_foot")] for s in "LR"}
    dt=float(model.opt.timestep)
    steps=int(math.ceil(duration/dt))
    stride=int(round(PERIOD/dt))
    samples=[];geometry=[]
    for step in range(steps):
        control(model,data,float(data.time),mode)
        mujoco.mj_step(model,data)
        if not np.isfinite(data.qpos).all():raise FloatingPointError("nonfinite state")
        if step%stride:continue
        left=taxels(model,data,"L")
        right=taxels(model,data,"R")
        x=float(data.qpos[0])
        # Placeholder thermal FIELD, not a native MuJoCo thermal simulation.
        temp=23+3*math.exp(-((x-.42)/.18)**2)+.012*rng.normal()
        sample={"t_s":float(data.time),"x_m":x,
                "pelvis_height_m":float(data.xpos[pelvis_body,2]),
                "pelvis_pitch_rad":float(data.qpos[2]),
                "left_taxels_N":left.tolist(),"right_taxels_N":right.tolist(),
                "left_force_N":float(sum(left)),"right_force_N":float(sum(right)),
                "imu_gyro_rad_s":sensed(model,data,"pelvis_gyro"),
                "imu_accel_m_s2":sensed(model,data,"pelvis_accelerometer"),
                "temperature_simulated_C":float(temp),
                "motor_torques_Nm":data.ctrl.copy().tolist()}
        for jointname in JOINTS:
            sample[jointname+"_rad"]=joint(model,data,jointname)[0]
        samples.append(sample)
        geometry.append({s:np.asarray(data.xpos[bodyids[s]]).copy().tolist() for s in "LR"})
    obs,health,windows,masks=to_u2(samples,mode)
    left=np.array([x["left_force_N"] for x in samples])
    right=np.array([x["right_force_N"] for x in samples])
    metrics={
        "physics_engine":"MuJoCo","source":"synthetic, no physical hardware",
        "gait":"harness-assisted scripted stepping, not autonomous walking",
        "mode":mode,"duration_s":float(data.time),
        "physics_timestep_s":dt,"observation_period_s":PERIOD,
        "observations":len(samples),"u2_32_frame_windows":len(windows),
        "root_displacement_m":float(samples[-1]["x_m"]-samples[0]["x_m"]),
        "minimum_pelvis_height_m":float(min(x["pelvis_height_m"] for x in samples)),
        "left_contact_frames":int((left>.15).sum()),
        "right_contact_frames":int((right>.15).sum()),
        "max_contact_force_N":float(max(left+right)),
        "gyro_channels":3,"accelerometer_channels":3,
        "joint_encoder_channels":6,"contact_taxels":8,
        "temperature_is_analytic_proxy":True,"masked_modalities":int((health==0).sum()),
        "real_sensor_records":0,"trained_U2_inference":False}
    if out_dir:
        out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
        (out/(mode+"_metrics.json")).write_text(json.dumps(metrics,indent=2))
        (out/(mode+"_samples.json")).write_text(json.dumps(samples))
        np.savez_compressed(out/(mode+"_u2_sensors.npz"),
                            obs=windows,quality=masks,raw=obs,raw_health=health,
                            context=np.zeros((len(windows),4,8),dtype=np.float32),
                            action_placeholder=np.zeros((len(windows),4,3),dtype=np.float32))
        if draw:render(samples,geometry,out)
    return metrics,samples

def render(samples,geometry,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,PillowWriter
    fig,ax=plt.subplots(figsize=(8,5))
    keys=list(range(0,len(samples),3))
    def frame(j):
        i=keys[j];ax.clear();ax.axhline(0,color="gray",linewidth=2)
        x=samples[i]["x_m"]
        for side,color in (("L","#299ed1"),("R","#e8803b")):
            points=np.array(geometry[i][side])
            ax.plot(points[:,0],points[:,2],"-o",color=color,lw=6,ms=6)
        ax.set_xlim(x-.7,x+.7);ax.set_ylim(-.1,1.6)
        ax.set_aspect("equal");ax.grid(alpha=.2)
        ax.set_title("MuJoCo assisted biped, t={:.2f}s".format(samples[i]["t_s"]))
        ax.set_xlabel("ground distance (m)");ax.set_ylabel("height (m)")
    a=FuncAnimation(fig,frame,frames=len(keys),interval=125)
    a.save(str(out/"assisted_biped.gif"),writer=PillowWriter(fps=8))
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    ax.plot([r["t_s"] for r in samples],[r["left_force_N"] for r in samples],label="Left contact")
    ax.plot([r["t_s"] for r in samples],[r["right_force_N"] for r in samples],label="Right contact")
    ax.set_xlabel("Time (s)");ax.set_ylabel("Solver normal force (N)")
    ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    fig.savefig(out/"contact_trace.png",dpi=140);plt.close(fig)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",default="nominal",choices=("nominal","slippery","slow_motors","faulty"))
    ap.add_argument("--duration",default=5.76,type=float)
    ap.add_argument("--seed",type=int,default=19)
    ap.add_argument("--out",default=str(HERE/"results"))
    ap.add_argument("--draw",action="store_true")
    args=ap.parse_args()
    metrics,_=simulate(args.mode,args.duration,args.seed,args.out,args.draw)
    print("SOMA_SIM_RESULT "+json.dumps(metrics,sort_keys=True),flush=True)
