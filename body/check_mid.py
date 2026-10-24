"""Physics checks for the v3 / mid duck: stand hold, spring-neck sag, head spin, jaw.

    python body/check_mid.py [model/duck_mid.xml]
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "model" / "duck_mid.xml"
m = mujoco.MjModel.from_xml_path(str(path))
d = mujoco.MjData(m)


def q(name):
    return d.qpos[m.jnt_qposadr[m.joint(name).id]]


def start(rng=None):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    if rng is not None:
        d.qpos[7:] += rng.normal(0, 0.05, m.nq - 7)
        ax = rng.normal(size=2) * np.radians(3) / np.sqrt(2)
        d.qpos[3:7] = [1, ax[0] / 2, ax[1] / 2, 0]
        d.qpos[3:7] /= np.linalg.norm(d.qpos[3:7])
    d.ctrl[:] = m.key("home").ctrl
    mujoco.mj_forward(m, d)
    # put the soles on the floor
    low = min(d.geom_xpos[g][2] - 0.02 for g in range(m.ngeom) if "foot_bottom_tpu" in (m.geom(g).name or ""))
    d.qpos[2] -= low + 0.0
    mujoco.mj_forward(m, d)
    for _ in range(200):
        bad = d.ncon and min(c.dist for c in d.contact[:d.ncon]) < -0.002
        if not bad:
            break
        d.qpos[2] += 0.001
        mujoco.mj_forward(m, d)


def tilt():
    return np.degrees(np.arccos(np.clip(d.xmat[m.body("trunk_assembly").id][8], -1, 1)))


def run(seconds, fn=None):
    worst = 0
    for i in range(int(seconds / m.opt.timestep)):
        if fn:
            fn(i * m.opt.timestep)
        mujoco.mj_step(m, d)
        worst = max(worst, tilt())
    return worst


print(f"model {path.name}: mass {m.body_subtreemass[1]:.3f} kg, timestep {m.opt.timestep}")
start()
run(2)
hb = m.body("head_assembly").id
top = max(d.geom_xpos[g][2] for g in range(m.ngeom) if m.geom_bodyid[g] == hb)
print(f"stand: base {d.qpos[2]*1000:.0f} mm, head centre {d.xpos[hb][2]*1000:.0f} mm, tilt {tilt():.2f} deg")
print(f"spring neck: neck_pitch {np.degrees(q('neck_pitch')):.1f} deg (spring rest "
      f"{np.degrees(m.qpos_spring[m.jnt_qposadr[m.joint('neck_pitch').id]]):.1f}), "
      f"head_roll {np.degrees(q('head_roll')):.1f} deg; neck range "
      f"{np.degrees(m.jnt_range[m.joint('neck_pitch').id]).round(0)}")
lead = np.array([d.actuator_force[i] for i in range(m.nu)])
print("holding torque N·m:", ", ".join(f"{m.actuator(i).name} {lead[i]:+.3f}" for i in range(m.nu)),
      f"(limit {m.actuator_forcerange[0,1]:.2f})")

rng = np.random.default_rng(0)
falls, worst = 0, 0
for t in range(10):
    start(rng)
    w = run(5)
    worst = max(worst, w)
    falls += tilt() > 30
print(f"stand hold 10 x 5 s from random starts: {falls} falls, worst tilt {worst:.1f} deg")

start()
yaw_ctrl = m.actuator("head_yaw").id
w = run(6, lambda t: d.ctrl.__setitem__(yaw_ctrl, np.pi * np.sin(2 * np.pi * t / 3)))
print(f"head spinning ±180 deg (3 s period) while standing: worst tilt {w:.1f} deg, fell: {tilt() > 30}")

start()
jaw = m.actuator("jaw").id
for target in (m.actuator_ctrlrange[jaw, 1], 0.0):
    d.ctrl[jaw] = target
    run(0.5)
    print(f"jaw command {np.degrees(target):.0f} deg -> {np.degrees(q('jaw')):.1f} deg")
