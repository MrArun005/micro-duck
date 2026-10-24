"""Find poses the duck can actually rest in, by trying them in physics (lesson from the Bolt dog run:
measure target poses, never guess them).

    python body/poses.py [--model model/duck_mid.xml]      -> model/pose_search_mid.json

Chosen targets are stored in model/poses_mid.json (skills_env.py reads them).

For "sit" the duck should rest its weight on its folded legs (shins or thighs on the floor, or the
belly on its legs), not hold a crouch with the servos. Each candidate (hip pitch, knee, ankle; both
legs mirrored, head at home) is dropped from just above the floor, held by the servos for 3 s, and
scored: body upright, which parts touch the floor, how much the leg servos must hold, and whether it
stays put over the last second.
"""
import argparse
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent


def leg_q(m, q, hip, knee, ankle):
    """Set both legs: left as given, right mirrored like the home keyframe (hip pitch sign flips)."""
    for side, s in (("left", 1), ("right", -1)):
        q[m.jnt_qposadr[m.joint(f"{side}_hip_pitch").id]] = s * hip
        q[m.jnt_qposadr[m.joint(f"{side}_knee").id]] = knee
        q[m.jnt_qposadr[m.joint(f"{side}_ankle").id]] = ankle
    return q


def try_pose(m, d, hip, knee, ankle, seconds=3.0):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    leg_q(m, d.qpos, hip, knee, ankle)
    acts = [m.actuator(i) for i in range(m.nu)]
    for i, a in enumerate(acts):
        d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
    mujoco.mj_forward(m, d)
    low = min(_bottom(m, d, g) for g in range(m.ngeom) if m.geom_bodyid[g] > 0 and m.geom_conaffinity[g] + m.geom_contype[g] > 0 and m.geom_group[g] in (0, 3, 1))
    d.qpos[2] -= low - 0.003
    trunk = m.body("trunk_assembly").id
    legs = [i for i, a in enumerate(acts) if any(k in a.name for k in ("hip", "knee", "ankle"))]
    n = int(seconds / m.opt.timestep)
    pos_hist, torque = [], 0.0
    for k in range(n):
        mujoco.mj_step(m, d)
        if k >= n - int(1.0 / m.opt.timestep):
            pos_hist.append(d.xpos[trunk].copy())
            torque += np.abs(d.actuator_force[legs]).mean()
    torque /= len(pos_hist)
    up = float(d.xmat[trunk][8])
    floor = m.geom("floor").id
    touch = set()
    for c in d.contact[:d.ncon]:
        if floor in (c.geom[0], c.geom[1]):
            other = c.geom[1] if c.geom[0] == floor else c.geom[0]
            touch.add(_part(m.body(m.geom_bodyid[other]).name))
    drift = float(np.linalg.norm(pos_hist[-1] - pos_hist[0]))
    return dict(hip=round(hip, 3), knee=round(knee, 3), ankle=round(ankle, 3), up=round(up, 3),
                height=round(float(d.qpos[2]), 4), trunk_bottom=round(min(_bottom(m, d, g) for g in range(m.ngeom) if m.geom_bodyid[g] == trunk and m.geom_contype[g] + m.geom_conaffinity[g] > 0), 4),
                touch=sorted(touch), torque=round(float(torque), 4), drift=round(drift, 4),
                qpos=d.qpos.copy().round(5).tolist())


def _part(body):
    if body.startswith("foot"):
        return "feet"
    if body in ("knee_and_ankle_assembly_2", "knee_and_ankle_assembly_4"):
        return "shins"
    if body in ("knee_and_ankle_assembly", "knee_and_ankle_assembly_3"):
        return "thighs"
    if body == "trunk_assembly":
        return "belly"
    if body in ("head_assembly", "jaw"):
        return "head"
    return body


def _bottom(m, d, g):
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        return (v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min()
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
        return np.inf
    return d.geom_xpos[g][2] - m.geom_rbound[g]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/duck_mid.xml")
    ap.add_argument("--out", default="model/pose_search_mid.json")
    a = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(str(HERE / a.model))
    d = mujoco.MjData(m)
    lo = {n: m.actuator_ctrlrange[m.actuator(f"left_{n}").id] for n in ("hip_pitch", "knee", "ankle")}
    hips = np.linspace(lo["hip_pitch"][0], lo["hip_pitch"][1], 9)
    knees = np.linspace(1.4, lo["knee"][1], 7)
    ankles = np.linspace(lo["ankle"][0], lo["ankle"][1], 11)
    home = try_pose(m, d, *[m.key("home").qpos[m.jnt_qposadr[m.joint(f"left_{n}").id]] for n in ("hip_pitch", "knee", "ankle")])
    res = [try_pose(m, d, h, k, an) for h, k, an in itertools.product(hips, knees, ankles)]
    rests = [r for r in res if r["up"] > 0.85 and r["drift"] < 0.003 and ({"shins", "thighs", "belly"} & set(r["touch"]))]
    rests.sort(key=lambda r: (r["torque"], r["trunk_bottom"]))
    out = dict(home=home, rest_on_legs=rests[:20], tried=len(res))
    (HERE / a.out).write_text(json.dumps(out, indent=1))
    print(f"home: height {home['height']}, leg torque {home['torque']} N·m, touching {home['touch']}")
    print(f"tried {len(res)} poses, {len(rests)} rest on the legs/belly upright and still; best:")
    for r in rests[:8]:
        print(f"  hip {r['hip']:+.2f} knee {r['knee']:.2f} ankle {r['ankle']:+.2f} | up {r['up']:.2f} "
              f"height {r['height']*1000:.0f} mm, belly {r['trunk_bottom']*1000:.0f} mm off floor | "
              f"touch {r['touch']} | leg torque {r['torque']:.3f} N·m")


if __name__ == "__main__":
    main()
