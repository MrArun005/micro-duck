"""Build body/model/duck.xml = upstream Open Duck Mini v2 + the overrides in tweaks.yaml.

    python body/build_model.py                      # tweaks.yaml -> model/duck.xml
    python body/build_model.py --tweaks my.yaml --out model/duck_test.xml

The upstream files in model/upstream/ are never edited. Everything is done through MuJoCo's
MjSpec editing API, and every number that can be measured (box sizes, home height) is computed
from the model rather than typed in.
"""
import argparse
from pathlib import Path

import mujoco
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
UPSTREAM_SCENE = HERE / "model" / "upstream" / "scene_flat_terrain.xml"
LEG_JOINTS = [f"{s}_{j}" for s in ("left", "right") for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")]
BNO055_POS = [-0.08711, 0.0, 0.0417909]  # BNO055 board origin in the trunk frame (from the CAD)

# Contact bits (contype, conaffinity); two geoms touch if (ct1 & ca2) or (ct2 & ca1).
# Floor is (1, 0) upstream, soles (1, 1). Legs touch the floor and each other; trunk and head touch
# the floor and each other (head nodding into the body) but not the hips they are bolted to.
LEG_BITS = (2, 3)
BITS = {"trunk_assembly": (0, 5), "head_assembly": (4, 1)}


def build(tweaks, out):
    spec = mujoco.MjSpec.from_file(str(UPSTREAM_SCENE))
    ref = mujoco.MjModel.from_xml_path(str(UPSTREAM_SCENE))  # untouched, for measuring
    applied = []

    cp = tweaks.get("collision_parts") or {}
    if cp.get("enabled"):
        n = 0
        for body, meshes in cp["parts"].items():
            ct, ca = BITS.get(body, LEG_BITS)
            b = spec.body(body)
            for g in list(b.geoms):
                if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.meshname in meshes and g.classname.name == "visual":
                    # Same printed part, as a collider (MuJoCo collides with its convex hull).
                    b.add_geom(name=f"{body}_{g.meshname}_col{n}", type=mujoco.mjtGeom.mjGEOM_MESH,
                               meshname=g.meshname, pos=g.pos, quat=g.quat, contype=ct, conaffinity=ca,
                               group=3, rgba=[0.9, 0.3, 0.2, 0.35], friction=[0.6, 0.005, 0.0001],
                               density=0)  # zero density: the bodies keep their measured inertia
                    n += 1
        applied.append(f"{n} collision hulls on {len(cp['parts'])} parts")

    if tweaks.get("imu_at_bno055"):
        spec.site("imu").pos = BNO055_POS
        applied.append("IMU site at the BNO055 board")

    # Servo command range (ctrlrange) = the joint range the servo is allowed to go to.
    # Hard joint stop = command range + margin, so the servo (not a free rigid stop) carries the load.
    ranges = {a.target: np.array(spec.joint(a.target).range, dtype=float) for a in spec.actuators}
    for jname, (lo, hi) in (tweaks.get("joint_ranges_deg") or {}).items():
        ranges[jname] = np.radians([lo, hi])
        applied.append(f"{jname} range {lo}…{hi}°")
    margin = np.radians(tweaks.get("hard_stop_margin_deg") or 0)
    for a in spec.actuators:
        a.inheritrange = 0
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.ctrlrange = ranges[a.target]
        spec.joint(a.target).range = ranges[a.target] + np.array([-margin, margin])
    if margin:
        applied.append(f"hard joint stops {np.degrees(margin):.0f}° beyond the servo command range")

    servo = tweaks.get("servo") or {}
    fl = tweaks.get("force_limit_nm")
    for a in spec.actuators:
        if fl:
            a.forcerange = [-fl, fl]
        if "kp" in servo:
            a.gainprm[0] = servo["kp"]
            a.biasprm[1] = -servo["kp"]
    for j in spec.joints:
        if j.name in [a.target for a in spec.actuators]:
            if "damping" in servo:
                j.damping[0] = servo["damping"]
            for k in ("frictionloss", "armature"):
                if k in servo:
                    setattr(j, k, servo[k])
    if fl:
        applied.append(f"servo force limit ±{fl} N·m")
    if servo:
        applied.append(f"servo params {servo}")

    if tweaks.get("backlash"):
        for jname in LEG_JOINTS:
            j = spec.joint(jname)
            j.parent.add_joint(name=f"{jname}_backlash", type=mujoco.mjtJoint.mjJNT_HINGE, axis=j.axis,
                               range=[-np.radians(0.5), np.radians(0.5)], limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                               damping=[0.01] + [0] * 2, armature=0.01, frictionloss=0)
        applied.append("±0.5° backlash on leg joints")

    for body, kg in (tweaks.get("masses_kg") or {}).items():
        b = spec.body(body)
        r = kg / b.mass
        b.mass = kg
        b.fullinertia = np.asarray(b.fullinertia) * r
        b.inertia = np.asarray(b.inertia) * r
        applied.append(f"{body} mass {kg} kg")

    for p in tweaks.get("payloads") or []:
        child = spec.body(p["body"]).add_body(name=p["name"], pos=p["pos_m"])
        child.mass = p["mass_kg"]
        child.inertia = [1e-7] * 3
        child.explicitinertial = True
        applied.append(f"payload {p['name']} {p['mass_kg']} kg")

    if tweaks.get("floor_friction") is not None:
        spec.geom("floor").friction[0] = tweaks["floor_friction"]
    s = tweaks.get("solver") or {}
    spec.option.iterations = s.get("iterations", spec.option.iterations)
    spec.option.ls_iterations = s.get("ls_iterations", spec.option.ls_iterations)

    # Nicer neutral floor for renders (visual only).
    tex = spec.texture("groundplane")
    tex.rgb1, tex.rgb2, tex.mark = [0.86, 0.87, 0.89], [0.80, 0.81, 0.84], mujoco.mjtMark.mjMARK_EDGE
    tex.markrgb = [0.6, 0.62, 0.66]
    spec.modelname = "duckbody"

    # Rebuild the home keyframe by joint name (backlash adds joints), soles on the floor.
    key = spec.key("home")
    up_q, up_ctrl = ref.key("home").qpos.copy(), ref.key("home").ctrl.copy()
    m = spec.compile()
    q = np.zeros(m.nq)
    q[:7] = up_q[:7]
    for j in range(1, ref.njnt):
        q[m.jnt_qposadr[m.joint(ref.joint(j).name).id]] = up_q[ref.jnt_qposadr[j]]
    if tweaks.get("home_on_floor"):
        import inspect_body as ib
        d = mujoco.MjData(m)
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        q[2] -= ib.lowest_point(m, d, ib.FOOT_GEOMS)
        applied.append(f"home keyframe base height {q[2] * 1000:.1f} mm (soles on floor)")
    key.qpos = q
    key.ctrl = up_ctrl
    m = spec.compile()

    out.parent.mkdir(parents=True, exist_ok=True)
    header = ("<!-- GENERATED by body/build_model.py from model/upstream (Open Duck Mini v2, Apache-2.0)\n"
              "     + body/tweaks.yaml. Do not edit by hand. Applied:\n" +
              "".join(f"       - {a}\n" for a in applied) + "-->\n")
    xml = spec.to_xml()
    meshdir = (UPSTREAM_SCENE.parent / "assets").relative_to(out.resolve().parent, walk_up=True).as_posix()
    assert 'meshdir="assets"' in xml
    out.write_text(header + xml.replace('meshdir="assets"', f'meshdir="{meshdir}"', 1))
    mujoco.MjModel.from_xml_path(str(out))  # must load back
    return applied, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tweaks", default=str(HERE / "tweaks.yaml"))
    ap.add_argument("--out", default=str(HERE / "model" / "duck.xml"))
    a = ap.parse_args()
    tweaks = yaml.safe_load(Path(a.tweaks).read_text()) or {}
    applied, m = build(tweaks, Path(a.out))
    print(f"wrote {a.out}: {m.nbody} bodies, {m.njnt} joints, {m.nu} actuators, {m.body_mass.sum():.3f} kg")
    for x in applied:
        print("  +", x)


if __name__ == "__main__":
    main()
