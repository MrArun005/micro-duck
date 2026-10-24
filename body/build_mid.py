"""Build the v3 duck (owner's package in v3/) with a proper mouth, full size and mid size (0.7x).

    python body/build_mid.py        -> model/duck_v3.xml (full size) and model/duck_mid.xml (0.7x)

Starts from model/duck_v3_upload.xml (the owner's v3: head yaw ±180°, passive spring neck pitch and
head roll, jaw servo) and:
  0. Neck: spring 4x stiffer (see NECK_SPRING_X). Knees fold to 125° (sit, sleep, lie).
  1. Mouth: replaces the flat plate that stuck out of the face with a duck bill. A fixed upper bill on
     the lower front of the head, and a lower jaw hinged 9 mm inside the face at the bottom of the
     head, so it closes flush under the upper bill and opens downward (0-35°) on a small servo.
  2. Mid size: every length x0.7 (positions, meshes), mass scaled to 0.80 kg (estimate, the same
     class as Pollen's 25 cm / 800 g Microduck; weigh the real parts and fix it), inertia x mass x 0.7²,
     leg/head servos swapped for Dynamixel XL330 (damping, friction, armature and torque limit from
     the Microduck project's measured model, microduck_rl joints_properties.xml). Stiffness kp 2.4
     N·m/rad, about the XL330's default firmware position gain: Microduck's kp 0.55 (firmware gain
     200) let this heavier-headed body sink and fall in every stand test, 2.4 holds it (check_mid.py).
     Spring neck stiffness scaled by mass x length, so the head sags about the same.
"""
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "model" / "duck_v3_upload.xml"
FULL = HERE / "model" / "duck_v3.xml"
MID = HERE / "model" / "duck_mid.xml"

S = 0.7                       # mid size: lengths
MID_MASS = 0.80               # kg, estimate
XL330 = dict(kp=2.4, force=0.96, damping=0.053, frictionloss=0.0048, armature=0.0018)
JAW = dict(kp=2.0, force=0.2, damping=0.02, armature=0.0005, range=0.61)   # small hobby servo, 35°

# Mouth in world coordinates at the home pose, full size (head shell: front face x=101 mm, bottom z=341 mm).
UPPER_BILL = dict(centre=[0.123, 0, 0.357], half=[0.030, 0.055, 0.009])
JAW_HINGE = [0.092, 0, 0.346]
LOWER_BILL = dict(offset=[0.030, 0, 0.0], half=[0.030, 0.050, 0.006], mass=0.012)
BILL_RGBA = "1 0.62 0.1 1"
# The uploaded spring (0.8 N·m/rad) can't hold the head once it turns: the head's centre of mass is off
# the yaw axis, so at 90° the neck tips 25° and at 180° the whole duck falls over (check_mid.py).
# 4x stiffer keeps the neck within ±4° at any head angle, at both sizes.
NECK_SPRING_X = 4.0
# Knees: ±90° only allows a 12 mm crouch; sit, sleep and lie need them folded. Nothing collides up to
# 126° (REVIEW.md #6), so allow 125°, rigid stop 5° beyond. Check on the real leg before trusting it.
KNEE_MAX_DEG = 125


def fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def head_frame():
    m = mujoco.MjModel.from_xml_path(str(SRC))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    b = m.body("head_assembly").id
    return d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()


def to_head(p, R, w):
    return R.T @ (np.asarray(w) - p)


def stiffen_neck(root):
    for j in root.iter("joint"):
        if j.get("name") == "neck_pitch":
            j.set("stiffness", f"{float(j.get('stiffness')) * NECK_SPRING_X:.4g}")
            j.set("damping", f"{float(j.get('damping')) * 2:.4g}")


def open_knees(root):
    hi = np.radians(KNEE_MAX_DEG)
    for j in root.iter("joint"):
        if j.get("name") in ("left_knee", "right_knee"):
            lo = float(j.get("range").split()[0])
            j.set("range", f"{lo:.6g} {hi + np.radians(5):.6g}")
    for a in root.find("actuator"):
        if a.get("name") in ("left_knee", "right_knee"):
            lo = float(a.get("ctrlrange").split()[0])
            a.set("ctrlrange", f"{lo:.6g} {hi:.6g}")


def fix_mouth(root):
    p, R = head_frame()
    head = next(b for b in root.iter("body") if b.get("name") == "head_assembly")
    old = head.find("body[@name='jaw']")
    if old is not None:
        head.remove(old)
    # The bill's ellipsoid axes are world-aligned at home; express that rotation in the head frame.
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.T.flatten())
    ET.SubElement(head, "geom", {
        "name": "upper_bill", "type": "ellipsoid", "size": fmt(UPPER_BILL["half"]),
        "pos": fmt(to_head(p, R, UPPER_BILL["centre"])), "quat": fmt(q), "rgba": BILL_RGBA,
        "contype": "4", "conaffinity": "1", "group": "1", "density": "0", "friction": "0.6"})
    hinge = to_head(p, R, JAW_HINGE)
    jaw = ET.SubElement(head, "body", {"name": "jaw", "pos": fmt(hinge), "quat": fmt(q)})
    # In the jaw frame the axes are world-aligned at home: +x forward, +y left, +z up.
    # Rotating about +y moves the tip down, so positive = open.
    ET.SubElement(jaw, "inertial", {"pos": fmt(LOWER_BILL["offset"]), "mass": str(LOWER_BILL["mass"]),
                                    "diaginertia": "2e-6 3e-6 4e-6"})
    ET.SubElement(jaw, "joint", {"name": "jaw", "axis": "0 1 0", "range": f"0 {JAW['range']}",
                                 "damping": str(JAW["damping"]), "armature": str(JAW["armature"])})
    ET.SubElement(jaw, "geom", {
        "name": "lower_bill", "type": "ellipsoid", "size": fmt(LOWER_BILL["half"]),
        "pos": fmt(LOWER_BILL["offset"]), "rgba": BILL_RGBA, "contype": "2", "conaffinity": "1",
        "group": "1", "density": "0", "friction": "0.6"})
    act = root.find("actuator")
    for a in list(act):
        if a.get("name") == "jaw":
            act.remove(a)
    ET.SubElement(act, "position", {"name": "jaw", "joint": "jaw", "kp": str(JAW["kp"]),
                                    "ctrlrange": f"0 {JAW['range']}", "forcerange": f"-{JAW['force']} {JAW['force']}"})


def scale_xyz(el, attr, s):
    if el.get(attr):
        el.set(attr, fmt(np.array(el.get(attr).split(), float) * s))


def make_mid(root):
    root.set("model", "duck_mid")
    total = sum(float(i.get("mass")) for i in root.iter("inertial"))
    k = MID_MASS / total
    for m in root.find("asset").iter("mesh"):
        m.set("scale", fmt([S] * 3))
    for el in root.find("worldbody").iter():
        if el.tag in ("body", "geom", "site", "inertial"):
            scale_xyz(el, "pos", S)
        if el.tag == "geom" and el.get("type") in ("ellipsoid", "box", "capsule", "sphere", "cylinder"):
            scale_xyz(el, "size", S)
        if el.tag == "inertial":
            el.set("mass", f"{float(el.get('mass')) * k:.6g}")
            scale_xyz(el, "diaginertia", k * S * S)
        if el.tag == "joint" and el.get("stiffness"):          # spring holds a head k*S times lighter*shorter
            el.set("stiffness", f"{float(el.get('stiffness')) * k * S:.4g}")
            el.set("damping", f"{float(el.get('damping')) * k * S:.4g}")
    for d in root.find("default").iter("default"):
        if d.get("class") == "sts3215":
            d.set("class", "sts3215")      # keep the class name so joints/actuators still resolve
            j, g = d.find("joint"), d.find("general")
            j.set("armature", str(XL330["armature"]))
            j.set("damping", str(XL330["damping"]))
            j.set("frictionloss", str(XL330["frictionloss"]))
            g.set("forcerange", f"-{XL330['force']} {XL330['force']}")
            g.set("gainprm", str(XL330["kp"]))
            g.set("biasprm", f"0 -{XL330['kp']}")
    key = root.find("keyframe").find("key")
    q = key.get("qpos").split()
    q[2] = f"{float(q[2]) * S:.6g}"
    key.set("qpos", " ".join(q))
    return k


def main():
    tree = ET.parse(SRC)
    root = tree.getroot()
    fix_mouth(root)
    stiffen_neck(root)
    open_knees(root)
    tree.write(FULL)
    mid = copy.deepcopy(root)
    k = make_mid(mid)
    ET.ElementTree(mid).write(MID)
    for path in (FULL, MID):
        m = mujoco.MjModel.from_xml_path(str(path))
        print(f"{path.name}: {m.body_subtreemass[1]:.3f} kg, {m.nu} servos: "
              + ", ".join(m.actuator(i).name for i in range(m.nu)))
    print(f"mid: lengths x{S}, mass x{k:.3f}, XL330 servos")


if __name__ == "__main__":
    main()
