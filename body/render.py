"""Render preview images of the built duck body into body/renders/.

    MUJOCO_GL=osmesa python body/render.py

Views: front, side, 3/4 for standing (home keyframe, settled) and the deepest crouch the
model allows; plus the optional 125° knee crouch, the collision parts, and lying on its back.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_model  # noqa: E402
from checks import leg_pose, place  # noqa: E402
from inspect_body import BUILT, home_pose  # noqa: E402

OUT = HERE / "renders"
W, H = 900, 700
VIEWS = {"front": 180, "side": 90, "34": 140}


def lit_model(path):
    spec = mujoco.MjSpec.from_file(str(path))
    spec.visual.headlight.ambient = [0.35, 0.35, 0.37]
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]
    spec.visual.quality.shadowsize = 4096
    spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
    spec.worldbody.add_light(pos=[0.6, -0.6, 1.4], dir=[-0.4, 0.4, -1], diffuse=[0.55, 0.55, 0.55], castshadow=True)
    spec.worldbody.add_light(pos=[-0.8, 0.5, 1.0], dir=[0.6, -0.3, -0.7], diffuse=[0.25, 0.25, 0.28], castshadow=False)
    for t in spec.textures:
        if t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            t.rgb1, t.rgb2 = [0.97, 0.97, 0.98], [0.90, 0.91, 0.93]
    return spec.compile()


def shoot(m, d, name, azimuth, lookat_z=0.2, distance=0.85, elevation=-12, collision=False):
    r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [d.qpos[0] - 0.02, d.qpos[1], lookat_z]
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
    opt = mujoco.MjvOption()
    if collision:
        opt.geomgroup[3] = 1
    r.update_scene(d, cam, opt)
    Image.fromarray(r.render()).save(OUT / f"{name}.png", optimize=True)
    r.close()
    print("wrote", OUT / f"{name}.png")


def settle_crouch(m, d, knee):
    place(m, d, leg_pose(m, knee))
    mujoco.mj_step(m, d, nstep=int(1.5 / m.opt.timestep))


def main():
    OUT.mkdir(exist_ok=True)
    m = lit_model(BUILT)
    d = mujoco.MjData(m)

    home_pose(m, d, settle=True)
    for v, az in VIEWS.items():
        shoot(m, d, f"stand_{v}", az)
    shoot(m, d, "collision_34", VIEWS["34"], collision=True)

    kmax = m.actuator_ctrlrange[m.actuator("left_knee").id, 1]
    settle_crouch(m, d, kmax)
    for v, az in VIEWS.items():
        shoot(m, d, f"crouch_{v}", az)

    # Lying on its back (shows the new collision parts carrying the body).
    q = m.key("home").qpos.copy()
    q[2], q[3:7] = 0.25, [np.cos(-np.pi / 4), 0, np.sin(-np.pi / 4), 0]
    mujoco.mj_resetData(m, d)
    d.qpos[:], d.ctrl[:] = q, q[7:]
    mujoco.mj_step(m, d, nstep=1000)
    shoot(m, d, "lying_back_34", VIEWS["34"], lookat_z=0.08, distance=0.9, elevation=-25)

    # Optional tweak preview: knees allowed to 125° (tweaks.yaml section B).
    tw = yaml.safe_load((HERE / "tweaks.yaml").read_text())
    tw["joint_ranges_deg"] = {"left_knee": [-90, 125], "right_knee": [-90, 125]}
    tmp = HERE / "model" / "_render_knee125.xml"
    build_model.build(tw, tmp)
    try:
        m2 = lit_model(tmp)
        d2 = mujoco.MjData(m2)
        settle_crouch(m2, d2, np.radians(125))
        for v in ("side", "34"):
            shoot(m2, d2, f"crouch125_{v}", VIEWS[v], lookat_z=0.17)
    finally:
        tmp.unlink()


if __name__ == "__main__":
    main()
