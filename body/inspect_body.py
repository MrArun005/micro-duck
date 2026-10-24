"""Print the duck body's specs: masses, inertias, joints, motors, leg geometry, stance.

    python body/inspect_body.py                    # the built model (body/model/duck.xml)
    python body/inspect_body.py --upstream         # the untouched upstream model
    python body/inspect_body.py --json specs.json  # also save everything as JSON

(Named inspect_body.py, not inspect.py: a file called inspect.py in this folder would shadow
Python's standard `inspect` module and break numpy/torch imports for every script here.)

Everything is measured from the simulated model itself (never typed in by hand), so the
numbers stay true when tweaks.yaml changes the body.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
BUILT = HERE / "model" / "duck.xml"
UPSTREAM = HERE / "model" / "upstream" / "scene_flat_terrain.xml"

LEG_JOINTS = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"]
HEAD_JOINTS = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
FOOT_GEOMS = ["left_foot_bottom_tpu", "right_foot_bottom_tpu"]
NM_PER_KGCM = 0.0980665


def load(path=None):
    path = Path(path) if path else BUILT
    m = mujoco.MjModel.from_xml_path(str(path))
    return m, mujoco.MjData(m)


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def geom_world_vertices(m, d, gid):
    """World-frame vertices of a mesh geom (or the 8 box corners for primitives)."""
    if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[gid]
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    else:
        c, h = m.geom_aabb[gid, :3], m.geom_aabb[gid, 3:]
        v = np.array([c + h * (2 * np.array(s) - 1) for s in np.ndindex(2, 2, 2)])
    R = d.geom_xmat[gid].reshape(3, 3)
    return v @ R.T + d.geom_xpos[gid]


def lowest_point(m, d, geoms):
    return min(geom_world_vertices(m, d, m.geom(g).id)[:, 2].min() for g in geoms)


def home_pose(m, d, settle=True):
    """Put the robot in its 'home' keyframe, soles exactly on the floor; optionally let it settle 1 s."""
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    d.qpos[2] -= lowest_point(m, d, FOOT_GEOMS)
    mujoco.mj_forward(m, d)
    if settle:
        mujoco.mj_step(m, d, nstep=int(1.0 / m.opt.timestep))
    return d


def support_polygon(m, d):
    """Convex hull (x, y) of both soles' vertices that touch the floor (within 2 mm)."""
    pts = []
    for g in FOOT_GEOMS:
        v = geom_world_vertices(m, d, m.geom(g).id)
        pts.append(v[v[:, 2] < v[:, 2].min() + 0.002, :2])
    pts = np.concatenate(pts)
    # Andrew's monotone chain
    pts = np.unique(pts.round(5), axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(points):
        h = []
        for p in points:
            while len(h) >= 2 and cross2(h[-1] - h[-2], p - h[-2]) <= 0:
                h.pop()
            h.append(p)
        return h
    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def com_margin(poly, p):
    """Signed distance (m) from point p to the polygon edge; positive = inside."""
    dmin = np.inf
    inside = True
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        e = b - a
        cross = cross2(e, p - a)
        inside &= cross >= 0
        t = np.clip(np.dot(p - a, e) / np.dot(e, e), 0, 1)
        dmin = min(dmin, np.linalg.norm(a + t * e - p))
    return dmin if inside else -dmin


def specs(m, d):
    out = {"bodies": [], "joints": []}
    for b in range(1, m.nbody):
        if m.body_mass[b] < 1e-6:
            continue
        out["bodies"].append(dict(name=m.body(b).name, mass=float(m.body_mass[b]),
                                  inertia=m.body_inertia[b].tolist(),
                                  com_local=m.body_ipos[b].tolist()))
    out["total_mass"] = float(m.body_mass.sum())
    for a in range(m.nu):
        j = m.actuator_trnid[a, 0]
        dof = m.jnt_dofadr[j]
        kp = float(m.actuator_gainprm[a, 0])
        out["joints"].append(dict(
            name=m.joint(j).name, range_deg=np.degrees(m.actuator_ctrlrange[a]).round(1).tolist(),
            hard_stop_deg=np.degrees(m.jnt_range[j]).round(1).tolist(),
            actuator=m.actuator(a).name, kp=kp, kv=float(abs(m.actuator_biasprm[a, 2])),
            force_limit=float(m.actuator_forcerange[a, 1]), armature=float(m.dof_armature[dof]),
            damping=float(m.dof_damping[dof]), frictionloss=float(m.dof_frictionloss[dof])))

    # Leg geometry, measured between joint axes in the zero pose.
    d0 = mujoco.MjData(m)
    mujoco.mj_forward(m, d0)
    jp = lambda n: d0.xanchor[m.joint(n).id].copy()
    side = "left"
    hip, knee, ankle = jp(f"{side}_hip_pitch"), jp(f"{side}_knee"), jp(f"{side}_ankle")
    sole = lowest_point(m, d0, [f"{side}_foot_bottom_tpu"])
    out["leg"] = dict(thigh=float(np.linalg.norm(knee - hip)), shin=float(np.linalg.norm(ankle - knee)),
                      ankle_to_sole=float(ankle[2] - sole), hip_pitch_to_sole_straight=float(hip[2] - sole),
                      hip_yaw_to_sole_straight=float(jp(f"{side}_hip_yaw")[2] - sole),
                      hip_spacing=float(abs(jp("left_hip_yaw")[1] - jp("right_hip_yaw")[1])))
    gid = m.geom(FOOT_GEOMS[0]).id
    v = geom_world_vertices(m, d0, gid)
    ext = v.max(0) - v.min(0)
    out["foot"] = dict(length=float(ext[0]), width=float(ext[1]), sole_thickness=float(ext[2]),
                       friction=m.geom_friction[gid].tolist())

    # Standing (home keyframe, settled 1 s).
    home_pose(m, d)
    poly = support_polygon(m, d)
    com = d.subtree_com[m.body("trunk_assembly").id]
    out["stand"] = dict(
        base_height=float(d.qpos[2]), trunk_site_height=float(d.site("trunk").xpos[2]),
        head_top=float(max(geom_world_vertices(m, d, g)[:, 2].max() for g in range(m.ngeom)
                           if m.geom_bodyid[g] == m.body("head_assembly").id)),
        com=com.tolist(), com_height=float(com[2]),
        support_x=[float(poly[:, 0].min()), float(poly[:, 0].max())],
        support_y=[float(poly[:, 1].min()), float(poly[:, 1].max())],
        com_margin=float(com_margin(poly, com[:2])),
        joint_torque=dict(zip([m.actuator(a).name for a in range(m.nu)], d.actuator_force.round(3).tolist())),
    )
    return out


def print_specs(s):
    print(f"{'body':32s} {'mass kg':>8s}  inertia diag (kg·m²)")
    for b in s["bodies"]:
        print(f"{b['name']:32s} {b['mass']:8.4f}  " + " ".join(f"{x:.2e}" for x in b["inertia"]))
    print(f"TOTAL MASS {s['total_mass']:.3f} kg\n")
    print(f"{'joint':18s} {'servo range°':>16s} {'hard stop°':>12s} {'kp':>6s} {'kv':>5s} {'τmax N·m':>8s} {'(kg·cm)':>7s} "
          f"{'armature':>8s} {'damping':>7s} {'friction':>8s}")
    for j in s["joints"]:
        r = f"{j['range_deg'][0]:.0f} … {j['range_deg'][1]:.0f}"
        h = f"{j['hard_stop_deg'][0]:.0f} … {j['hard_stop_deg'][1]:.0f}"
        print(f"{j['name']:18s} {r:>16s} {h:>12s} {j['kp']:6.2f} {j['kv']:5.2f} {j['force_limit']:8.2f} "
              f"{j['force_limit'] / NM_PER_KGCM:7.1f} {j['armature']:8.3f} {j['damping']:7.2f} {j['frictionloss']:8.3f}")
    L, F, S = s["leg"], s["foot"], s["stand"]
    print(f"\nleg: thigh {L['thigh']*1000:.1f} mm, shin {L['shin']*1000:.1f} mm, ankle→sole {L['ankle_to_sole']*1000:.1f} mm, "
          f"hip-pitch→sole straight {L['hip_pitch_to_sole_straight']*1000:.1f} mm, hip spacing {L['hip_spacing']*1000:.0f} mm")
    print(f"foot: {F['length']*1000:.0f} × {F['width']*1000:.0f} mm, sole {F['sole_thickness']*1000:.1f} mm, friction {F['friction']}")
    print(f"stand (home, settled 1 s): base {S['base_height']*1000:.0f} mm, trunk site {S['trunk_site_height']*1000:.0f} mm, "
          f"head top {S['head_top']*1000:.0f} mm")
    print(f"  CoM ({S['com'][0]*1000:.0f}, {S['com'][1]*1000:.0f}, {S['com'][2]*1000:.0f}) mm; support x {S['support_x'][0]*1000:.0f}…{S['support_x'][1]*1000:.0f} mm, "
          f"y {S['support_y'][0]*1000:.0f}…{S['support_y'][1]*1000:.0f} mm; CoM margin to edge {S['com_margin']*1000:.0f} mm")
    print("  holding torque N·m: " + ", ".join(f"{k} {v:+.2f}" for k, v in S["joint_torque"].items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", action="store_true", help="inspect the untouched upstream model")
    ap.add_argument("--model", help="any MJCF path")
    ap.add_argument("--json", help="write specs as JSON here")
    a = ap.parse_args()
    m, d = load(a.model or (UPSTREAM if a.upstream else BUILT))
    s = specs(m, d)
    print_specs(s)
    if a.json:
        Path(a.json).write_text(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
