"""Physics checks a hardware engineer would do before training anything on this body.

    python body/checks.py                 # built model, writes body/checks.json
    python body/checks.py --upstream      # untouched upstream model, for comparison

1. Stand hold: start from the home pose with small random joint and tilt errors, hold the
   pose with the servos for 5 s, report tilt and height drift (LESSONS.md section 1).
2. Crouch sweep: how low the body gets as the knees fold, and the holding torque it costs.
3. Self-collision scan (convex hulls of the visual parts, so it errs on the cautious side).
4. Lying test: put the robot on its back / front and see whether the body rests on the floor
   or sinks through it (only geoms with collision enabled can touch the floor).
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from inspect_body import BUILT, FOOT_GEOMS, HERE, UPSTREAM, geom_world_vertices, home_pose, load, lowest_point

STALL_7V4 = 19.5 * 0.0980665  # N·m, STS3215 7.4 V datasheet stall torque (C001 variant)


def tilt_deg(m, d):
    R = d.xmat[m.body("trunk_assembly").id].reshape(3, 3)
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))


def qadr(m, name):
    return m.jnt_qposadr[m.joint(name).id]


def leg_pose(m, knee, hip_frac=0.46, base=None):
    """Feet-flat crouch: knee flexion `knee`, hip takes `hip_frac` of it, ankle the rest."""
    q = (m.key("home").qpos if base is None else base).copy()
    h = knee * hip_frac
    for side, sign in (("left", 1), ("right", -1)):
        q[qadr(m, f"{side}_knee")] = knee
        q[qadr(m, f"{side}_hip_pitch")] = -sign * h
        q[qadr(m, f"{side}_ankle")] = -(knee - h)
    return q


def place(m, d, q, clip=True):
    mujoco.mj_resetData(m, d)
    d.qpos[:] = q
    if clip:
        for a in range(m.nu):
            i = m.jnt_qposadr[m.actuator_trnid[a, 0]]
            d.qpos[i] = np.clip(d.qpos[i], *m.actuator_ctrlrange[a])
    d.ctrl[:] = d.qpos[7:]
    mujoco.mj_forward(m, d)
    d.qpos[2] -= lowest_point(m, d, FOOT_GEOMS)
    mujoco.mj_forward(m, d)


def stand_hold(m, trials=10, secs=5.0, joint_noise=0.05, tilt_noise_deg=3.0, seed=0):
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    home_pose(m, d, settle=True)
    h_ref = float(d.qpos[2])
    target = m.key("home").ctrl.copy()
    rows = []
    for t in range(trials):
        q = m.key("home").qpos.copy()
        q[7:] += rng.normal(0, joint_noise, m.nu)
        ax = rng.normal(size=2)
        ax /= np.linalg.norm(ax)
        ang = np.radians(rng.uniform(-tilt_noise_deg, tilt_noise_deg))
        q[3:7] = [np.cos(ang / 2), ax[0] * np.sin(ang / 2), ax[1] * np.sin(ang / 2), 0]
        place(m, d, q)
        d.ctrl[:] = target
        tilt0, max_tilt, heights = tilt_deg(m, d), 0.0, []
        n = int(secs / m.opt.timestep)
        for i in range(n):
            mujoco.mj_step(m, d)
            if i == n // 5:
                xy1 = d.qpos[:2].copy()
            if i % 10 == 0:
                max_tilt = max(max_tilt, tilt_deg(m, d))
                heights.append(d.qpos[2])
        xy_drift = float(np.linalg.norm(d.qpos[:2]))
        slide = float(np.linalg.norm(d.qpos[:2] - xy1))
        rows.append(dict(start_tilt=tilt0, max_tilt=max_tilt, end_tilt=tilt_deg(m, d),
                         end_height_err_mm=(heights[-1] - h_ref) * 1000,
                         height_drift_last4s_mm=(heights[-1] - heights[len(heights) // 5]) * 1000,
                         xy_drift_mm=xy_drift * 1000, slide_last4s_mm=slide * 1000, fell=bool(tilt_deg(m, d) > 30 or d.qpos[2] < 0.6 * h_ref)))
    return dict(ref_height_mm=h_ref * 1000, trials=rows,
                summary=dict(max_tilt=max(r["max_tilt"] for r in rows), worst_end_tilt=max(r["end_tilt"] for r in rows),
                             worst_height_err_mm=max(abs(r["end_height_err_mm"]) for r in rows),
                             worst_drift_mm=max(abs(r["height_drift_last4s_mm"]) for r in rows),
                             worst_xy_drift_mm=max(r["xy_drift_mm"] for r in rows),
                             worst_slide_last4s_mm=max(r["slide_last4s_mm"] for r in rows),
                             falls=sum(r["fell"] for r in rows)))


def crouch_sweep(m, knees=(1.37, 1.57, 1.8, 2.0, 2.18)):
    d = mujoco.MjData(m)
    T = m.body("trunk_assembly").id
    kmax = m.actuator_ctrlrange[m.actuator("left_knee").id, 1]
    rows = []
    for k in knees:
        if k > kmax + 1e-6:
            rows.append(dict(knee_cmd_deg=float(np.degrees(k)), reachable=False))
            continue
        place(m, d, leg_pose(m, k))
        mujoco.mj_step(m, d, nstep=int(1.5 / m.opt.timestep))
        tb = min(geom_world_vertices(m, d, g)[:, 2].min() for g in range(m.ngeom) if m.geom_bodyid[g] == T)
        knee_tau = max(abs(d.actuator_force[m.actuator("left_knee").id]), abs(d.actuator_force[m.actuator("right_knee").id]))
        rows.append(dict(knee_cmd_deg=float(np.degrees(k)), reachable=bool(k <= kmax + 1e-6),
                         knee_actual_deg=float(np.degrees(d.qpos[qadr(m, "left_knee")])),
                         base_mm=float(d.qpos[2] * 1000), trunk_bottom_mm=float(tb * 1000), tilt=tilt_deg(m, d),
                         knee_torque=float(knee_tau),
                         max_torque=float(np.abs(d.actuator_force).max())))
    return rows


def sit_to_stand(m, durations=(1.0, 0.5, 0.3)):
    """Scripted rise from the deepest reachable crouch to home: peak servo torque and whether it stays up."""
    d = mujoco.MjData(m)
    kmax = min(m.actuator_ctrlrange[m.actuator("left_knee").id, 1], 2.05)
    start, goal = leg_pose(m, kmax), m.key("home").qpos.copy()
    rows = []
    for T in durations:
        place(m, d, start)
        mujoco.mj_step(m, d, nstep=int(1.0 / m.opt.timestep))
        peak, n = 0.0, int((T + 1.0) / m.opt.timestep)
        for i in range(n):
            a = min(i * m.opt.timestep / T, 1.0)
            d.ctrl[:] = (1 - a) * start[7:] + a * goal[7:]
            mujoco.mj_step(m, d)
            peak = max(peak, float(np.abs(d.actuator_force).max()))
        rows.append(dict(seconds=T, from_knee_deg=float(np.degrees(kmax)), peak_torque=peak,
                         saturated=bool(peak >= m.actuator_forcerange[:, 1].min() - 1e-3),
                         end_tilt=tilt_deg(m, d), end_height_mm=float(d.qpos[2] * 1000)))
    return rows


def self_collision_scan(path, knees=(1.37, 1.57, 1.8, 2.0, 2.2, 2.4, 2.6)):
    """Turn every visual part into a (convex-hull) collider and list touching body pairs per pose.
    Pairs that touch in every pose (parts that are bolted together) are reported once as 'always'."""
    spec = mujoco.MjSpec.from_file(str(path))
    for g in spec.geoms:
        if g.name != "floor":
            g.contype, g.conaffinity = 2, 2
    m = spec.compile()
    d = mujoco.MjData(m)
    per_pose = {}
    for k in knees:
        place(m, d, leg_pose(m, k), clip=False)
        pairs = set()
        for c in d.contact[:d.ncon]:
            if c.dist < -0.002:
                b = sorted(m.body(m.geom_bodyid[g]).name for g in c.geom)
                pairs.add(" / ".join(b))
        per_pose[round(float(np.degrees(k)))] = pairs
    always = set.intersection(*per_pose.values())
    return dict(always=sorted(always), new_by_knee_deg={k: sorted(v - always) for k, v in per_pose.items()})


def lying_test(m):
    d = mujoco.MjData(m)
    robot = [g for g in range(m.ngeom) if m.geom(g).name != "floor"]
    out = {}
    for name, quat in (("on_back", [np.cos(-np.pi / 4), 0, np.sin(-np.pi / 4), 0]),
                       ("on_front", [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]),
                       ("on_side", [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])):
        q = m.key("home").qpos.copy()
        q[2] = 0.3
        q[3:7] = quat
        mujoco.mj_resetData(m, d)
        d.qpos[:] = q
        d.ctrl[:] = q[7:]
        mujoco.mj_step(m, d, nstep=int(2.0 / m.opt.timestep))
        low = min(geom_world_vertices(m, d, g)[:, 2].min() for g in robot if m.geom_contype[g] | m.geom_conaffinity[g])
        ghosts = sorted({m.mesh(m.geom_dataid[g]).name for g in robot if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                         and geom_world_vertices(m, d, g)[:, 2].min() < -0.005})
        touching = sorted({m.body(m.geom_bodyid[g]).name for c in d.contact[:d.ncon] for g in c.geom if g in robot})
        out[name] = dict(trunk_com_z_mm=float(d.xipos[m.body("trunk_assembly").id][2] * 1000),
                         lowest_part_mm=float(low * 1000), sinks_through_floor=bool(low < -0.02 or d.xipos[m.body("trunk_assembly").id][2] < 0.02), touching=touching,
                         visual_parts_below_floor=ghosts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--out", help="JSON output (default body/checks.json, or checks_upstream.json)")
    a = ap.parse_args()
    path = Path(a.model) if a.model else (UPSTREAM if a.upstream else BUILT)
    m, _ = load(path)
    res = dict(model=str(path.relative_to(HERE.parent)) if path.is_relative_to(HERE.parent) else str(path),
               solver=dict(iterations=int(m.opt.iterations), ls_iterations=int(m.opt.ls_iterations), timestep=m.opt.timestep))
    res["stand_hold"] = sh = stand_hold(m)
    s = sh["summary"]
    print(f"STAND HOLD 5 s × {len(sh['trials'])} (joint noise σ=0.05 rad, tilt ±3°): max tilt {s['max_tilt']:.2f}°, "
          f"worst end tilt {s['worst_end_tilt']:.2f}°, worst height error {s['worst_height_err_mm']:.1f} mm, "
          f"worst drift over last 4 s {s['worst_drift_mm']:.2f} mm, base moved {s['worst_xy_drift_mm']:.1f} mm in total, {s['worst_slide_last4s_mm']:.2f} mm over the last 4 s, falls {s['falls']}")
    res["crouch"] = cr = crouch_sweep(m)
    print("\nCROUCH (feet flat, servos hold):")
    for r in cr:
        if not r["reachable"]:
            print(f"  knee cmd {r['knee_cmd_deg']:5.0f}°: beyond the knee joint limit")
            continue
        print(f"  knee cmd {r['knee_cmd_deg']:5.0f}° (reached {r['knee_actual_deg']:4.0f}°): "
              f"base {r['base_mm']:4.0f} mm, trunk bottom {r['trunk_bottom_mm']:4.0f} mm, tilt {r['tilt']:4.1f}°, "
              f"knee torque {r['knee_torque']:.2f} N·m ({r['knee_torque'] / STALL_7V4 * 100:.0f}% of 7.4 V stall)")
    res["sit_to_stand"] = st = sit_to_stand(m)
    print("\nSIT-TO-STAND (scripted ramp, deepest crouch -> home):")
    for r in st:
        print(f"  {r['seconds']:.1f} s from knee {r['from_knee_deg']:.0f}°: peak servo torque {r['peak_torque']:.2f} N·m "
              f"({r['peak_torque'] / STALL_7V4 * 100:.0f}% of 7.4 V stall{', AT LIMIT' if r['saturated'] else ''}), "
              f"end tilt {r['end_tilt']:.1f}°, height {r['end_height_mm']:.0f} mm")
    res["self_collision"] = sc = self_collision_scan(path)
    print("\nSELF-COLLISION (convex hulls): always touching:", "; ".join(sc["always"]) or "none")
    for k, v in sc["new_by_knee_deg"].items():
        print(f"  knee {k}°: {'; '.join(v) or 'clear'}")
    res["lying"] = ly = lying_test(m)
    print("\nLYING (dropped from 30 cm, 2 s):")
    for k, v in ly.items():
        print(f"  {k}: lowest colliding part {v['lowest_part_mm']:.0f} mm, trunk CoM {v['trunk_com_z_mm']:.0f} mm, "
              f"{'SINKS THROUGH FLOOR' if v['sinks_through_floor'] else 'rests on floor'}; touching: {', '.join(v['touching']) or 'nothing'}"
              + (f"; non-colliding parts below floor: {', '.join(v['visual_parts_below_floor'])}" if v['visual_parts_below_floor'] else ""))
    out = Path(a.out) if a.out else HERE / ("checks_upstream.json" if a.upstream else "checks.json")
    out.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
