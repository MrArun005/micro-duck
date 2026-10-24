"""Data for the live training page: the duck's 3D mesh per body, and replays of the latest brain.

    python body/live.py glb   --model model/duck_mid.xml --out DIR     # DIR/duck_mesh.json from v3/*.glb
    python body/live.py mesh  --model model/duck_mid.xml --out DIR     # same, straight from the model
    python body/live.py snap  --model model/duck_mid.xml --run runs/mid_stand --log mid_stand.log --out DIR

`mesh` writes every visual part in its body's frame (metres, MuJoCo z-up), so the page can move each
body with the simulated position and orientation. `snap` loads the newest checkpoint, plays one
10 s episode (with shoves, head turning and talking, as in training) and writes DIR/rollout.json
with per-frame body poses, plus the training curve parsed from the log.
"""
import argparse
import json
import re
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent


def mesh(model_path, out):
    m = mujoco.MjModel.from_xml_path(str(model_path))
    bodies = {}
    for g in range(m.ngeom):
        if m.geom_group[g] == 3 or m.geom_bodyid[g] == 0:
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            f = m.mesh_face[m.mesh_faceadr[mid]: m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            v, f = _ellipsoid(m.geom_size[g])
        else:
            continue
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        v = v @ R.reshape(3, 3).T + m.geom_pos[g]
        rgba = m.geom_rgba[g] if m.geom_matid[g] < 0 else m.mat_rgba[m.geom_matid[g]]
        name = m.body(m.geom_bodyid[g]).name
        bodies.setdefault(name, []).append(dict(
            color="#%02x%02x%02x" % tuple(int(255 * c) for c in rgba[:3]),
            v=np.round(v * 1e4).astype(int).ravel().tolist(), f=f.ravel().tolist()))
    names = [m.body(b).name for b in range(1, m.nbody)]
    (out / "duck_mesh.json").write_text(json.dumps(dict(scale=1e-4, bodies=names, parts=bodies),
                                                   separators=(",", ":")))
    return names


def glb_mesh(model_path, out, glb=HERE / "v3" / "duck_mesh_v3_head60_jaw22.glb",
             src=HERE / "model" / "duck_v3_upload.xml", scale=0.7, pose=(("head_yaw", 60), ("jaw", 22))):
    """The owner's v3 GLB, cut back into moving parts.

    The GLB is every visual part of duck_v3_upload.xml fused into one mesh, posed (head 60°, jaw 22°),
    scaled 0.7 and recentred. Its vertices are in geom order, so each geom's block is found by count,
    moved back into its body's frame and scaled to `model_path`'s size. The old flat jaw plate is
    replaced by the bill (upper bill + lower jaw) from the mid model.
    """
    import struct
    b = glb.read_bytes()
    n = struct.unpack("<I", b[12:16])[0]
    j = json.loads(b[20:20 + n])
    base = 20 + n + 8

    def acc(i):
        a = j["accessors"][i]
        bv = j["bufferViews"][a["bufferView"]]
        dt = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[a["componentType"]]
        k = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[a["type"]]
        o = base + bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        return np.frombuffer(b[o:o + a["count"] * k * np.dtype(dt).itemsize], dt).reshape(-1, k)

    prim = j["meshes"][0]["primitives"][0]
    V, C = acc(prim["attributes"]["POSITION"]).astype(float), acc(prim["attributes"]["COLOR_0"])
    F = acc(prim["indices"]).reshape(-1, 3)

    m = mujoco.MjModel.from_xml_path(str(src))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for name, deg in pose:
        d.qpos[m.jnt_qposadr[m.joint(name).id]] = np.radians(deg)
    mujoco.mj_forward(m, d)
    blocks, start, posed = [], 0, []
    for g in range(m.ngeom):
        if m.geom_group[g] == 3:
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            nv = m.mesh_vertnum[m.geom_dataid[g]]
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + nv]
            posed.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and m.geom_bodyid[g] == m.body("jaw").id:
            nv = 8
            posed.append(None)
        else:
            continue
        blocks.append((g, start, start + nv))
        start += nv
    assert start == len(V), f"GLB has {len(V)} vertices, model parts {start}"
    # The export scaled by `scale` and recentred; recover that shift from the mesh blocks themselves.
    shift = np.mean(np.vstack([V[s:e] - p * scale for (g, s, e), p in zip(blocks, posed) if p is not None]), 0)

    mm = mujoco.MjModel.from_xml_path(str(model_path))
    size = scale if mm.nmesh and np.allclose(mm.mesh_scale[0], scale) else 1.0
    parts, err = {}, 0.0
    for (g, s, e), p in zip(blocks, posed):
        if p is None:
            continue                                                  # old jaw plate: replaced below
        world = (V[s:e] - shift) / scale                              # full-size, posed
        err = max(err, float(np.abs(world - p).max()))
        bid = m.geom_bodyid[g]
        local = (world - d.xpos[bid]) @ d.xmat[bid].reshape(3, 3)     # body frame, full size
        tri = F[(F >= s).all(1) & (F < e).all(1)] - s
        # The GLB's vertex colours are a flat grey, so colour each part like the model's material.
        rgba = m.geom_rgba[g] if m.geom_matid[g] < 0 else m.mat_rgba[m.geom_matid[g]]
        parts.setdefault(m.body(bid).name, []).append(dict(
            color="#%02x%02x%02x" % tuple(int(255 * c) for c in rgba[:3]),
            v=np.round(local * size * 1e4).astype(int).ravel().tolist(), f=tri.ravel().tolist()))
    for g in range(mm.ngeom):                                         # the bill, from the target model
        if mm.geom(g).name in ("upper_bill", "lower_bill"):
            v, f = _ellipsoid(mm.geom_size[g])
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, mm.geom_quat[g])
            v = v @ R.reshape(3, 3).T + mm.geom_pos[g]
            parts.setdefault(mm.body(mm.geom_bodyid[g]).name, []).append(dict(
                color="#ff9e1a", v=np.round(v * 1e4).astype(int).ravel().tolist(), f=f.ravel().tolist()))
    names = [mm.body(i).name for i in range(1, mm.nbody)]
    (out / "duck_mesh.json").write_text(json.dumps(dict(scale=1e-4, bodies=names, parts=parts, source=glb.name),
                                                   separators=(",", ":")))
    return err


def _ellipsoid(size, n=16):
    th, ph = np.meshgrid(np.linspace(0, np.pi, n), np.linspace(0, 2 * np.pi, 2 * n))
    v = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1).reshape(-1, 3) * size
    f = []
    for i in range(2 * n - 1):
        for j in range(n - 1):
            a, b = i * n + j, (i + 1) * n + j
            f += [[a, b, a + 1], [a + 1, b, b + 1]]
    return v, np.array(f)


def curve(log):
    """[(steps, mean episode reward)] from the SB3 log: each block prints the reward, then the step count."""
    pts, rew = [], None
    for line in Path(log).read_text(errors="ignore").splitlines():
        if "ep_rew_mean" in line:
            rew = float(line.split("|")[2])
        elif "total_timesteps" in line and rew is not None:
            pts.append([int(line.split("|")[2]), round(rew, 2)])
            rew = None
    return pts


def snap(model_path, run, log, out, seconds=None, fps=25, task="stand", skills=False):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from env import DuckEnv
    from skills_env import SECONDS, SkillEnv

    ckpts = sorted((run / "checkpoints").glob("policy_*_steps.zip"), key=lambda p: int(re.findall(r"\d+", p.name)[0]))
    final = run / "policy.zip"
    if final.exists() and (not ckpts or final.stat().st_mtime > ckpts[-1].stat().st_mtime):
        ckpt, stats, steps = final, run / "vecnormalize.pkl", None
    elif ckpts:
        ckpt = ckpts[-1]
        steps = int(re.findall(r"\d+", ckpt.name)[0])
        stats = ckpt.with_name(f"policy_vecnormalize_{steps}_steps.pkl")
    else:
        ckpt = None

    cls = SkillEnv if skills else DuckEnv
    env = cls(task=task, randomize=True, seed=7, model_path=str(model_path), max_seconds=seconds or SECONDS.get(task, 10))
    m, d = env.model, env.data
    obs, _ = env.reset(seed=7)
    if ckpt:
        venv = VecNormalize.load(str(stats), DummyVecEnv([lambda: env]))
        venv.training = False
        policy = PPO.load(ckpt, device="cpu")
    every = max(1, int(round(50 / fps)))
    frames, falls, t, done, info = [], 0, 0, False, {}
    while not done:
        if ckpt:
            a, _ = policy.predict(venv.normalize_obs(obs[None]), deterministic=True)
            a = a[0]
        else:
            a = np.zeros(env.action_space.shape)
        obs, r, term, trunc, info = env.step(a)
        if t % every == 0:
            frames.append(np.concatenate([d.xpos[1:], d.xquat[1:]], 1).round(5).ravel().tolist())
        t += 1
        done = term or trunc
    res = dict(fps=fps, seconds=t / 50, fell=bool(term), steps=steps if ckpt else 0, task=task,
               success=bool(info.get("success")),
               checkpoint=ckpt.name if ckpt else None, frames=frames,
               bodies=[m.body(b).name for b in range(1, m.nbody)],
               curve=curve(log) if log and Path(log).exists() else [])
    (out / "rollout.json").write_text(json.dumps(res, separators=(",", ":")))
    return res


def latest_brain(run):
    run = Path(run)
    final = run / "policy.zip"
    ckpts = sorted((run / "checkpoints").glob("policy_*_steps.zip"), key=lambda p: int(re.findall(r"\d+", p.name)[0]))
    if final.exists() and (not ckpts or final.stat().st_mtime >= ckpts[-1].stat().st_mtime):
        return final, run / "vecnormalize.pkl", "final"
    if ckpts:
        n = int(re.findall(r"\d+", ckpts[-1].name)[0])
        return ckpts[-1], ckpts[-1].with_name(f"policy_vecnormalize_{n}_steps.pkl"), n
    return None, None, 0


def test_all(out, episodes=5, only=None):
    """Exam every skill with its newest brain (or none yet: servos hold the home pose); save a replay of each."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from env import DuckEnv
    from skills_env import SECONDS, SKILL_TASKS, SkillEnv

    results = json.loads((out / "tests.json").read_text()) if (out / "tests.json").exists() else {}
    for task in ([only] if only else SKILL_TASKS):
        run = HERE / "runs" / f"mid_{task}"
        ckpt, stats, steps = latest_brain(run)
        if task == "stand" and ckpt is None:          # the first stand run, on the original layout
            ckpt, stats, steps = latest_brain(HERE / "runs" / "mid_stand_v1")
            cls, model_path, layout = DuckEnv, HERE / "model" / "duck_mid_knee90.xml", "first stand layout"
        else:
            cls, model_path, layout = SkillEnv, HERE / "model" / "duck_mid.xml", "skills layout"
        wins, rewards, frames_ep = 0, [], None
        policy = PPO.load(ckpt, device="cpu") if ckpt else None
        for ep in range(episodes):
            env = cls(task=task, randomize=True, seed=100 + ep, model_path=str(model_path), max_seconds=SECONDS[task])
            m, d = env.model, env.data
            obs, _ = env.reset(seed=100 + ep)
            if policy:
                venv = VecNormalize.load(str(stats), DummyVecEnv([lambda: env]))
                venv.training = False
            frames, total, done, t = [], 0.0, False, 0
            while not done:
                a = policy.predict(venv.normalize_obs(obs[None]), deterministic=True)[0][0] if policy else np.zeros(env.action_space.shape)
                obs, r, term, trunc, info = env.step(a)
                total += r
                if ep == 0 and t % 2 == 0:
                    frames.append(np.concatenate([d.xpos[1:], d.xquat[1:]], 1).round(5).ravel().tolist())
                t += 1
                done = term or trunc
            wins += info["success"]
            rewards.append(total)
            if ep == 0:
                frames_ep = dict(fps=25, seconds=t / 50, fell=bool(term), success=bool(info["success"]), task=task,
                                 steps=steps, checkpoint=ckpt.name if ckpt else None, frames=frames,
                                 bodies=[m.body(b).name for b in range(1, m.nbody)])
        (out / f"rollout_{task}.json").write_text(json.dumps(frames_ep, separators=(",", ":")))
        prev = results.get(task, {})
        results[task] = dict(success=wins / episodes, episodes=episodes, mean_reward=round(float(np.mean(rewards)), 1),
                             brain=(f"{steps:,} steps" if isinstance(steps, int) and steps else ("final" if ckpt else "none yet")),
                             layout=layout, best=prev.get("best") if prev.get("layout") == layout else None)
        # Keep the best checkpoint, not just the last: skills can appear and vanish between checkpoints.
        best = results[task]["best"]
        score = (results[task]["success"], results[task]["mean_reward"])
        if ckpt and layout == "skills layout" and (best is None or score > (best["success"], best["mean_reward"])):
            import shutil
            dst = run / "best"
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(ckpt, dst / "policy.zip")
            shutil.copy(stats, dst / "vecnormalize.pkl")
            results[task]["best"] = dict(success=score[0], mean_reward=score[1], brain=results[task]["brain"])
        print(f"{task:13s} {wins}/{episodes} succeeded, reward {np.mean(rewards):6.1f}, brain {results[task]['brain']}")
    (out / "tests.json").write_text(json.dumps(results))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["mesh", "glb", "snap", "test"])
    ap.add_argument("--model", default="model/duck_mid.xml")
    ap.add_argument("--run", default="runs/mid_stand")
    ap.add_argument("--log", default="mid_stand.log")
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="stand")
    ap.add_argument("--skills", action="store_true")
    ap.add_argument("--only", help="test: just this skill")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = (Path.cwd() / a.model).resolve()
    if a.what == "test":
        test_all(out, only=a.only)
    elif a.what == "glb":
        print(f"owner's GLB split into parts; max mismatch vs the model {glb_mesh(model, out) * 1000:.3f} mm")
    elif a.what == "mesh":
        print(len(mesh(model, out)), "bodies")
    else:
        r = snap(model, (Path.cwd() / a.run).resolve(), a.log, out, task=a.task, skills=a.skills)
        print(r["checkpoint"], "fell" if r["fell"] else "stayed up", len(r["frames"]), "frames", len(r["curve"]), "curve points")
