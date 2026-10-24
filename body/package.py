"""Export a trained skill brain of the mid duck to ONNX, for the robot's onboard computer.

    python body/package.py --skill stand          # models/mid-stand/policy.onnx + model.json + MODEL.md
    python body/package.py --all                  # every models/mid-*/ that has a policy.zip
    python body/package.py --skill sit --from body/runs/mid_sit/best   # brain from another folder
    python body/package.py --all --models /tmp/models                      # another models root

One graph does everything the training loop did around the network (LESSONS §3):

    raw obs (60) → (obs − mean) / sqrt(var + eps) → clip ±clip_obs → actor MLP → action mean → clip ±1

mean / var / clip_obs / eps come from the VecNormalize statistics saved next to the policy, so the robot
feeds raw sensor values. The export is checked against the stable-baselines3 model on ~1000
observations recorded from SkillEnv rollouts (randomized physics, sensor noise) and the worst
difference is printed and written to model.json.

model.json holds everything the onboard runtime (runtime.py) needs besides the ONNX file: the
observation layout, servo names in order, which servos the brain drives, action scale, home pose,
servo ranges and the control rate.
"""
import argparse
import datetime as dt
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from skills_env import GAIT_HZ, QUIET, SKILL_COMMANDS, SKILL_TASKS, TASK_COMMAND, SkillEnv  # noqa: E402

ROOT = HERE.parent
MODELS = ROOT / "models"
ROBOT_XML = HERE / "model" / "duck_mid.xml"
CTRL_HZ = 50


def obs_layout(n_servos, n_policy):
    return [
        ("gravity_in_imu_frame", 3, "unit vector, from the IMU orientation"),
        ("gyro_x0.25", 3, "IMU angular velocity, rad/s × 0.25"),
        ("accelerometer_div_9.81", 3, "IMU linear acceleration, m/s² / 9.81"),
        ("joint_pos_minus_home", n_servos, "every servo, rad, minus the home pose"),
        ("joint_vel_x0.05", n_servos, "every servo, rad/s × 0.05"),
        ("last_action", n_policy, "the brain's previous output, after clipping to ±1"),
        ("command_one_hot", len(SKILL_COMMANDS), "[" + ", ".join(SKILL_COMMANDS) + "]"),
        ("velocity_command", 3, "vx, vy (m/s), yaw rate (rad/s); zeros unless walk/run"),
        ("gait_clock", 2, "sin, cos of 2π·f·t; zeros unless walk (1.8 Hz) / run (2.6 Hz)"),
        ("foot_switches", 2, "left, right sole on the floor (0/1)"),
    ]


class OnnxBrain(torch.nn.Module):
    """VecNormalize + deterministic actor + clip, exactly what predict(normalize_obs(obs)) does."""

    def __init__(self, model, norm):
        super().__init__()
        rms = norm.obs_rms
        self.register_buffer("mean", torch.tensor(rms.mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(np.sqrt(rms.var + norm.epsilon), dtype=torch.float32))
        self.clip = float(norm.clip_obs)
        p = model.policy
        self.extract, self.actor, self.action_net = p.features_extractor, p.mlp_extractor.policy_net, p.action_net

    def forward(self, obs):
        x = torch.clamp((obs - self.mean) / self.std, -self.clip, self.clip)
        return torch.clamp(self.action_net(self.actor(self.extract(x))), -1.0, 1.0)


def load_brain(src):
    from stable_baselines3 import PPO
    model = PPO.load(src / "policy.zip", device="cpu")
    with open(src / "vecnormalize.pkl", "rb") as f:
        norm = pickle.load(f)          # VecNormalize without its venv: obs_rms, clip_obs, epsilon
    norm.training = False
    return model, norm


def sb3_act(model, norm, obs):
    """The reference: what train.py's exam does (normalize_obs, then deterministic predict)."""
    o = np.clip((obs - norm.obs_rms.mean) / np.sqrt(norm.obs_rms.var + norm.epsilon), -norm.clip_obs, norm.clip_obs)
    a, _ = model.predict(o.astype(np.float32), deterministic=True)
    return np.clip(a, -1, 1)


def export(model, norm, path, n_obs):
    net = OnnxBrain(model, norm).eval()
    dummy = torch.zeros(1, n_obs)
    kw = dict(input_names=["obs"], output_names=["action"], dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}})
    try:
        torch.onnx.export(net, (dummy,), str(path), dynamo=False, **kw)
    except Exception as e:                               # newer torch: only the dynamo exporter
        print(f"legacy exporter failed ({e}); using dynamo")
        torch.onnx.export(net, (dummy,), str(path), dynamo=True, **kw)


def record_obs(model, norm, task, samples=1000, seed=7):
    """Raw observations from SkillEnv rollouts of this brain (randomized physics and sensor noise)."""
    env = SkillEnv(task=task, randomize=True, seed=seed, model_path=str(ROBOT_XML))
    obs, _ = env.reset(seed=seed)
    out, ep = [], 0
    while len(out) < samples:
        out.append(obs)
        obs, _, term, trunc, _ = env.step(sb3_act(model, norm, obs[None])[0])
        if term or trunc:
            ep += 1
            obs, _ = env.reset(seed=seed + ep)
    return np.asarray(out, np.float32), env


def check(model, norm, path, obs):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"obs": obs})[0]
    ref = sb3_act(model, norm, obs)
    return float(np.abs(got - ref).max())


def read_summary(folder, skill):
    """Steps and test score from summary.json (train.py's format, or the hand-written 'best' one)."""
    p = folder / "summary.json"
    if not p.exists():
        return None, None
    s = json.loads(p.read_text())
    if "exam" in s and skill in s["exam"]:
        e = s["exam"][skill]
        return s.get("steps"), f"{e['success']:.0%} success, mean reward {e['mean_reward']:.1f} (train.py exam)"
    if "best" in s:
        b = s["best"]
        note = f"; {s['note']}" if s.get("note") else ""
        return None, f"{b.get('success', 0):.0%} success, mean reward {b.get('mean_reward', 0):.1f} ({b.get('brain', '')}{note})"
    return s.get("steps"), None


def robot_facts(env):
    m = env.model
    names = [m.actuator(i).name for i in range(m.nu)]
    return dict(
        robot="body/model/duck_mid.xml", control_hz=CTRL_HZ,
        servo_names=names, policy_servos=[names[i] for i in env.pol],
        external_servos=[names[i] for i in range(m.nu) if env.ext[i]],
        home=np.round(env.home, 6).tolist(), action_scale=env.action_scale.tolist(),
        ctrl_low=np.round(env.lo, 6).tolist(), ctrl_high=np.round(env.hi, 6).tolist(),
        size_L=round(float(env.L), 6), stand_height=round(float(env.stand_height), 6),
        commands=SKILL_COMMANDS, gait_hz=GAIT_HZ, quiet_skills=list(QUIET),
        head_yaw_rate=2.0, obs_noise_in_training=0.01,
    )


def model_card(f, onnx_err, score):
    rows = "\n".join(f"| {i} | {n} | {s} | {d} |" for i, (n, s, d) in enumerate(f["obs_layout_rows"]))
    servos = "\n".join(f"| {i} | {n} | {h:+.3f} | {lo:+.3f} … {hi:+.3f} | {sc if sc is not None else 'external'} |"
                       for i, (n, h, lo, hi, sc) in enumerate(zip(f["servo_names"], f["home"], f["ctrl_low"],
                                                                  f["ctrl_high"], f["scale_by_servo"])))
    steps = f"{f['steps']:,}" if f.get("steps") else "unknown"
    return f"""# mid-{f['skill']}

Skill brain **{f['skill']}** for the mid duck (`{f['robot']}`), trained with PPO in MuJoCo
(`body/train.py --skills --task {f['skill']}`). Command it was trained under: `{f['command']}`.
Packaged {f['packaged']}.

- Training steps: **{steps}** (about {(f['steps'] or 0) / CTRL_HZ / 3600:.1f} h of simulated duck time)
- Test score: {score or 'no summary.json'}
- ONNX vs stable-baselines3 on {f['onnx_check_samples']} observations from SkillEnv rollouts: max abs
  difference **{onnx_err:.1e}**

## Files

| File | What |
|---|---|
| `policy.onnx` | The brain. Input `obs` (N×{f['obs_size']} **raw** values), output `action` (N×{f['action_size']}, each −1…1). Normalizer built in. |
| `model.json` | The same facts, machine-readable (used by `body/runtime.py`) |
| `policy.zip` + `vecnormalize.pkl` | To continue training |

## Every 1/{CTRL_HZ} s ({CTRL_HZ} Hz)

1. Build the {f['obs_size']} inputs below from the sensors (no normalization, no filtering).
2. `action = brain(obs)`.
3. Servo targets = home; policy servos += action_scale × action; head_yaw / jaw from the personality
   layer (look, talk); clip to the servo range. No smoothing: the brain was trained without it.
4. Remember `action` for the next observation's `last_action`.

## Observation (raw)

| # | Name | Size | Meaning |
|---|---|---|---|
{rows}

## Servos (observation order; the action covers the policy servos in this order, skipping external ones)

| # | Servo | Home (rad) | Range (rad) | Action scale (rad per unit) |
|---|---|---|---|---|
{servos}

```python
import onnxruntime as ort
brain = ort.InferenceSession("policy.onnx")
action = brain.run(None, {{"obs": obs[None].astype("float32")}})[0][0]
```
"""


def package(skill, src=None, samples=1000, models=MODELS):
    if skill not in SKILL_TASKS:
        raise SystemExit(f"unknown skill {skill!r}; one of {SKILL_TASKS}")
    out = Path(models) / f"mid-{skill}"
    src = Path(src) if src else out
    out.mkdir(parents=True, exist_ok=True)
    if src.resolve() != out.resolve():
        import shutil
        for name in ("policy.zip", "vecnormalize.pkl", "summary.json"):
            if (src / name).exists():
                shutil.copy(src / name, out / name)
    model, norm = load_brain(out)
    obs, env = record_obs(model, norm, skill, samples)
    n_obs, n_act = obs.shape[1], len(env.pol)
    assert norm.obs_rms.mean.shape == (n_obs,), f"normalizer has {norm.obs_rms.mean.shape}, env obs {n_obs}"
    export(model, norm, out / "policy.onnx", n_obs)
    err = check(model, norm, out / "policy.onnx", obs)
    print(f"[{skill}] ONNX vs SB3 on {len(obs)} SkillEnv observations: max abs diff {err:.2e}")

    steps, score = read_summary(out, skill)
    steps = steps or int(model.num_timesteps)
    facts = robot_facts(env)
    layout = obs_layout(len(facts["servo_names"]), n_act)
    scale = dict(zip(facts["policy_servos"], facts["action_scale"]))
    facts.update(skill=skill, command=TASK_COMMAND[skill], steps=steps, test_score=score,
                 packaged=dt.date.today().isoformat(), obs_size=n_obs, action_size=n_act,
                 obs_layout=[dict(name=n, size=s, meaning=d) for n, s, d in layout],
                 normalizer=dict(clip_obs=float(norm.clip_obs), epsilon=float(norm.epsilon)),
                 onnx_max_difference=err, onnx_check_samples=len(obs))
    (out / "model.json").write_text(json.dumps(facts, indent=1))
    card = dict(facts, obs_layout_rows=layout, scale_by_servo=[scale.get(n) for n in facts["servo_names"]])
    (out / "MODEL.md").write_text(model_card(card, err, score))
    print(f"[{skill}] wrote {out / 'policy.onnx'}, model.json, MODEL.md")
    return err


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--skill", help="skill / task name, e.g. stand, sit, sit_to_stand, walk")
    g.add_argument("--all", action="store_true", help="every models/mid-*/ with a policy.zip")
    ap.add_argument("--from", dest="src", help="take policy.zip + vecnormalize.pkl from this folder")
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--models", default=str(MODELS), help="models root (default: repo models/)")
    a = ap.parse_args()
    torch.set_num_threads(1)
    if a.all:
        skills = sorted(p.name[4:] for p in Path(a.models).glob("mid-*") if (p / "policy.zip").exists())
        results = {s: package(s, samples=a.samples, models=a.models) for s in skills}
        print("summary:", ", ".join(f"{s} {e:.1e}" for s, e in results.items()))
    else:
        package(a.skill, a.src, a.samples, a.models)


if __name__ == "__main__":
    main()
