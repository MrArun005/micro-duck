"""Train the duck body with PPO (same recipe as train/train.py for Bolt).

    python body/train.py --quick                       # smoke test: a few thousand steps
    python body/train.py --task stand --steps 5000000  # one skill
    python body/train.py --steps 60000000 --smooth-after 20000000   # all four tasks
    python body/train.py --resume runs/duck/checkpoints/policy_5000000_steps.zip
    python body/train.py --model model/duck_mid.xml --task stand --name mid_stand   # the 0.7x v3 duck

Saves runs/<name>/policy.zip + vecnormalize.pkl (bake the normalizer into the ONNX export:
LESSONS.md §3). Watch: tensorboard --logdir body/runs
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from env import TASKS, DuckEnv
from skills_env import SKILL_TASKS, SkillEnv

HERE = Path(__file__).resolve().parent


class SmoothCurriculum(BaseCallback):
    """Turn smoothness penalties on only after the skill exists (LESSONS §4)."""

    def __init__(self, after, ramp=5_000_000, weight=1.0):
        super().__init__()
        self.after, self.ramp, self.weight = after, ramp, weight

    def _on_step(self):
        if self.after is not None and self.num_timesteps >= self.after:
            w = self.weight * min(1.0, (self.num_timesteps - self.after) / self.ramp)
            self.training_env.set_attr("smooth_weight", w)
        return True


def exam(model, venv, tasks, episodes=4, seed=123, model_path=None, env_cls=DuckEnv):
    """Deterministic rollouts per task: success rate at the end of the episode, mean reward."""
    out = {}
    for task in tasks:
        env = env_cls(task=task, randomize=True, seed=seed, model_path=model_path)
        wins, rewards = 0, []
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            total, done = 0.0, False
            while not done:
                o = venv.normalize_obs(obs[None]) if isinstance(venv, VecNormalize) else obs[None]
                a, _ = model.predict(o, deterministic=True)
                obs, r, term, trunc, info = env.step(a[0])
                total += r
                done = term or trunc
            wins += info["success"]
            rewards.append(total)
        out[task] = dict(success=wins / episodes, mean_reward=float(np.mean(rewards)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60_000_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--task", choices=sorted(set(TASKS) | set(SKILL_TASKS)), help="train one task only (default: all, mixed)")
    ap.add_argument("--skills", action="store_true", help="all-skills layout (skills_env.py): run, sit, lie, jump, roll …")
    ap.add_argument("--name", default="duck")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smooth-after", type=int, default=None, help="step at which smoothness penalties start")
    ap.add_argument("--quick", action="store_true", help="smoke test: 4096 steps, 2 envs, tiny exam")
    ap.add_argument("--resume")
    ap.add_argument("--log-std", type=float, default=-2.3, help="initial log std of the exploration noise: -2.3 is about 0.1 rad")
    ap.add_argument("--ckpt-every", type=int, default=1_000_000, help="save a checkpoint every N steps")
    ap.add_argument("--model", help="model XML (default model/duck.xml), e.g. model/duck_mid.xml")
    a = ap.parse_args()
    if a.quick:
        a.steps, a.envs, a.name = 4096, 2, a.name + "_smoke"

    out = HERE / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    model_path = str((Path.cwd() / a.model).resolve()) if a.model else None
    kw = dict(task=a.task, model_path=model_path)
    vec_cls = DummyVecEnv if a.quick else SubprocVecEnv
    env_cls = SkillEnv if a.skills else DuckEnv
    venv = make_vec_env(env_cls, n_envs=a.envs, seed=a.seed, vec_env_cls=vec_cls, env_kwargs=kw)
    if a.resume:
        ckpt = Path(a.resume)
        stats = ckpt.with_name("vecnormalize.pkl")
        if not stats.exists():
            stats = ckpt.with_name(ckpt.name.replace("policy_", "policy_vecnormalize_", 1)).with_suffix(".pkl")
        venv = VecNormalize.load(str(stats), venv)
        model = PPO.load(ckpt, env=venv, device="cpu", tensorboard_log=str(out.parent))
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        model = PPO("MlpPolicy", venv, n_steps=1024 if a.quick else 512, batch_size=1024 if a.quick else 4096,
                    n_epochs=5, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
                    max_grad_norm=1.0, target_kl=0.03,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=torch.nn.ELU,
                                       log_std_init=a.log_std),
                    tensorboard_log=str(out.parent), seed=a.seed, device="cpu", verbose=1)
    cbs = [CheckpointCallback(max(a.ckpt_every // a.envs, 1), str(out / "checkpoints"), name_prefix="policy",
                              save_vecnormalize=True), SmoothCurriculum(a.smooth_after)]
    t0 = time.time()
    model.learn(total_timesteps=a.steps - model.num_timesteps, callback=cbs, tb_log_name=a.name,
                reset_num_timesteps=not a.resume, progress_bar=False)
    secs = time.time() - t0
    model.save(out / "policy.zip")
    venv.save(str(out / "vecnormalize.pkl"))
    venv.training = False
    tasks = [a.task] if a.task else (SKILL_TASKS if a.skills else TASKS)
    res = dict(steps=int(model.num_timesteps), seconds=secs, steps_per_s=model.num_timesteps / secs,
               model=a.model or "model/duck.xml",
               exam=exam(model, venv, tasks, episodes=1 if a.quick else 8, model_path=model_path, env_cls=env_cls))
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
